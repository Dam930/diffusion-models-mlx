from typing import Any, List
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import mnist
import math
from tqdm import tqdm
import cv2

# Parameters
num_epochs = 10
lr = 0.0001
batch_size = 128
timesteps = 1000
dim_encoding = 10


# Definition of the input of the NN
@dataclass
class Input:
    image: mx.array
    number: mx.array
    t: mx.array


# Upsample class
class UpSample(nn.Module):
    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        x = mx.broadcast_to(
            x[:, :, None, :, None, :], (B, H, self.scale, W, self.scale, C)
        )
        x = x.reshape(B, H * self.scale, W * self.scale, C)
        return x


# Sinusoidal embedding for time encoding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, x):
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = mx.exp(mx.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = mx.concatenate((emb.sin(), emb.cos()), axis=-1)
        return emb


# Define InvertedResidualBlock in MLX
class InvertedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv_in = nn.Conv2d(channels, channels * dilation, 3, padding=1)
        self.conv_rb = nn.Conv2d(
            channels * dilation, channels, 3, padding=1
        )  # This should be two different conv but we don't have depth wise conv
        self.norm = nn.GroupNorm(channels, channels, pytorch_compatible=True)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.norm(x)
        y = self.conv_in(x)
        y = nn.gelu(y)
        y = self.conv_rb(y)
        return x + y


# Define UNet in MLX
class UNet(nn.Module):
    def __init__(
        self,
        mlp_time: int = [10, 32],
        mlp_number: int = [32, 32],
        channels: List[int] = [16, 32, 64, 128, 256],
        dilations: List[int] = [2, 2, 2, 2, 2],
        strides: List[int] = [1, 1, 2, 2, 2],
    ):
        super().__init__()
        assert len(channels) == len(strides) == len(dilations)
        self._strides = strides

        self.number_embeddings = nn.Embedding(10, mlp_number[0])

        self.levels_downsample = []
        self.time_mlps = []
        self.number_mlps = []
        self.levels_upsample = []
        in_channel = 1
        # Define encoder and decoder
        for n, (channel, dilation, stride) in enumerate(
            zip(channels, dilations, strides)
        ):
            # encoder
            self.levels_downsample.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, channel, 3, stride=stride, padding=1),
                    nn.GELU(),
                    InvertedResidualBlock(channel, dilation),
                )
            )
            # time net
            mlp = [
                [nn.Linear(idim, odim), nn.GELU()]
                for idim, odim in zip(mlp_time[:-1], mlp_time[1:])
            ]
            mlp = [item for sublist in mlp for item in sublist]
            mlp.append(nn.Linear(mlp_time[-1], channel * 2))
            self.time_mlps.append(nn.Sequential(*mlp))
            # number net
            mlp = [
                [nn.Linear(idim, odim), nn.GELU()]
                for idim, odim in zip(mlp_number[:-1], mlp_number[1:])
            ]
            mlp = [item for sublist in mlp for item in sublist]
            mlp.append(nn.Linear(mlp_number[-1], channel * 2))
            self.number_mlps.append(nn.Sequential(*mlp))

            # decoder
            channel_upsample = channel if n == len(channels) - 1 else channel * 2
            self.levels_upsample.append(
                nn.Sequential(
                    UpSample(stride),
                    nn.Conv2d(channel_upsample, in_channel, 3, stride=1, padding=1),
                    nn.GELU(),
                    InvertedResidualBlock(in_channel, dilation),
                )
            )

            in_channel = channel

        # Reverse the decoder from bottleneck to output
        self.levels_upsample.reverse()

    def __call__(self, input: Input) -> mx.array:
        x, number, t = input.image, input.number, input.t

        # Encoder
        features = []
        for level_downsample, time_mlp, number_mlp in zip(
            self.levels_downsample, self.time_mlps, self.number_mlps
        ):
            x = level_downsample(x)

            # Infuse time
            t_elab = time_mlp(t)
            t_elab = t_elab.reshape([t_elab.shape[0], 1, 1, t_elab.shape[1]])
            scale, offset = t_elab[..., : x.shape[-1]], t_elab[..., x.shape[-1] :]
            x = x * scale + offset

            # Infuse number
            n_elab = number_mlp(self.number_embeddings(number))
            n_elab = n_elab.reshape([n_elab.shape[0], 1, 1, n_elab.shape[1]])
            scale, offset = n_elab[..., : x.shape[-1]], n_elab[..., x.shape[-1] :]
            x = x * scale + offset

            features.append(x)

        # Decoder
        features = features[:-1]
        features.reverse()
        for n, level_upsample in enumerate(self.levels_upsample):
            x = level_upsample(x)
            if n < len(features):
                x = x[:, : features[n].shape[1], : features[n].shape[2]]
                x = mx.concatenate((x, features[n]), -1)
        return x


# Define the L2 loss
def loss_fn(model: nn.Module, x: mx.array, y: mx.array) -> mx.array:
    y_hat = model(x)
    return (y_hat - y).square().mean()


# Define the dataset
train_images = mx.array(mnist.train_images())
train_labels = mx.array(mnist.train_labels())
test_images = mx.array(mnist.test_images())
test_labels = mx.array(mnist.test_labels())


def cosine_variance_schedule(timesteps: int, epsilon: float = 0.008):
    steps = np.linspace(0, timesteps, num=timesteps + 1, dtype=np.float32)
    f_t = np.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5) ** 2
    betas = np.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
    return betas


t_enc = SinusoidalPosEmb(dim_encoding, timesteps)
betas = cosine_variance_schedule(timesteps)

alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = mx.array(np.sqrt(alphas_cumprod))
sqrt_one_minus_alphas_cumprod = mx.array(np.sqrt(1 - alphas_cumprod))


def batch_iterate(batch_size: int, X: np.array, y: np.array):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        # Get the original image
        x = X[ids].reshape((ids.shape[0], X.shape[1], X.shape[2], 1)) / 255.0 * 2 - 1

        # Sample noise
        noise = mx.random.normal(shape=x.shape)
        t = mx.random.randint(low=1, high=timesteps, shape=[ids.shape[0]])
        x = mx.multiply(
            x,
            sqrt_alphas_cumprod[t].reshape(sqrt_alphas_cumprod[t].shape[0], 1, 1, 1),
        ) + mx.multiply(
            noise,
            sqrt_one_minus_alphas_cumprod[t].reshape(
                sqrt_one_minus_alphas_cumprod[t].shape[0], 1, 1, 1
            ),
        )
        t = t_enc(t)

        # get class
        y_batch = y[ids]
        input_net = Input(x, y_batch, t)
        yield input_net, noise


if __name__ == "__main__":
    # Define the net
    unet = UNet()
    mx.eval(unet.parameters())

    # Instantiate the optimizer
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad_fn = nn.value_and_grad(unet, loss_fn)
    best_val = 0

    for e in range(num_epochs):
        pbar = tqdm(
            batch_iterate(batch_size, train_images, train_labels),
            total=len(train_images) // batch_size,
        )
        for x, y in pbar:
            loss, grads = loss_and_grad_fn(unet, x, y)

            # Update the optimizer state and model parameters
            # in a single call
            optimizer.update(unet, grads)

            # Force a graph evaluation
            mx.eval(unet.parameters(), optimizer.state)
            pbar.set_description(f"Epoch {e} - Loss: {loss.item():.4f}")

        test_losses = []
        for x, y in batch_iterate(batch_size, test_images, test_labels):
            test_losses.append(loss_fn(unet, x, y).item())

        test_loss = np.array(test_losses).mean()
        print(f"Epoch {e}: Test Loss {test_loss.item():.3f}")

        if e == 0 or test_loss < best_val:
            best_val = test_loss
            unet.save_weights("best")

    # Try to generate an image
    unet.load_weights("best.npz")
    number = 0
    shape = train_images[:1].shape + [1]
    img = mx.random.normal(shape=shape)
    for t in range(timesteps - 1, -1, -1):
        if t > 0:
            noise = mx.random.normal(shape=shape)
        else:
            noise = mx.zeros(shape)
        input_net = Input(img, mx.array([number]), t_enc(mx.array([t])))
        pred = unet(input_net)
        std = np.sqrt(
            (1 - alphas[t]) * (1.0 - alphas_cumprod[t - 1]) / (1.0 - alphas_cumprod[t])
        )
        img = mx.array(
            (
                (1.0 / np.sqrt(alphas[t]))
                * (
                    img
                    - (
                        mx.array(
                            (1 - alphas[t]) / sqrt_one_minus_alphas_cumprod[t] * pred
                        )
                    )
                )
            )
            + (noise * std)
        )
        img = mx.where(img > 1, 1, img)
        img = mx.where(img < -1, -1, img)

    img = np.array(img.tolist())
    img = np.array(img * 128 + 127, dtype=np.uint8)
    cv2.imwrite("out.png", img[0, :, :, 0])
