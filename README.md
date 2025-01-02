# Diffusion Models with MLX for Apple Silicon

This repository demonstrates how to implement a simple diffusion model using the MLX library on Apple Silicon. The model trains a UNet from scratch to generate MNIST digits.

## Features
- Implements a UNet architecture from scratch.
- Trains a diffusion model to generate MNIST digits.
- Optimized for Apple Silicon using the MLX library.

## Prerequisites
- macOS with Apple Silicon (M1/M2 series).
- Python 3.12 or later.
- Conda installed (for environment management).

## Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/diffusion-models-mlx.git
cd diffusion-models-mlx
```

### Step 2: Create a Conda Environment
```bash
conda create --name diffusion-env python=3.12 -y
conda activate diffusion-env
```

### Step 3: Install Dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### Step 4: Run the Training Script
Execute the training script to start training the UNet on the MNIST dataset:
```bash
python diffusion.py
```

## File Structure
```
.
├── diffusion.py         # Main script to train the diffusion model
├── LICENSE              # LICENSE information
├── requirements.txt     # List of Python dependencies
├── README.md            # Project documentation
```

## Results
After training, the model will generate MNIST digits. Generated sample will be saved in the `out.png` directory. The training progress can be monitored via terminal logs.

## Contributions
Contributions are welcome! Feel free to fork the repository, create issues, or submit pull requests to enhance the project.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
