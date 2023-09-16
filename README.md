# PyTorch Training Pipeline with Weights and Biases (W&B)
> created by Zihan Pan -- Sep.17 2023 
<img width="1720" alt="PipeLineCover" src="https://github.com/RiptidePzh/ClassificationPipeline/assets/85790664/778ee025-3ddd-449a-b61e-c9827e05d361">

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Contributing](#contributing)
- [SSH Error](#ssh-error)

## Overview

This PyTorch training pipeline is designed to streamline the process of training deep learning models. It includes components for data loading, model creation, training, and evaluation. Additionally, it integrates with W&B for comprehensive experiment tracking and visualization of training metrics.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. Install the required Python packages. It's recommended to create a virtual environment before installing dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Pre-trained Model Specification
The code also loads a pre-trained deep learning model as the foundation for the project:
```python
# Load a pretrained model
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
```
In this code:
- `from torchvision.models import resnet50` imports the `ResNet-50` model architecture from the `torchvision` library. 
- `model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)` loads the model to GPU or CPU specified earlier using the `device` variable.
- By changing the model variable you should be able to train the model from different starting point

### GPU Backend
The choice of GPU or CPU depends on various factors, including the availability of GPU hardware.


For **MacOS**, MPS stands for the GPU device.
```python
# Call GPU backend
device = torch.device("mps" if torch.has_mps else "cpu")
```

For **NVidia**, you may want to change the code to use 'cuda' instead:
```python
# Call GPU backend
device = torch.device("cuda" if torch.has_mps else "cpu")
```

### Configuration

Before using the training pipeline, you need to configure it. Modify the `config` dictionary to suit your specific project:

- `epochs`: The number of training epochs.
- `classes`: The number of classes in your dataset.
- `batch_size`: The batch size for training.
- `learning_rate`: The learning rate for the optimizer.
- `dataset_dir`: The path to your dataset directory.

### Training

To start training your deep learning model, follow these steps:

1. Ensure you have configured the `config` dictionary and have valid `dataset` in this format:
   - **DataSet**
     - **train**
       - label1
       - label2
       - label3
       - ...
     - **val**
       - label1
       - label2
       - label3
       - ...


2. Run the training pipeline by executing the `train.py` script:

   ```bash
   python train.py
   ```

3. This will initiate the training process. W&B will log and visualize training metrics, including loss, accuracy, precision, recall, and F1-score, in real-time.

4. Monitor the training progress in your W&B project dashboard.

### Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository on GitHub.

2. Clone your forked repository:

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   ```

3. Create a new branch for your contributions:

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. Make your changes, add your improvements, or fix bugs.

5. Commit your changes:

   ```bash
   git commit -m "Add new feature"
   ```

6. Push your changes to your forked repository:

   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a pull request on GitHub to merge your changes into the main project.

---
## Common Issues & Debugging
### SSH Error

To fix the `SSLCertVerificationError` issue while loading pre-treained model/dataset from an url, open terminal:
```
pip install --upgrade certifi
```
