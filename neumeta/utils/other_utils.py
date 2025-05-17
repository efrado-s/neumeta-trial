import random

import torch
import numpy as np

# Functions to handle (or parse) arguments


# Functions to handle random seed
def set_seed(seed_value=42):
    """Set the seed for generating random numbers for PyTorch and other libraries to ensure reproducibility.

    Args:
        seed_value (int, optional): The seed value. Defaults to 42 (a commonly used value in randomized algorithms requiring a seed).
    """
    print(f'Setting seed... {seed_value} for reproducibility')
    # Set the seed for generating random numbers in Python's random library.
    random.seed(seed_value)

    # Set the seed for generating random numbers in NumPy, which can also affect randomness in cases where PyTorch relies on NumPy.
    np.random.seed(seed_value)

    # Set the seed for generating random numbers in PyTorch. This affects the randomness of various PyTorch functions and classes.
    torch.manual_seed(seed_value)

    # If you are using CUDA, and want to generate random numbers on the GPU, you need to set the seed for CUDA as well.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        # For multi-GPU, if you are using more than one GPU.
        torch.cuda.manual_seed_all(seed_value)

        # Additionally, for even more deterministic behavior, you might need to set the following environment, though it may slow down the performance.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Calculating average
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset values"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        """Update the average of current values"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EMA:
    """Exponential Moving Average"""
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        self.set_shadow(model)

    def set_shadow(self, model):
        # Initialize the shadow weights with the model's weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def apply(self):
        # Backup the current model weights and set the model's weights to the shadow weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        # Restore the original model weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

    def update(self):
        # Update the shadow weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data


# Functions for saving and loading checkpoints
