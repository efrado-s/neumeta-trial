

# Functions to handle (or parse) arguments


# Functions to handle random seed



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
