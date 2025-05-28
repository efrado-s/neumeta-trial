import os

import torch

from .resnet_cifar import *
from .lenet import *
from .densenet import *
from .utils import fuse_module
from smooth.permute import PermutationManager, compute_tv_loss_for_network

def create_model_cifar10(model_name, hidden_dim, path=None, smooth=False):
    """
    Create a model based on the specified name.

    :param model_name: String that specifies the model to use.
    :param path: Optional path for the model's weights.
    :return: The initialized model.
    """
    if model_name == 'ResNet20': # Add other models as you support them
        model = cifar10_resnet20(hidden_dim=hidden_dim)
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    fuse_module(model)  # Fuse convolutional and batch normalization layers

    # Load checkpoint if the weights exist already
    if path:
        if os.path.exists(path):
            print('Loading model from', path)
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            load_checkpoint(model, state_dict)

    # Smooth initial weights manifold before processing it
    if smooth:
        print('Smooth the parameters of the model')
        print(f'Old TV original model: {compute_tv_loss_for_network(model, lambda_tv=1.0).item()}')
        input_tensor = torch.randn(1, 3, 32, 32)
        permute_func = PermutationManager(model, input_tensor)
        permute_dict = permute_func.compute_permute_dict()
        model = permute_func.apply_permutations(permute_dict, ignored_keys=[
            ('conv1.weight', 'in_channels'),
            ('fc.weight', 'out_channels'),
            ('fc.bias', 'out_channels')
        ])
        print(f'Permuted TV original model: {compute_tv_loss_for_network(model, lambda_tv=1.0).item()}')
    
    return model


def create_mnist_model(model_name, hidden_dim, depths=None, path=None):
    if model_name == "LeNet":
        model = MnistNet(hidden_dim=hidden_dim)
    return model


def create_densenet_model(model_name, layers, growth, compression, bottleneck, drop_rate, hidden_dim, smooth=False,
                          path=None):
    if model_name == 'DenseNet':
        model = DenseNet3(layers, 10, growth, compression, bottleneck, drop_rate, hidden_dim)
    
    # Load checkpoint
    if path:
        if os.path.exists(path):
            print('Loading model from', path)
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            pretrained_dict = checkpoint['model_state_dict']
            
            model_dict = model.state_dict()

            filtered_dict = {k: v
                             for k, v in pretrained_dict.items()
                             if k in model_dict and v.shape == model_dict[k].shape}
            
            model_dict.update(filtered_dict)

            model.load_state_dict(model_dict)

    # Fuse module
    fuse_module(model)

    # Smooth initial weights manifold before processing it
    if smooth:
        print('Smooth the parameters of the model')
        print(f'Old TV original model: {compute_tv_loss_for_network(model, lambda_tv=1.0).item()}')
        input_tensor = torch.randn(1, 3, 32, 32)
        permute_func = PermutationManager(model, input_tensor)
        permute_dict = permute_func.compute_permute_dict()
        model = permute_func.apply_permutations(permute_dict, ignored_keys=[
            ('conv1.weight', 'in_channels'),
            ('fc.weight', 'out_channels'),
            ('fc.bias', 'out_channels')
        ])
        print(f'Permuted TV original model: {compute_tv_loss_for_network(model, lambda_tv=1.0).item()}')

    return model