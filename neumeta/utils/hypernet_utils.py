import random
import copy

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR

import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # To show progress bar

from neumeta.hypermodel import NeRF_MLP_Compose, NeRF_ResMLP_Compose


# Functions related to coordinates handling
def sample_coordinates(model_cls):
    """
    Sample coordinates for the given model_cls.

    Args:
        model_cls: The model class to sample coordinates for.

    Returns:
        A tuple containing:
        - coords_tensor: A tensor containing the sampled coordinates.
        - keys_list: A list of keys corresponding to each coordinate.
        - indices_list: A list of in_channel and out_channel indices.
    """
    checkpoint = model_cls.learnable_parameter

    coords_list = []  # List to store all coordinates
    keys_list = []  # List to store keys corresponding to each coordinate, key is the name of the layer in that layers
    indices_list = []  # List to store in_channel and out_channel indices
    size_list = []  # List to store the size of each channel

    layer_num = len(checkpoint)
    # Iterate over the new model's weight
    for i, (k, tensor) in enumerate(checkpoint.items()):
        # Handle 2D tensors (e.g., weight matrices)
        if len(tensor.shape) == 4:
            for in_channel in range(tensor.shape[0]):
                for out_channel in range(tensor.shape[1]):
                    coords = [i, in_channel, out_channel,  # specific index
                              layer_num, tensor.shape[0], tensor.shape[1]]  # Used to normalized
                    coords_list.append(coords)
                    keys_list.append(k)
                    indices_list.append((
                        in_channel, out_channel,
                        tensor.shape[2], tensor.shape[3]
                    ))
                    size_list.append(4)
        # Handle 2D matrices
        elif len(tensor.shape) == 2:
            for in_channel in range(tensor.shape[0]):
                for out_channel in range(tensor.shape[1]):
                    coords = [i, in_channel, out_channel, layer_num, 
                              tensor.shape[0], tensor.shape[1]]
                    coords_list.append(coords)
                    keys_list.append(k)
                    indices_list.append((
                        in_channel, out_channel, 
                        0, 0
                    ))
                    size_list.append(2)
        # Handle 1D tensors (e.g., biases)
        elif len(tensor.shape) == 1:
            for in_channel in range(tensor.shape[0]):
                coords = [i, in_channel, 0, layer_num, tensor.shape[0], 1]
                coords_list.append(coords)
                keys_list.append(k)
                indices_list.append((in_channel, 0, 0, 0))
                size_list.append(1)

    # Convert list of coordinates to tensor
    coords_tensor = torch.tensor(coords_list, dtype=torch.float32)
    indices_list = torch.tensor(indices_list, dtype=torch.int64)
    size_list = torch.tensor(size_list, dtype=torch.int64)
    keys_list = np.array(keys_list)

    return coords_tensor, keys_list, indices_list, size_list

def create_key_masks(keys_list):
    """
    Creates a dictionary of key masks for the given list of keys.

    Args:
    - keys_list (list): A list of keys for which to create masks.

    Returns:
    - key_mask_dict (dict): A dictionary of key masks, where each key corresponds to a tensor of zeros and ones.
    """
    unique_keys, key_inverse = np.unique(keys_list, return_inverse=True)
    key_mask_dict = {
        key: torch.tensor(key_inverse == i, dtype=torch.long)
        for i, key in enumerate(unique_keys)}
    return key_mask_dict

def shuffle_coordinates(coords_tensor, keys_list, indices_list, size_list):
    """
    Shuffle the coordinates tensor, keys list, and indices list randomly.

    Args:
        coords_tensor (numpy.ndarray): The coordinates tensor to shuffle.
        keys_list (list): The list of keys to shuffle in the same order as the coordinates tensor.
        indices_list (list): The list of indices to shuffle in the same order as the coordinates tensor.

    Returns:
        numpy.ndarray: The shuffled coordinates tensor.
        list: The shuffled list of keys.
        list: The shuffled list of indices.
    """
    # Create indexes to shuffle the tensors
    shuffled_indices = np.arange(len(coords_tensor))
    np.random.shuffle(shuffled_indices)

    # Create new version of shuffled tensors
    coords_tensor = coords_tensor[shuffled_indices]
    keys_list = keys_list[shuffled_indices]
    indices_list = indices_list[shuffled_indices]
    size_list = size_list[shuffled_indices]

    return coords_tensor, keys_list, indices_list, size_list

def shuffle_coordinates_all(dim_dict):
    """
    Shuffle the coordinates of all dimensions in the given range list.

    Args:
        dim_dict (dict): A dictionary containing the coordinates, keys, and indices for each dimension.

    Returns:
        dict: The updated dictionary with shuffled coordinates, keys, and indices for the specified dimensions.
    """
    for dim in dim_dict:
        (model_cls, coords_tensor, keys_list, indices_list, size_list, key_mask) = dim_dict[dim]
        # Somehow this is commented out in original code
        # coords_tensor, keys_list, indices_list, size_list = shuffle_coordinates(coords_tensor, keys_list, indices_list, size_list)
        key_mask = create_key_masks(keys_list)  # Reshuffle the mask for keys
        dim_dict[dim] = (model_cls, coords_tensor, keys_list, indices_list, size_list, key_mask)
    return dim_dict


# Function to create optimizer

# Function to find weight loss

# Functions to create hypernetwork and using it

# Functions for training
def sample_subset():
    return None
