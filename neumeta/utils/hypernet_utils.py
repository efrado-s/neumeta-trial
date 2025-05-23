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


# Functions to create hypernetwork and using it
def get_hypernetwork(args, number_param, device='cuda'):
    """
    Returns a hypernetwork model based on the specified hyper_model_type in the arguments.

    Args:
        args (argparse.Namespace): Arguments containing hyper_model_type and hyperparameters.
        number_param (int): Number of parameters to be generated by the hypernetwork.

    Returns:
        torch.nn.Module: Hypernetwork model based on the specified hyper_model_type.
    """
    hyper_model_type = args.hyper_model.get('type', 'mlp')
    print(f'Hyper model type: {hyper_model_type}')
    if hyper_model_type == 'mlp':
        hyper_model = NeRF_MLP_Compose(
            input_dim=args.hyper_model.input_dim,
            hidden_dim=args.hyper_model.hidden_dim,
            num_layers=args.hyper_model.num_layers,
            output_dim=args.hyper_model.output_dim,
            num_freqs=args.hyper_model.num_freqs,
            num_compose=number_param
        ).to(device)
    elif hyper_model_type == 'resmlp':
        print(f"Using scalar {args.hyper_model.get('scalar', 0.1)}")
        hyper_model = NeRF_ResMLP_Compose(
            input_dim=args.hyper_model.input_dim,
            hidden_dim=args.hyper_model.hidden_dim,
            num_layers=args.hyper_model.num_layers,
            output_dim=args.hyper_model.output_dim,
            num_freqs=args.hyper_model.num_freqs,
            scalar=args.hyper_model.get('scalar', 0.1),
            num_compose=number_param
        ).to(device)
    else:
        raise ValueError(f'Unsupported hyper_model_type: {hyper_model_type}')

    return hyper_model

def sample_weights(model, model_cls,
                   coords_tensor, keys_list, indices_list, size_list, 
                   key_mask, selected_keys=None,
                   device='cuda', large_batch_size=4096, NORM=1):
    """
    Samples weights from the model and updates the predicted_checkpoint using the batch of predicted weights.

    Args:
        model (nn.Module): The neural network model.
        model_cls (nn.Module): The neural network model class.
        coords_tensor (torch.Tensor): The coordinates tensor.
        keys_list (list): The list of keys.
        indices_list (list): The list of indices.
        selected_keys (list, optional): The list of selected keys. Defaults to None.
        device (str, optional): The device to use. Defaults to 'cuda'.
        large_batch_size (int, optional): The large batch size. Defaults to 4096.

    Returns:
        tuple: A tuple containing the updated model_cls and the list of predicted weights.
    """
    
    # Get selected keys (or neurons in each layer)
    if selected_keys is None:
        predicted_checkpoint = model_cls.learnable_parameter
    else:
        predicted_checkpoint = {k: v
                                for k, v in model_cls.learnable_parameter.items()
                                if k in selected_keys}
    
    # Sample a batch of coordinates
    coords_tensor = coords_tensor.to(device)
    layer_id = coords_tensor[:, 0].int()
    input_dim = coords_tensor[:, -1] 
    input_tensor = (coords_tensor/NORM)
    
    # Predict weights
    predicted_weights = model(input_tensor, layer_id=layer_id, input_dim=input_dim)

    selected_mask = sum([key_mask[k] for k in selected_keys]).bool()
    # Iterate over the keys that have been selected for processing
    for key in selected_keys:
        # Create a boolean mask based on the selected mask from key_mask dictionary.
        boolean_mask = key_mask[key][selected_mask].bool()

        # Check the size information for the current mask and proceed accordingly.
        if size_list[boolean_mask][0] == 4:
            # Extract height and width from the indices list
            height, width = indices_list[boolean_mask][0, 2:]

            # Extract the relevant weights based on the mask
            current_weights = predicted_weights[boolean_mask]
            total_weights = current_weights.size(-1)

            # Adjust weigths if the don't match the expected size (h*w)
            if height * width < total_weights:
                start_index = torch.div(total_weights, 2, rounding_mode='trunc') - torch.div(height * width, 2, rounding_mode='trunc')
                end_index = start_index + height * width
                current_weights = current_weights[:, start_index:end_index]

            # Reshape and assign the adjusted weights to the appropriate position in the checkpoint dictionary.
            predicted_checkpoint[key][indices_list[boolean_mask][:, 0], indices_list[boolean_mask][:, 1]] = current_weights.view(-1, height, width)
        
        elif size_list[boolean_mask][0] == 2: 
            # Directly assign the weights without reshaping
            predicted_checkpoint[key][indices_list[boolean_mask][:, 0], indices_list[boolean_mask][:, 1]] = predicted_weights[boolean_mask][:, 0]
        elif size_list[boolean_mask][0] == 1:
            predicted_checkpoint[key][indices_list[boolean_mask][:, 0]] = predicted_weights[boolean_mask][:, 0]

    # Replace weight of each neuron in the target model with newly predicted weights
    for name, param in model_cls.learnable_parameter.items():
        if name in predicted_checkpoint:
            param.data = predicted_checkpoint[name].data

    return model_cls, list(predicted_checkpoint.values())

# Functions for training
def sample_subset(coords_tensor, keys_list, indices_list, size_list, key_mask, ratio=0.5):
    """
    Samples a subset of the input data based on the given ratio.

    Args:
        coords_tensor (numpy.ndarray): The coordinates tensor.
        keys_list (list): The list of keys.
        indices_list (list): The list of indices.
        ratio (float): The ratio of the subset to the original data.

    Returns:
        tuple: A tuple containing the subset coordinates tensor, the subset keys list,
            the subset indices list, and the selected keys list.
    """
    # Return immediately if ratio is 1.0
    if ratio >= 1.0:
        return coords_tensor, keys_list, indices_list, size_list, np.unique(keys_list)
    assert len(coords_tensor) == len(keys_list) == len(indices_list)

    # Gets the index of unique coordinates based on ratio
    unique_keys = np.unique(keys_list)
    num_samples = int(len(unique_keys) * ratio)
    selected_keys = random.sample(list(unique_keys), k=num_samples)

    # Create key_mask based on selected keys
    if key_mask is not None:
        selected_mask = sum([key_mask[k] for k in selected_keys]).bool()

    # Get the subsets of coordinates based on ratio using selected_mask as filter
    subset_coords = coords_tensor[selected_mask]
    subset_keys = keys_list[selected_mask]
    subset_indices = indices_list[selected_mask]
    subset_size = size_list[selected_mask]

    return subset_coords, subset_keys, subset_indices, subset_size, selected_keys


# Function to create optimizer
def get_optimizer(args, hyper_model):
    criterion = torch.nn.CrossEntropyLoss()
    val_criterion = torch.nn.CrossEntropyLoss()

    # Get optimizer for training
    optimizer_name = args.training.get('optimizer', 'adamw')
    if optimizer_name == 'adamw':
        optimizer = AdamW(hyper_model.parameters(),
                          lr=args.training.learning_rate,
                          weight_decay=args.training.weight_decay)
    elif optimizer_name == 'adam':
        optimizer = Adam(hyper_model.parameters(),
                         lr=args.training.learning_rate,
                         weight_decay=args.training.weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(hyper_model.parameters(),
                                    lr=args.training.learning_rate,
                                    momentum=args.training.get('momentum', 0.9),
                                    weight_decay=args.training.weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(hyper_model.parameters(),
                                        lr=args.training.learning_rate,
                                        momentum=args.training.get('momentum', 0.9),
                                        weight_decay=args.training.weight_decay)
    elif optimizer_name == 'adagrad':
        optimizer = torch.optim.Adagrad(hyper_model.parameters(),
                                        lr=args.training.learning_rate,
                                        weight_decay=args.training.weight_decay)
    else:
        raise ValueError(f'Unknown optimizer name: {optimizer_name}')
    
    # Decides which step scheduler to use
    scheduler_name = args.training.get('scheduler', 'multistep')
    if scheduler_name == 'cosine':
        print(f'Using cosine scheduler, T_max: {args.training.T_max}')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.training.T_max)
    elif scheduler_name == 'multistep':
        scheduler = MultiStepLR(optimizer,
                                milestones=args.training.get('lr_steps', [args.experiment.num_epochs]),
                                gamma=0.1)

    return criterion, val_criterion, optimizer, scheduler


# Functions for evaluating or testing model
def sample_single_model(hyper_model, model, device='cuda', cfg=None):
    """Create one model from hypernetwork"""
    # Initialize a model to accumulate the weights over K samples
    coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model)
    key_mask = create_key_masks(keys_list=keys_list)
    
    # Sample weight from hypernetwork
    # use torch.no_grad() to prevent gradient changing
    with torch.no_grad():
        model, _ = sample_weights(hyper_model, model, 
                                  coords_tensor, keys_list, indices_list, size_list,
                                  key_mask, list(key_mask.keys()),
                                  device=device, NORM=cfg.dimensions.norm)
    model.eval()

    return model

def average_models(models):
    """
    Average the weights of multiple PyTorch models.
    
    Args:
    models (list of torch.nn.Module): List of models with the same architecture.
    
    Returns:
    torch.nn.Module: A model with the average weights.
    """
    # Validate if models list is not empty
    if not models:
        raise ValueError('No models to average.')
    
    # Create a copy of the first model which will be used to store the average weights
    averaged_model = copy.deepcopy(models[0])

    # Initialize a dict to hold the sum of all model parameters
    param_sum = dict()

    for model in models:
        for name, param in model.named_parameters():
            if name not in param_sum:
                # Initialize the sum for this parameter as a tensor filled with zeros with the same shape as the parameter
                param_sum[name] = torch.zeros_like(param.data)
            
            # Add the parameter
            param_sum[name] += param.data

    # Average the sum of parameters by the numbers of models
    for name in param_sum:
        param_sum[name] = param_sum[name] / len(models)

    # Update the averaged model with the new averaged weights
    for name, param in averaged_model.named_parameters():
        param.data = param_sum[name]
    
    return averaged_model

def sample_merge_model(hyper_model, model, args, K=50, device='cuda'):
    # Initialize a model to accumulate the weights over K samples
    hyper_model.eval()
    models = []

    for k in range(K):
        model_cls_temp = copy.deepcopy(model)
        model_cls_temp.to(device)

        # Sampling and merging weights
        coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model_cls_temp)
        key_mask = create_key_masks(keys_list=keys_list)
        if k > 0:
            coords_tensor = coords_tensor + (torch.rand_like(coords_tensor) - 0.5) * args.training.coordinate_noise
        model_cls_temp, _ = sample_weights(hyper_model, model,
                                           coords_tensor, keys_list, indices_list, size_list,
                                           key_mask, list(key_mask.keys()),
                                           device=device, NORM=args.dimensions.norm)
        models.append(model_cls_temp)

    # Average the weights of the models
    accumulated_model = average_models(models)

    accumulated_model.eval()
    return accumulated_model


# Functions for getting losses
def weighted_regression_loss(reconstructed_weights, gt_selected_weights, epsilon=1e-6):
    """
    Calculate reconstruct loss by calculating weighted regression loss, with each element in the ground truth weights
    contributing a weight proportional to its absolute value.

    Args:
    - reconstructed_weights (list of Tensors): List of newly reconstructed weight tensors.
    - gt_selected_weights (list of Tensors): List of original weights from pretrained network; ground truth weight tensors.
    Returns:
    - torch.Tensor: The mean of the weighted regression losses.
    """
    # Initialize an empty list to store individual loss
    losses = []

    # Iterate over the pairs of reconstructed and ground truth weights
    for w, w_gt in zip(reconstructed_weights, gt_selected_weights):
        # Calculate the weights as the absolute values of the ground truth weights
        element_weights = torch.abs(w_gt) + epsilon
        element_weights = element_weights / torch.max(element_weights)

        # Compute the MSE loss element-wise
        element_loss = (w - w_gt) ** 2

        # Apply the weights to the loss
        weighted_loss = element_loss * element_weights

        # Sum the weighted loss for the current pair and add it to the list
        losses.append(weighted_loss.mean())

    # Calculate the mean of the weighted losses
    reconstruct_loss = torch.mean(torch.stack(losses))

    return reconstruct_loss

def validate_single(model_cls, val_loader, criterion, args=None, device='cuda'):
    val_loss = 0.0
    preds = []
    gt = []

    model_cls.eval()

    with torch.no_grad():
        for x, target in tqdm(val_loader):
            
            x, target = x.to(device), target.to(device)
            predict = model_cls(x)
            pred = torch.argmax(predict, dim=-1)
            
            preds.append(pred)
            gt.append(target)

            loss = criterion(predict, target)
            val_loss += loss.item()
        
    return val_loss / len(val_loader), accuracy_score(torch.cat(gt).cpu().numpy(), torch.cat(preds).cpu().numpy())
