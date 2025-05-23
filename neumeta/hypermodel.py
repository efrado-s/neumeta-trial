import torch.nn as nn
import math
import numpy as np
import torch

def weights_init_uniform_relu(module):
    """Initialize uniform weights for Linear layer"""
    
    classname = module.__class__.__name__  # Get name of layers

    # for every Linear layer in a model
    if classname.find('Linear') != -1:  
        torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))  # init weights for layer using kaiming_uniform
        # init uniform bias' values if layer has bias
        if module.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(module.bias, -bound, bound)


class LearnablePositionalEmbeddings(nn.Module):
    """
    Layers for creating learnable positional embedding 
    """
    def __init__(self, d_model, seq_len):
        super(LearnablePositionalEmbeddings, self).__init__()
        self.positional_embeddings = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x):
        return x + self.positional_embeddings
    

class PositionalEncoding(nn.Module):
    """Encode coordinates as specified in the paper"""
    def __init__(self, num_freqs = 10, input_dim = 5):
        super(PositionalEncoding, self).__init__()
        
        print("num_freqs: ", num_freqs, type(num_freqs))
        
        if isinstance(num_freqs, int):
            self.freqs_low = 0
            self.freqs_high = num_freqs
        elif len(num_freqs) == 2:
            self.freqs_low = num_freqs[0]
            self.freqs_high = num_freqs[1]
        else:
            raise ValueError("num_freqs should be either an integer or a list of length 2.")
        
        self.input_dim = input_dim

    def forward(self, x):

        out = [x]

        # Extract fourier features
        for i in range(self.freqs_low, self.freqs_high):
            freq = 2.0 ** i * np.pi
            for j in range(self.input_dim):
                # Create columns of Fourier encoded coordinates from the original columns
                out.append(torch.sin(freq * x[:, j].unsqueeze(-1)))
                out.append(torch.cos(freq * x[:, j].unsqueeze(-1)))
        # Combine each rectangular matrix by stacking them together orderly to the right side of the previous one
        # ex: X, Y matrix into [X Y] matrix
        return torch.cat(out, dim = -1)  
    

class NeRF_MLP_Residual_Scaled(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_freqs = None, num_layers = 4, scalar = 0.0):
        """
        Construct a residual scaled MPL for NeRF

        Args:
            input_dim (int): input dimension size; input size of the MLP.
            hidden_dim (int): hidden dimension size; input and output size of each layers inside the MLP.
            output_dim (int): output dimension size; output size of the MLP.
            num_freqs (int): values for creating Fourier encoded features
            num_layers (int): define the number of layers (excluding input and output layer) in the MLP.
            scalar (float): 

        """
        super(NeRF_MLP_Residual_Scaled, self).__init__()
        
        # Define the initial layer
        self.initial_layer = nn.Linear(input_dim, hidden_dim)

        # Create ModuleList for residual layers and scalars
        self.residual_blocks = nn.ModuleList()
        self.scalars = nn.ParameterList()

        # Adding residual blocks and their corresponding scalars
        for _ in range(num_layers - 1):
            self.residual_blocks.append(nn.Linear(hidden_dim, hidden_dim))
            self.scalars.append(nn.Parameter(torch.tensor(scalar), requires_grad=True))

        # Activation function
        self.act = nn.ReLU(inplace=True)

        # Define the output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.act(self.initial_layer(x))

        # Process x through each residual block
        for block, scale in zip(self.residual_blocks, self.scalars):  
            residual = x                    
            out = block(x)
            x = scale * self.act(out) + residual    
        
        x = self.output_layer(x)
        return x
    

class NeRF_MLP_Compose(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_freqs=10, num_layers=4, num_compose=4, normalizing_factor=1.0):
        """
        NeRF_MLP_Compose is a class that represents a composition of NeRF_MLP_Residual_Scaled models.

        Args:
            input_dim (int): The input dimension of the model.
            hidden_dim (int): The hidden dimension of the model.
            output_dim (int): The output dimension of the model.
            num_freqs (int or list of length 2, optional): The number of frequencies used for positional encoding. Defaults to 10.
            num_layers (int, optional): The number of layers in each NeRF_MLP_Residual_Scaled model. Defaults to 4.
            num_compose (int, optional): The number of NeRF_MLP_Residual_Scaled models to compose. Defaults to 4.
            normalizing_factor (float, optional): The normalizing factor for layer_id and input_dim. Defaults to 1.0.
        """
        super(NeRF_MLP_Compose, self).__init__()

        self.positional_encoding = PositionalEncoding(num_freqs, input_dim=input_dim)
        self.output_dim = output_dim
        self.model = nn.ModuleList()
        self.norm = normalizing_factor

        if isinstance(num_freqs, int):
            num_freqs = num_freqs
        elif len(num_freqs) == 2:
            num_freqs = num_freqs[1] - num_freqs[0]
        else:
            raise ValueError("num_freqs should be either an integer or a list of length 2.")
        
        for _ in range(num_compose):
            self.model.append(NeRF_MLP_Residual_Scaled(input_dim + 2 * input_dim * num_freqs, hidden_dim, output_dim, num_freqs, num_layers))
        
        # Initialize weights for all Linear layers
        self.apply(weights_init_uniform_relu)

    def forward(self, x, layer_id=None, input_dim=None):
        """
        Forward pass of the NeRF_MLP_Compose model.

        Args:
            x (torch.Tensor): The input tensor that encodes the coordinates of each neuron (or channel) in the targeted network.
            layer_id (torch.Tensor, optional): The layer ID tensor. Defaults to None.
            input_dim (torch.Tensor, optional): The input dimension tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        if layer_id is None:
            layer_id = (x[:, 0] * self.norm).int()
        if input_dim is None:
            input_dim = (x[:, -1] * self.norm)

        x[:, :3] = x[:, :3] / x[:, 3:]  # Normalize coordinates tensor
        x = self.positional_encoding(x)  # Apply Fourier encoding
        unique_layer_ids = torch.unique(layer_id)
        output_x = torch.zeros((x.size(0), self.output_dim)).to(x.device)  # Create output tensor with size of input tensor

        # Process each coordinate in Fourier encoded x tensor using residual block
        for lid in unique_layer_ids:
            mask = lid == layer_id  # Get the corresponding neurons (or channels) in each layer
            output_x[mask] = self.model[lid].forward(x[mask])  # Process that layer
        return output_x / (input_dim.unsqueeze(-1))
    
class NeRF_ResMLP_Compose(NeRF_MLP_Compose):
    """
    NeRF_ResMLP_Compose is a class that represents a compositional multi-layer perceptron (MLP) model with residual connections.
    I.e. Blocks of INR, composed as one.

    Args:
        input_dim (int): The dimensionality of the input features.
        hidden_dim (int): The dimensionality of the hidden layers.
        output_dim (int): The dimensionality of the output.
        num_freqs (int, optional): The number of frequencies used in positional encoding. Defaults to 10.
        num_layers (int, optional): The number of layers in the MLP. Defaults to 4.
        num_compose (int, optional): The number of compositional MLPs to be composed. Defaults to 4.
        normalizing_factor (float, optional): The normalizing factor for the model. Defaults to 1.0.
        scalar (float, optional): The scalar value used in the residual connections. Defaults to 0.1.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_freqs = 10, num_layers = 4, num_compose = 4, normalizing_factor = 1.0, scalar = 0.1):
        super(NeRF_ResMLP_Compose, self).__init__(input_dim, hidden_dim, output_dim, num_freqs, num_layers, num_compose, normalizing_factor)

        self.model = nn.ModuleList()
        self.norm = normalizing_factor

        for _ in range(num_compose):
            self.model.append(NeRF_MLP_Residual_Scaled(input_dim + 2 * input_dim * num_freqs, hidden_dim, output_dim, num_freqs, num_layers, scalar=scalar))

        # Initialize weights for all Linear layers
        self.apply(weights_init_uniform_relu)