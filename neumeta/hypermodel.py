import torch.nn as nn
import math
import numpy as np
import torch

def weights_init_uniform_relu(m):
   
    """
    Initialize uniform weights for Linear layer:
    1. Get layer type
    2. If layer type is Linear
    3. Initialize uniform weights for the layer using kaiming_uniform
        3.1 If layer has bias, then initialize uniform bias for it too.
    
    Parameters:
        m (nn.Module): Neural network layer.

    Returns: 
        None.
    """

    # Get name of layer
    classname = m.__class__.__name__

    # for every Linear layer in a model
    if classname.find('Linear') != -1:  

        # init weights for layer using kaiming_uniform
        torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

        # init uniform bias' values if layer has bias
        if m.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(m.bias, -bound, bound)


class LearnablePositionalEmbeddings(nn.Module):
    """

    A    
    
    """

    def __init__(self, d_model, seq_len):
        """
        
        Parameters:
            d_model: 
            seq_len: 
        
        """

        super(LearnablePositionalEmbeddings, self).__init__()

        # Learnable positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x):
        # Add the learnable positional embeddings to the input
        return x + self.positional_embeddings
    


class PositionalEncoding(nn.Module):

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

        for i in range(self.freqs_low, self.freqs_high):
            freq = 2.0 ** i * np.pi
            for j in range(self.input_dim):
                out.append(torch.sin(freq * x[:, j].unsqueeze(-1)))
                out.append(torch.cos(freq * x[:, j].unsqueeze(-1)))
        return torch.cat(out, dim = -1)
    


class NeRF_MLP_Residual_Scaled(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_freqs = None, num_layers = 4, scalar = 0.0):
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

        # Initial transformation
        x = self.act(self.initial_layer(x))

        # Process through each residual block
        for block, scale in zip(self.residual_blocks, self.scalars):
            
            # Store the residual
            residual = x                            
            
            out = block(x)

            # Apply scaled activation and add residual
            x = scale * self.act(out) + residual    

        # Final transformation
        x = self.output_layer(x)
        return x
    

class NeRF_MLP_Compose(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_freqs=10, num_layers=4, num_compose=4, normalizing_factor=1.0):
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
        
        self.apply(weights_init_uniform_relu)

    def forward(self, x, layer_id=None, input_dim=None):
        
        if layer_id is None:
            layer_id = (x[:, 0] * self.norm).int()

        if input_dim is None:
            input_dim = (x[:, -1] * self.norm)

        x[:, :3] = x[:, :3] / x[:, 3:]
        x = self.positional_encoding
        unique_layer_ids = torch.unique(layer_id)
        output_x = torch.zeros((x.size(0), self.output_dim).to(x.device))
        for lid in unique_layer_ids:
            mask = lid == layer_id
            output_x[mask] = self.model[lid].forward(x[mask])
        return output_x / (input_dim.unsqueeze(-1))
    
class NeRF_ResMLP_Compose(NeRF_MLP_Compose):
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_freqs = 10, num_layers = 4, num_compose = 4, normalizing_factor = 1.0, scalar = 0.1):
        super(NeRF_ResMLP_Compose, self).__init__(input_dim, hidden_dim, output_dim, num_freqs, num_layers, num_compose, normalizing_factor)

        self.model = nn.ModuleList()
        self.norm = normalizing_factor

        for _ in range(num_compose):
            self.model.append(NeRF_MLP_Residual_Scaled(input_dim + 2 * input_dim * num_freqs, hidden_dim, output_dim, num_freqs, num_layers, scalar=scalar))

        self.apply(weights_init_uniform_relu)