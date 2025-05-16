'''
Modified from https://raw.githubusercontent.com/pytorch/vision/v0.9.1/torchvision/models/resnet.py

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from .utils import load_checkpoint
from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional


cifar10_pretrained_weight_urls = {
    'resnet20': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet20-4118986f.pt',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    """
    Simple ResNet block:
    - Input size: in_planes
    - Output size: out_planes
    """
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)


class BasicBlock_Resize(BasicBlock):
    """
    Resized version of ResNet block:
    - Input size: in_planes
    - Output size: in_planes
    """
    expansion = 1
    
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super().__init__(in_planes, out_planes, stride, downsample)

        # Change the output size of conv2 into in_planes
        # Now the output has the same size as input of this block
        self.conv2 = conv3x3(out_planes, in_planes)
        self.bn2 = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class CifarResNet(nn.Module):
    def __init__(
            self, block, hidden_dim, layers, 
            num_classes=10, num_layers_inr=1):
        super(CifarResNet, self).__init__()
        
        # Defining number of layers
        self.layers = layers  # Tell number of ResNet blocks for each layer1 to layer3 below
        self.num_layers_inr = num_layers_inr  # Define the amount of last blocks in layer3 that we want to change
        
        # First time processing
        self.in_planes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Create new layers
        # Creates layers[0] number of ResNet block stacked orderly.
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Ends with MLP block ( , 10)
        # 10 outputs
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # Set changeable
        self.set_changeable(block, hidden_dim, stride=1, num_classes=num_classes)

        # Loop through layers to initializing weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_planes, blocks, stride=1):
        """
        Makes new layer by stacking ResNet blocks
        
        Args:
            block (BasicBlock): ResNet block.
            out_planes (int): Output size.
            blocks (int): Taken from the elements in self.layers,
                          Define the number of blocks needed to be made in this layer.
            stride (int): Stride for convolutional kernel.
        """

        # If stride > 1 or input size and output size is different,
        # Then downsample is needed to make the size correct
        downsample = None
        if stride != 1 or self.in_planes != out_planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, out_planes * block.expansion, stride),
                nn.BatchNorm1d(out_planes * block.expansion)
            )

        # Create blocks for each layer (stacks orderly)
        layers = []
        layers.append(block(self.in_planes, out_planes, stride, downsample))  # Add the first block into the layer list
        self.in_planes = out_planes * block.expansion  # Next block will have the output size of current block as input size
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, out_planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial process
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Process for adding new layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Final process
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten x for MLP process
        x = self.fc(x)

        return x
    
    def set_changeable(self, block, planes, stride, num_classes=10):
        # Change the last blocks in layer3 as specfied in num_layers_inr
        for name, child in self.named_children():
            if name == 'layer3':
                print(f'Replace the last 2 block of layer3 with new block with hidden dim {planes}')
                layers = list(child.children())[:-self.num_layers_inr]  # Get all blocks in layer 3 except the last block
                for i in range(self.num_layers_inr):  # Add new blocks to layer 3
                    layers.append(BasicBlock_Resize(64, planes, stride))
                self._modules[name] = nn.Sequential(*layers)
    
    @property
    def learnable_parameter(self):
        """Set parameters for last blocks in layer3 specified in num_layers_inr to trainable parameter"""
        # Get the layers which we want to train parameters on
        self.keys = [k 
                     for k, w in self.named_parameters() 
                     if k.startswith(f'layer3.{self.layers[-1]-1}')]

        return {k: v 
                for k, v in self.state_dict().items()
                if k in self.keys}

def _resnet(
        arch: str,
        hidden_dim: int, layers: List[int],
        model_urls: Dict[str, str],
        progress: bool = True, pretrained: bool = True,
        **kwargs: Any
) -> CifarResNet:
    # Initialize model
    model = CifarResNet(BasicBlock, hidden_dim, layers, **kwargs)

    # Load weights, if pretrained
    # By default, it is pretrained
    if pretrained:
        print(f'Loading pretrained weights for {arch}')
        state_dict = load_state_dict_from_url(
            model_urls[arch], 
            progress=progress)
        load_checkpoint(model, state_dict)

    return model

# Functions for CIFAR-10
def cifar10_resnet20(
        hidden_dim, num_classes=10, pretrained=True, 
        *args, **kwargs):
    return _resnet(
        arch='resnet20',
        hidden_dim=hidden_dim,
        layers=[3] * 3,  # Indicates the repetitions of certain block
        model_urls=cifar10_pretrained_weight_urls,
        num_classes=num_classes,
        pretrained=pretrained,
        *args,
        **kwargs
    )