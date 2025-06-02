import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, droprate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = droprate
        self.dropout = nn.Dropout(p=self.droprate)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = self.dropout(out)
        # returns the concatenation of input and output
        return torch.cat([x, out], 1)
    

class BottleneckBlock(nn.Module):
    """Handles bottleneck and compression"""
    def __init__(self, in_planes, out_planes, droprate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)

        # Enlarge the first convolutional layer's output
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)

        # Then reduce it in the second convolutional layer's
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        self.droprate = droprate
        self.dropout = nn.Dropout(p=self.droprate)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = self.dropout(out)

        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = self.dropout(out)

        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, droprate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        self.droprate = droprate
        self.dropout = nn.Dropout(p=self.droprate)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = self.dropout(out)

        return self.avgpool(out)


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)

class DenseNet3_changeable(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, droprate=0.0,
                 hidden_dim=48, inr_layer=1):
        super(DenseNet3_changeable, self).__init__()
        
        self.inr_layer = inr_layer

        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n / 2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)

        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, droprate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), droprate=droprate)
        in_planes = int(math.floor(in_planes * reduction))

        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, droprate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), droprate=droprate)
        in_planes = int(math.floor(in_planes * reduction))

        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, droprate)
        in_planes = int(in_planes+n*growth_rate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        self.set_changeable(hidden_dim, self.inr_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(-1, self.in_planes)
        return self.fc(out)
    
    def set_changeable(self, hidden_dim, inr_layer):

        all_layers = []
        
        for block_name in ['block1', 'block2', 'block3']:
            block = getattr(self, block_name)
            for layer in block.layer:
                all_layers.append(layer)

        # Reverse the layers from bottom
        all_layers = all_layers[::-1]

        for i in range(min(inr_layer, len(all_layers))):
            module = all_layers[i]

            old_conv1 = module.conv1
            module.conv1 = nn.Conv2d(
                in_channels=old_conv1.in_channels,
                out_channels=hidden_dim,
                kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride,
                padding=old_conv1.padding,
                bias=old_conv1.bias is not None
            )

            # Replace bn2
            old_bn2 = module.bn2
            module.bn2 = nn.BatchNorm2d(
                num_features=hidden_dim,
                eps=old_bn2.eps,
                momentum=old_bn2.momentum,
                affine=old_bn2.affine,
                track_running_stats=old_bn2.track_running_stats
            )

            # Replace conv2
            old_conv2 = module.conv2
            module.conv2 = nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=old_conv2.out_channels,
                kernel_size=old_conv2.kernel_size,
                stride=old_conv2.stride,
                padding=old_conv2.padding,
                bias=old_conv2.bias is not None
            )  
    
    @property
    def learnable_parameter(self):

        all_layers = []
        
        for block_name in ['block1', 'block2', 'block3']:
            block = getattr(self, block_name)
            for layer in block.layer:
                all_layers.append(layer)

        selected_layers = all_layers[::-1][:self.inr_layer]

        self.keys = []

        for name, module in self.named_modules():
            if any(module is layer for layer in selected_layers):
                for param_name, _ in module.named_parameters():
                    full_name = f'{name}.{param_name}' if name else param_name
                    if 'conv' in param_name:
                        self.keys.append(full_name)
        
        # self.keys = [k 
        #              for k, w in self.named_parameters()
        #              if k.startswith('block3.layer.5.conv')]
        
        return {k: v
                for k, v in self.state_dict().items()
                if k in self.keys}
        

        # return {k: v.detach().clone()
        #         for k, v in dict(self.named_parameters()).items()}