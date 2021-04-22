import torch
import torch.nn as nn
import math
from functools import reduce
from utils import identity
from utils.module import DropoutAUG

__all__ = ['ConvNet']

class Conv2dBlock(nn.Conv2d):
    """
    """
    def __init__(self,
        in_channel, 
        out_channel,
        kernel_size,
        normalizer = None,
        activation = None,
        **kwargs
    ):
        super().__init__(in_channel, out_channel, kernel_size, **kwargs)
        if isinstance(normalizer, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            assert normalizer.num_features == out_channel
        elif normalizer is not None:
            raise ValueError("Only batchnorm and instancenorm are supported.")
        else:
            normalizer = identity
        self.normalizer = normalizer
        self.activation = activation or identity
    
    def forward(self, x):
        return self.activation(self.normalizer(super().forward(x)))


class ConvNet(nn.Module):
    """
    Args:
    @param: num_stage (int), number of conv blocks in model.
    @param: out_channels (int or tuple), out channels of each block.
    @param: required_input_size (int or tuple), the required input size, e.g. 84 or (84,84).
    @param: transductive_batchnorm (bool), If set to True, with use BatchNorm2d with 
    track_running_stats=False for normalization so that the module will not track 
    batch normalization statistics and always use batch statistics in both train and 
    eval modes, which is called transductive batch normalization (TBN) in few-shot learning.
    """
    def __init__(self, num_stage, out_channels, required_input_size, transductive_batchnorm=False):
        super().__init__()
        assert num_stage > 0
        self.num_stage = num_stage
        self.out_channels = [out_channels]*num_stage if isinstance(out_channels, int) else out_channels
        assert len(self.out_channels) == self.num_stage
        self.in_channels = [3, *(self.out_channels[:-1])]
        #
        trunk = []
        for i in range(self.num_stage):
            trunk.append(
                Conv2dBlock(
                    self.in_channels[i],
                    self.out_channels[i],
                    kernel_size = 3,
                    padding = 1, 
                    stride = 1,
                    bias = False,
                    normalizer = nn.BatchNorm2d(self.out_channels[i], track_running_stats=not transductive_batchnorm),
                    activation = nn.Sequential(
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                    )
                )
                )
        self.extractor = nn.Sequential(*trunk)
        #
        self.image_height, self.image_width = [required_input_size]*2 if isinstance(required_input_size, int) else required_input_size
        factor = 1.0 / math.pow(2, self.num_stage)
        self.expected_out_size = (self.out_channels[-1], int(self.image_height * factor), int(self.image_width * factor))

    def insert(self, key, value):
        if getattr(self, key, None) is not None:
            raise KeyError("Attribute %s already exists, try another name." % key)
        setattr(self, key, value)
    
    @property
    def feat_size(self):
        return reduce((lambda x,y: x*y), self.expected_out_size)
    
    @property
    def out_dim(self):
        return self.expected_out_size[0]
    
    def forward(self, x):
        N,C,H,W = x.shape
        assert C == self.in_channels[0]
        assert H == self.image_height
        assert W == self.image_width
        x = self.extractor(x)
        _,_,Ho,Wo = x.shape
        assert Ho == self.expected_out_size[1]
        assert Wo == self.expected_out_size[2]
        return x