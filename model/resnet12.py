import torch
import torch.nn as nn
import math
import functools
from functools import reduce
from utils import identity
from utils.module import DropoutAUG

__all__ = ['ResNet']

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, 
        in_channel,
        out_channel,
        kernel = 3,
        stride = 1,
        downsample = None,
        maxpool = True,
        drop_rate=0.0,
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        transductive_batchnorm = False,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel = kernel
        self.stride = stride
        if not kernel in [1,3]:
            raise ValueError('unrecognized kernel size %d, only support 1 or 3.' % kernel)
        if not stride in [1,2]:
            raise ValueError('unrecognized stride size %d, only support 1 or 2.' % stride)
        self.padding = 1 if kernel == 3 else 0
        self.downsample = downsample
        self.dropout = DropoutAUG(
            p=drop_rate,
            drop_block=drop_block,
            drop_size=drop_size,
            drop_stablization=drop_stablization
        )
        self.relu = nn.LeakyReLU(0.1)
        self.maxpool = nn.MaxPool2d(2) if not not maxpool else identity
        self.conv1 = nn.Conv2d(in_channel, out_channel, self.kernel, stride=1, padding=self.padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel, track_running_stats=not transductive_batchnorm)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, stride=self.stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel, track_running_stats=not transductive_batchnorm)
        self.conv3 = nn.Conv2d(out_channel, out_channel, self.kernel, stride=1, padding=self.padding, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel, track_running_stats=not transductive_batchnorm)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            out += self.downsample(x)
        out = self.maxpool(self.relu(out))
        out = self.dropout(out)
        return out


class ResNet(nn.Module):
    def __init__(self,
        block,
        num_blocks, 
        out_channels, 
        required_input_size,
        maxpool = True,
        drop_rate=0.0,
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        transductive_batchnorm=False):
        super().__init__()
        num_stage = len(num_blocks)
        assert num_stage > 0
        self.num_stage = num_stage
        self.num_blocks = num_blocks
        self.out_channels = [out_channels]*num_stage if isinstance(out_channels, int) else out_channels
        assert len(self.out_channels) == self.num_stage
        self.maxpool = not not maxpool
        self.block = functools.partial(
            block,
            kernel = 3,
            drop_rate = drop_rate,
            drop_block = drop_block,
            drop_size = drop_size,
            drop_stablization = drop_stablization,
        )
        self.block_expansion = block.expansion
        self.in_channels = [3, *(i*block.expansion for i in self.out_channels[:-1])]
        self.image_height, self.image_width = [required_input_size]*2 if isinstance(required_input_size, int) else required_input_size
        self.expected_out_size = self._expected_out_size
        #
        trunk = []
        for i in range(self.num_stage):
            trunk.append(
                self._make_layer(
                    self.block, 
                    self.block_expansion, 
                    self.num_blocks[i], 
                    self.in_channels[i], 
                    self.out_channels[i], 
                    maxpool=self.maxpool,
                    transductive_batchnorm=transductive_batchnorm)
            )
        self.extractor = nn.Sequential(*trunk)
        #
    
    def _make_layer(self, block, expansion, nblocks, inplanes, outplanes, maxpool=True, transductive_batchnorm=False):
        downsample = None
        stride = 1 if maxpool else 2
        if stride == 2 or inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, 1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(outplanes, track_running_stats= not transductive_batchnorm)
            )
        layers = []
        layers.append(block(
            inplanes,
            outplanes,
            maxpool = maxpool,
            stride = stride,
            downsample = downsample,
            transductive_batchnorm = transductive_batchnorm
            ))
        #
        inplanes = expansion * outplanes
        for _ in range(1, nblocks):
            layers.append(block(
                inplanes,
                outplanes,
                maxpool = False,
                stride = 1,
                downsample = None,
                transductive_batchnorm = transductive_batchnorm
            ))
        return nn.Sequential(*layers)
    
    def insert(self, key, value):
        if getattr(self, key, None) is not None:
            raise KeyError("Attribute %s already exists, try another name." % key)
        setattr(self, key, value)
    
    @property
    def _expected_out_size(self):
        factor = 1.0 / math.pow(2, self.num_stage)
        C = self.block_expansion * self.out_channels[-1]
        H = int(self.image_height * factor)
        W = int(self.image_width * factor)
        if not self.maxpool:
            H = H + 1
            W = W + 1
        return C, H, W

    @property
    def feat_size(self):
        return reduce((lambda x,y: x*y), self.expected_out_size)
    
    @property
    def out_dim(self):
        return self.expected_out_size[0]
    
    def forward(self, x):
        N,C,H,W = x.shape
        # import pdb; pdb.set_trace()
        assert C == 3
        assert H == self.image_height
        assert W == self.image_width
        x = self.extractor(x)
        _,_,Ho,Wo = x.shape
        assert Ho == self.expected_out_size[1]
        assert Wo == self.expected_out_size[2]
        return x
