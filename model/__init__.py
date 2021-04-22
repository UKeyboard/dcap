import torch
import torch.nn as nn
import numpy as np
from functools import reduce
from model.convnet import ConvNet as _ConvNet
from model.resnet12 import BasicBlock as _BasicBlock, ResNet as _ResNet
from utils.module import LinearClassifier, DenseClassifier, CosineClassifier, ProjectionClassifier

models = {}
def _register_model(name):
    """A general register for models.

    Args:
    @param name: str or list, the model alias(s), e.g. 'conv64' or ['conv64', 'convnet64'].
    """
    if isinstance(name, str): name = [name]
    assert all(map(lambda x: isinstance(x, str), name))
    def wrapper(cls):
        for c in name:
            T = models.get(c, None)
            assert T is None
            models[c] = cls
        return cls
    return wrapper


class ConvNet(_ConvNet):
    """Return BxCxHxW feature map.
    """
    def __init__(self,
        num_stage,
        out_channels,
        required_input_size, 
        num_classes=-1, 
        biased=False, # use biased classifier if True (only works when num_classes > 0)
        avgpool=False, 
        transductive_bn=False, 
        final_activation=None):
        super().__init__(num_stage, out_channels, required_input_size=required_input_size, transductive_batchnorm=transductive_bn)
        self.avgpool = not not avgpool
        self.final_activation = final_activation
        self.num_classes = -1
        self.is_biased_classifier = not not biased
        if num_classes is not None and num_classes > 0:
            self.num_classes = num_classes
            self.insert(
                'classifier', 
                LinearClassifier(self.feat_size, self.num_classes, bias=self.is_biased_classifier),
                # DenseClassifier(self.feat_size, self.num_classes, bias=self.is_biased_classifier),
                # CosineClassifier(self.feat_size, self.num_classes, bias=self.is_biased_classifier),
                # ProjectionClassifier(self.feat_size, self.num_classes, bias=self.is_biased_classifier),
                )
        self.initialized_when_created = False
    
    @property
    def feat_size(self):
        if self.avgpool:
            return self.expected_out_size[0]
        return reduce((lambda x,y: x*y), self.expected_out_size)
    
    def forward(self, x):
        x = super().forward(x) # the raw output feature map
        if self.final_activation is not None: x = self.final_activation(x)
        if self.avgpool:
            x = x.mean(dim=(-2,-1), keepdim=True)
        return x


class ConvNet_wC(ConvNet):
    """Return Bx#class logits.
    """
    def __init__(self, num_stage, out_channels, required_input_size, num_classes, biased=False, avgpool=False, transductive_bn=False, final_activation=None):
        assert isinstance(num_classes, int)
        assert num_classes > 0
        super().__init__(num_stage, out_channels, required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)
    
    def forward(self, x):
        x = super().forward(x)
        batch_size, *dim = x.shape
        x = x.view(batch_size, self.feat_size)
        x = self.classifier(x)
        return x


class ConvNet_wCplus(ConvNet):
    """Return BxCxHxW feature map and Bx#class logits.
    """
    def __init__(self, num_stage, out_channels, required_input_size, num_classes, biased=False, avgpool=False, transductive_bn=False, final_activation=None):
        assert isinstance(num_classes, int)
        assert num_classes > 0
        super().__init__(num_stage, out_channels, required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)
    
    def forward(self, x):
        x = super().forward(x)
        batch_size, *dim = x.shape
        y = x.view(batch_size, self.feat_size)
        y = self.classifier(y)
        return x, y

class ConvNet_wDC(ConvNet):
    """Return Bx#classxHxW logits.
    """
    def __init__(self, num_stage, out_channels, required_input_size, num_classes, biased=False, avgpool=False, transductive_bn=False, final_activation=None):
        assert isinstance(num_classes, int)
        assert num_classes > 0
        assert avgpool == False
        super().__init__(num_stage, out_channels, required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)
    
    @property
    def feat_size(self):
        return self.expected_out_size[0]

    def forward(self, x):
        x = super().forward(x)
        batch_size, C, H, W = x.shape
        assert C == self.feat_size
        x = x.view(batch_size, C, H*W)
        x = x.transpose(1,2)
        x = x.reshape(batch_size*H*W, C)
        x = self.classifier(x)
        x = x.view(batch_size, H*W, -1)
        x = x.transpose(1,2)
        x = x.reshape(batch_size, -1, H, W)
        return x


class ConvNet_wDCplus(ConvNet):
    """Return BxCxHxW feature map and Bx #class xHxW logits.
    """
    def __init__(self, num_stage, out_channels, required_input_size, num_classes, biased=False, avgpool=False, transductive_bn=False, final_activation=None):
        assert isinstance(num_classes, int)
        assert num_classes > 0
        assert avgpool == False
        super().__init__(num_stage, out_channels, required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)

    @property
    def feat_size(self):
        return self.expected_out_size[0]

    def forward(self, x):
        x = super().forward(x)
        batch_size, C, H, W = x.shape
        assert C == self.feat_size
        y = x.view(batch_size, C, H*W)
        y = y.transpose(1,2)
        y = y.reshape(batch_size*H*W, C)
        y = self.classifier(y)
        y = y.view(batch_size, H*W, -1)
        y = y.transpose(1,2)
        y = y.reshape(batch_size, -1, H, W)
        return x, y


class ResNet(_ResNet):
    """Return BxCxHxW feature map.
    """
    def __init__(self, 
        block,
        num_blocks, 
        out_channels, 
        required_input_size,
        num_classes = -1,
        biased=False, # use biased classifier if True (only works when num_classes > 0)
        maxpool = True,
        drop_rate=0.0,
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        avgpool = False,
        transductive_bn=False,
        final_activation=None):
        super().__init__(
            block, num_blocks, out_channels,
            required_input_size=required_input_size,
            maxpool = maxpool,
            drop_rate=drop_rate,
            drop_block=drop_block,
            drop_size=drop_size,
            drop_stablization=drop_stablization,
            transductive_batchnorm=transductive_bn
        )
        self.avgpool = not not avgpool
        self.final_activation = final_activation
        self.num_classes = -1
        self.is_biased_classifier = not not biased
        if num_classes is not None and num_classes > 0:
            self.num_classes = num_classes
            self.insert(
                'classifier',
                LinearClassifier(self.feat_size, self.num_classes, bias=self.is_biased_classifier),
                # DenseClassifier(self.feat_size, self.num_classes, bias=self.is_biased_classifier),
                # CosineClassifier(self.feat_size, self.num_classes, bias=self.is_biased_classifier),
                # ProjectionClassifier(self.feat_size, self.num_classes, bias=self.is_biased_classifier),
                )
        self.initialized_when_created = False
    
    @property
    def feat_size(self):
        if self.avgpool:
            return self.expected_out_size[0]
        return reduce((lambda x,y: x*y), self.expected_out_size)
    
    def forward(self, x):
        x = super().forward(x) # the raw output feature map
        if self.final_activation is not None: x = self.final_activation(x)
        if self.avgpool:
            x = x.mean(dim=(-2,-1), keepdim=True)
        return x

class ResNet_wC(ResNet):
    """Return Bx#class logits.
    """
    def __init__(self,
        block,
        num_blocks, 
        out_channels,
        required_input_size,
        num_classes,
        biased=False,
        maxpool=True,
        drop_rate=0.0,
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        avgpool = False,
        transductive_bn=False,
        final_activation=None):
        assert isinstance(num_classes, int)
        assert num_classes > 0
        super().__init__(
            block,
            num_blocks, 
            out_channels,
            required_input_size, 
            num_classes, 
            biased,
            maxpool,
            drop_rate, 
            drop_block, 
            drop_size,
            drop_stablization,
            avgpool,
            transductive_bn,
            final_activation)

    def forward(self, x):
        x = super().forward(x)
        batch_size, *dim = x.shape
        x = x.view(batch_size, self.feat_size)
        x = self.classifier(x)
        return x


class ResNet_wCplus(ResNet):
    """Return BxCxHxW feature map and Bx#class logits.
    """
    def __init__(self, 
        block,
        num_blocks, 
        out_channels,
        required_input_size,
        num_classes,
        biased=False,
        maxpool=True,
        drop_rate=0.0,
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        avgpool = False,
        transductive_bn=False,
        final_activation=None):
        assert isinstance(num_classes, int)
        assert num_classes > 0
        super().__init__(
            block,
            num_blocks, 
            out_channels,
            required_input_size, 
            num_classes, 
            biased,
            maxpool,
            drop_rate, 
            drop_block, 
            drop_size,
            drop_stablization,
            avgpool,
            transductive_bn,
            final_activation)
            
    def forward(self, x):
        x = super().forward(x)
        batch_size, *dim = x.shape
        y = x.view(batch_size, self.feat_size)
        y = self.classifier(y)
        return x, y


class ResNet_wDC(ResNet): # dense classification
    """Return Bx#classxHxW logits.
    """
    def __init__(self, 
        block,
        num_blocks, 
        out_channels,
        required_input_size,
        num_classes,
        biased=False,
        maxpool=True,
        drop_rate=0.0,
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        avgpool = False,
        transductive_bn=False,
        final_activation=None):
        assert isinstance(num_classes, int)
        assert num_classes > 0
        assert avgpool == False
        super().__init__(
            block,
            num_blocks, 
            out_channels,
            required_input_size, 
            num_classes,
            biased,
            maxpool,
            drop_rate, 
            drop_block, 
            drop_size,
            drop_stablization,
            avgpool,
            transductive_bn,
            final_activation)

    @property
    def feat_size(self):
        return self.expected_out_size[0]

    def forward(self, x):
        x = super().forward(x)
        batch_size, C, H, W = x.shape
        assert C == self.feat_size
        x = x.view(batch_size, C, H*W)
        x = x.transpose(1,2)
        x = x.reshape(batch_size*H*W, C)
        x = self.classifier(x)
        x = x.view(batch_size, H*W, -1)
        x = x.transpose(1,2)
        x = x.reshape(batch_size, -1, H, W)
        return x


class ResNet_wDCplus(ResNet):
    """Return BxCxHxW feature map and Bx #class xHxW logits.
    """
    def __init__(self, 
        block,
        num_blocks, 
        out_channels,
        required_input_size,
        num_classes,
        biased=False,
        maxpool=True,
        drop_rate=0.0,
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        avgpool = False,
        transductive_bn=False,
        final_activation=None):
        assert isinstance(num_classes, int)
        assert num_classes > 0
        assert avgpool == False
        super().__init__(
            block,
            num_blocks, 
            out_channels,
            required_input_size, 
            num_classes,
            biased,
            maxpool,
            drop_rate, 
            drop_block, 
            drop_size,
            drop_stablization,
            avgpool,
            transductive_bn,
            final_activation)

    @property
    def feat_size(self):
        return self.expected_out_size[0]

    def forward(self, x):
        x = super().forward(x)
        batch_size, C, H, W = x.shape
        assert C == self.feat_size
        y = x.view(batch_size, C, H*W)
        y = y.transpose(1,2)
        y = y.reshape(batch_size*H*W, C)
        y = self.classifier(y)
        y = y.view(batch_size, H*W, -1)
        y = y.transpose(1,2)
        y = y.reshape(batch_size, -1, H, W)
        return x, y


@_register_model('conv64')
class ConvNet64(ConvNet):
    """Return BxCxHxW feature map.
    """
    def __init__(self, 
        required_input_size, 
        num_classes=-1, 
        biased=False, # use biased classifier if True (only works when num_classes > 0)
        avgpool=False, 
        transductive_bn=False, 
        final_activation=None):
        super().__init__(4, [64,64,64,64], required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)

@_register_model('conv64_c')
class ConvNet64_wC(ConvNet_wC):
    """Return Bx#class logits.
    """
    def __init__(self, 
        required_input_size, 
        num_classes, 
        biased=False, 
        avgpool=False, 
        transductive_bn=False, 
        final_activation=None):
        super().__init__(4, [64,64,64,64], required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)


@_register_model('conv64_c+')
class ConvNet64_wCplus(ConvNet_wCplus):
    """Return BxCxHxW feature map and Bx#class logits.
    """
    def __init__(self, 
        required_input_size, 
        num_classes, 
        biased=False, 
        avgpool=False, 
        transductive_bn=False, 
        final_activation=None):
        super().__init__(4, [64,64,64,64], required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)

@_register_model('conv64_dc')
class ConvNet64_wDC(ConvNet_wDC):
    """Return Bx#classxHxW logits.
    """
    def __init__(self, 
        required_input_size, 
        num_classes, 
        biased=False, 
        avgpool=False, 
        transductive_bn=False, 
        final_activation=None):
        super().__init__(4, [64,64,64,64], required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)


@_register_model('conv64_dc+')
class ConvNet64_wDCplus(ConvNet_wDCplus):
    """Return BxCxHxW feature map and Bx #class xHxW logits.
    """
    def __init__(self, 
        required_input_size, 
        num_classes, 
        biased=False, 
        avgpool=False, 
        transductive_bn=False, 
        final_activation=None):
        super().__init__(4, [64,64,64,64], required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)


@_register_model('conv128')
class ConvNet128(ConvNet):
    """Return BxCxHxW feature map.
    """
    def __init__(self, 
        required_input_size, 
        num_classes=-1, 
        biased=False, # use biased classifier if True (only works when num_classes > 0)
        avgpool=False, 
        transductive_bn=False, 
        final_activation=None):
        super().__init__(4, [64,64,128,128], required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)

@_register_model('conv128_c')
class ConvNet128_wC(ConvNet_wC):
    """Return Bx#class logits.
    """
    def __init__(self, 
        required_input_size, 
        num_classes, 
        biased=False, 
        avgpool=False, 
        transductive_bn=False, 
        final_activation=None):
        super().__init__(4, [64,64,128,128], required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)


@_register_model('conv128_c+')
class ConvNet128_wCplus(ConvNet_wCplus):
    """Return BxCxHxW feature map and Bx#class logits.
    """
    def __init__(self, 
        required_input_size, 
        num_classes, 
        biased=False, 
        avgpool=False, 
        transductive_bn=False, 
        final_activation=None):
        super().__init__(4, [64,64,128,128], required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)

@_register_model('conv128_dc')
class ConvNet128_wDC(ConvNet_wDC):
    """Return Bx#classxHxW logits.
    """
    def __init__(self, 
        required_input_size, 
        num_classes, 
        biased=False, 
        avgpool=False, 
        transductive_bn=False, 
        final_activation=None):
        super().__init__(4, [64,64,128,128], required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)


@_register_model('conv128_dc+')
class ConvNet128_wDCplus(ConvNet_wDCplus):
    """Return BxCxHxW feature map and Bx #class xHxW logits.
    """
    def __init__(self, 
        required_input_size, 
        num_classes, 
        biased=False, 
        avgpool=False, 
        transductive_bn=False, 
        final_activation=None):
        super().__init__(4, [64,64,128,128], required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)
    

@_register_model('conv256')
class ConvNet256(ConvNet):
    """Return BxCxHxW feature map.
    """
    def __init__(self, 
        required_input_size, 
        num_classes=-1, 
        biased=False, # use biased classifier if True (only works when num_classes > 0)
        avgpool=False, 
        transductive_bn=False, 
        final_activation=None):
        super().__init__(4, [64,96,128,256], required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)

@_register_model('conv256_c')
class ConvNet256_wC(ConvNet_wC):
    """Return Bx#class logits.
    """
    def __init__(self, 
        required_input_size, 
        num_classes, 
        biased=False, 
        avgpool=False, 
        transductive_bn=False, 
        final_activation=None):
        super().__init__(4, [64,96,128,256], required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)


@_register_model('conv256_c+')
class ConvNet256_wCplus(ConvNet_wCplus):
    """Return BxCxHxW feature map and Bx#class logits.
    """
    def __init__(self, 
        required_input_size, 
        num_classes, 
        biased=False, 
        avgpool=False, 
        transductive_bn=False, 
        final_activation=None):
        super().__init__(4, [64,96,128,256], required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)

@_register_model('conv256_dc')
class ConvNet256_wDC(ConvNet_wDC):
    """Return Bx#classxHxW logits.
    """
    def __init__(self, 
        required_input_size, 
        num_classes, 
        biased=False, 
        avgpool=False, 
        transductive_bn=False, 
        final_activation=None):
        super().__init__(4, [64,96,128,256], required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)


@_register_model('conv256_dc+')
class ConvNet256_wDCplus(ConvNet_wDCplus):
    """Return BxCxHxW feature map and Bx #class xHxW logits.
    """
    def __init__(self, 
        required_input_size, 
        num_classes, 
        biased=False, 
        avgpool=False, 
        transductive_bn=False, 
        final_activation=None):
        super().__init__(4, [64,96,128,256], required_input_size, num_classes, biased, avgpool, transductive_bn, final_activation)


@_register_model('resnet12')
class ResNet12(ResNet):
    """Return BxCxHxW feature map.
    """
    def __init__(self, 
        required_input_size,
        num_classes = -1,
        biased=False, # use biased classifier if True (only works when num_classes > 0)
        maxpool = True,
        drop_rate=0.0,
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        avgpool = False,
        transductive_bn=False,
        final_activation=None):
        super().__init__(
            _BasicBlock, [1,1,1,1], [64, 128, 256, 512],
            required_input_size,
            num_classes,
            biased,
            maxpool,
            drop_rate,
            drop_block, 
            drop_size, 
            drop_stablization,
            avgpool,
            transductive_bn,
            final_activation
        )

@_register_model('resnet12_c')
class ResNet12_wC(ResNet_wC):
    """Return Bx#class logits.
    """
    def __init__(self, 
        required_input_size,
        num_classes,
        biased=False,
        maxpool=True,
        drop_rate=0.0,
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        avgpool = False,
        transductive_bn=False,
        final_activation=None):
        super().__init__(
            _BasicBlock, [1,1,1,1], [64, 128, 256, 512],
            required_input_size,
            num_classes,
            biased,
            maxpool,
            drop_rate,
            drop_block, 
            drop_size, 
            drop_stablization,
            avgpool,
            transductive_bn,
            final_activation)

@_register_model('resnet12_c+')
class ResNet12_wCplus(ResNet_wCplus):
    """Return BxCxHxW feature map and Bx#class logits.
    """
    def __init__(self, 
        required_input_size,
        num_classes,
        biased=False,
        maxpool=True,
        drop_rate=0.0,
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        avgpool = False,
        transductive_bn=False,
        final_activation=None):
        super().__init__(
            _BasicBlock, [1,1,1,1], [64, 128, 256, 512],
            required_input_size,
            num_classes,
            biased,
            maxpool,
            drop_rate,
            drop_block, 
            drop_size, 
            drop_stablization,
            avgpool,
            transductive_bn,
            final_activation)

@_register_model('resnet12_dc')
class ResNet12_wDC(ResNet_wDC): # dense classification
    """Return Bx#classxHxW logits.
    """
    def __init__(self, 
        required_input_size,
        num_classes,
        biased=False,
        maxpool=True,
        drop_rate=0.0,
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        avgpool = False,
        transductive_bn=False,
        final_activation=None):
        super().__init__(
            _BasicBlock, [1,1,1,1], [64, 128, 256, 512],
            required_input_size,
            num_classes,
            biased,
            maxpool,
            drop_rate,
            drop_block, 
            drop_size, 
            drop_stablization,
            avgpool,
            transductive_bn,
            final_activation)

@_register_model('resnet12_dc+')
class ResNet12_wDCplus(ResNet_wDCplus):
    """Return BxCxHxW feature map and Bx #class xHxW logits.
    """
    def __init__(self, 
        required_input_size,
        num_classes,
        biased=False,
        maxpool=True,
        drop_rate=0.0,
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        avgpool = False,
        transductive_bn=False,
        final_activation=None):
        super().__init__(
            _BasicBlock, [1,1,1,1], [64, 128, 256, 512],
            required_input_size,
            num_classes,
            biased,
            maxpool,
            drop_rate,
            drop_block, 
            drop_size, 
            drop_stablization,
            avgpool,
            transductive_bn,
            final_activation)

@_register_model('resnet12wider')
class ResNet12wider(ResNet):
    """Return BxCxHxW feature map.
    """
    def __init__(self, 
        required_input_size,
        num_classes = -1,
        biased=False, # use biased classifier if True (only works when num_classes > 0)
        maxpool = True,
        drop_rate=0.0,
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        avgpool = False,
        transductive_bn=False,
        final_activation=None):
        super().__init__(
            _BasicBlock, [1,1,1,1], [64, 160, 320, 640],
            required_input_size,
            num_classes,
            biased,
            maxpool,
            drop_rate,
            drop_block, 
            drop_size, 
            drop_stablization,
            avgpool,
            transductive_bn,
            final_activation
        )

@_register_model('resnet12wider_c')
class ResNet12wider_wC(ResNet_wC):
    """Return Bx#class logits.
    """
    def __init__(self, 
        required_input_size,
        num_classes,
        biased=False,
        maxpool=True,
        drop_rate=0.0,
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        avgpool = False,
        transductive_bn=False,
        final_activation=None):
        super().__init__(
            _BasicBlock, [1,1,1,1], [64, 160, 320, 640],
            required_input_size,
            num_classes,
            biased,
            maxpool,
            drop_rate,
            drop_block, 
            drop_size, 
            drop_stablization,
            avgpool,
            transductive_bn,
            final_activation)

@_register_model('resnet12wider_c+')
class ResNet12wider_wCplus(ResNet_wCplus):
    """Return BxCxHxW feature map and Bx#class logits.
    """
    def __init__(self, 
        required_input_size,
        num_classes,
        biased=False,
        maxpool=True,
        drop_rate=0.0,
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        avgpool = False,
        transductive_bn=False,
        final_activation=None):
        super().__init__(
            _BasicBlock, [1,1,1,1], [64, 160, 320, 640],
            required_input_size,
            num_classes,
            biased,
            maxpool,
            drop_rate,
            drop_block, 
            drop_size, 
            drop_stablization,
            avgpool,
            transductive_bn,
            final_activation)

@_register_model('resnet12wider_dc')
class ResNet12wider_wDC(ResNet_wDC): # dense classification
    """Return Bx#classxHxW logits.
    """
    def __init__(self, 
        required_input_size,
        num_classes,
        biased=False,
        maxpool=True,
        drop_rate=0.0,
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        avgpool = False,
        transductive_bn=False,
        final_activation=None):
        super().__init__(
            _BasicBlock, [1,1,1,1], [64, 160, 320, 640],
            required_input_size,
            num_classes,
            biased,
            maxpool,
            drop_rate,
            drop_block, 
            drop_size, 
            drop_stablization,
            avgpool,
            transductive_bn,
            final_activation)

@_register_model('resnet12wider_dc+')
class ResNet12wider_wDCplus(ResNet_wDCplus):
    """Return BxCxHxW feature map and Bx #class xHxW logits.
    """
    def __init__(self, 
        required_input_size,
        num_classes,
        biased=False,
        maxpool=True,
        drop_rate=0.0,
        drop_block=False, 
        drop_size=1, 
        drop_stablization=1,
        avgpool = False,
        transductive_bn=False,
        final_activation=None):
        super().__init__(
            _BasicBlock, [1,1,1,1], [64, 160, 320, 640],
            required_input_size,
            num_classes,
            biased,
            maxpool,
            drop_rate,
            drop_block, 
            drop_size, 
            drop_stablization,
            avgpool,
            transductive_bn,
            final_activation)

#
def get_model(name):
    T = models.get(name, None)
    if T is None:
        raise ValueError('Model named %s does not exist.' % name)
    return T