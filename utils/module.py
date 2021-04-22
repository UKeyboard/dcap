import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

identity = lambda x: x

class Identity(nn.Module):
    def forward(self, x):
        return x

class Negative(nn.Module):
    def forward(self, x):
        return -x

class DropBlock(nn.Module):
    """
    DropBlock: A regularization method for convolutional networks
    """
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x, gamma):
        if self.training:
            assert x.dim()==4
            N,C,H,W = x.size()
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((N, C, H, W))
            mask = mask.to(x.device)
            block_mask = self._compute_block_mask(mask)
            inv_blk_mask = 1 - block_mask
            countM = inv_blk_mask.size()[0] * inv_blk_mask.size()[1] * inv_blk_mask.size()[2] * inv_blk_mask.size()[3]
            count_ones = inv_blk_mask.sum()
            return inv_blk_mask * x * (countM / count_ones)
        else:
            return x
    
    def _compute_block_mask(self, mask):
        # assert mask.dim()==4
        N, C, H, W = mask.size()
        non_zero_idxs = mask.nonzero()
        n_blocks = len(non_zero_idxs)
        block_mask = mask + 0
        
        if n_blocks > 0 and self.block_size>1:
            offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                torch.arange(self.block_size).repeat(self.block_size),
            ]
            ).t()
            offsets = torch.cat((torch.zeros(self.block_size**2, 2).long(), offsets.long()), 1).to(mask.device)
            block_idxs = (non_zero_idxs.unsqueeze(dim=-2) + offsets.unsqueeze(dim=0)).view(-1,4)
            block_idxs = block_idxs[block_idxs[:,2]<H]
            block_idxs = block_idxs[block_idxs[:,3]<W]
            block_mask[block_idxs[:,0],block_idxs[:,1],block_idxs[:,2],block_idxs[:,3]] += 1
            block_mask[block_mask>1] = 1

        return block_mask


class DropoutAUG(nn.Module):
    """A augmented dropout.

    Args:
    @param: p (float), probability of an element to be zeroed, default 0.5.
    @param: inplace (bool), if set to True, will do this Dropout in-place, default False.
    @param: drop_block (bool), if set to True, will enable DropBlock, default False.
    @param: drop_size (int), size of block that will be zeroed out in DropBlock, default 1.
    @param: drop_stablization (float), a statblization hyperparameter, default 1.
    """
    def __init__(self, p=0.5, inplace=False, 
        drop_block=False,
        drop_size=1,
        drop_stablization=1
    ):
        super().__init__()
        assert drop_size > 0
        assert drop_stablization > 0
        self.drop_rate = p
        self.drop_inplace = not not inplace
        self.drop_size = drop_size
        self.drop_stablization = drop_stablization
        self.drop_block = not not drop_block
        self.num_batches_tracked = 0
        if self.drop_block and self.drop_size > 1 and self.drop_rate > 0:
            self.DropBlock = DropBlock(block_size=self.drop_size)
    
    def forward(self, x):
        if self.drop_rate > 0:
            self.num_batches_tracked += 1
            if self.drop_block and self.drop_size > 1:
                # apply DropBlock
                N,C,H,W = x.shape
                keep_rate = max(
                    1.0 - self.drop_rate/float(self.drop_stablization)*self.num_batches_tracked,
                    1.0 - self.drop_rate
                )
                gamma = (1.0 - keep_rate) / float(self.drop_size**2) * float(H) * float(W) / float(H - self.drop_size + 1) / float(W - self.drop_size + 1)
                return self.DropBlock(x, gamma)
            else:
                return F.dropout(x, p=self.drop_rate, training=self.training, inplace=self.drop_inplace)
        return x


class LinearClassifier(nn.Linear): pass

class DenseClassifier(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        hidden_features = int(numpy.sqrt(in_features*out_features)/2.0)
        self.linear = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features, bias)
        )
    def forward(self, x):
        return self.linear(x)


class CosineClassifier(nn.Module):
    def __init__(self, in_features, out_features, temperature=10.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.Tensor(out_features, in_features))
        self.tau = nn.parameter.Parameter(torch.Tensor([temperature]))
        if not not bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.kaiming_normal_(self.weight, a=numpy.sqrt(5.))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / numpy.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.tau * F.linear(F.normalize(x,p=2,dim=-1), F.normalize(self.weight, p=2, dim=1), self.bias)


class ProjectionClassifier(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.Tensor(out_features, in_features))
        if not not bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.kaiming_normal_(self.weight, a=numpy.sqrt(5.))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / numpy.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.linear(x, F.normalize(self.weight, p=2, dim=1), self.bias)