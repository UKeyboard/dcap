import os
import csv
import pickle
import random
import numpy
import itertools
import torch
import torch.utils.data as torda
from torchvision.datasets.cifar import CIFAR100 as CIFAR100
from PIL import Image

# Like miniImagenet and tieredImagenet, there exist two datasets derived from CIFAR100 dataset, i.e. CIFAR-FS & FC100.
# CIFAR-FS is proposed in paper 'Meta-learning with differentiable closedform solvers'.
# FC100 is proposed in paper 'TADAM: Task dependent adaptive metric for improved few-shot learning'.
#
# The meta files of CIFAR-FS are '<train|val|test>.txt'.
# The meta files of FC100 are '<train|val|test>_tiered.txt'

__all__ = ['Cifar100Horizontal', 'Cifar100']

class Cifar100Virtual(torda.Dataset):
    """A virtual class for cifar100 dataset.

    Args:  
    @param root: str, the directory where the meta files locate. 
    @param cifardir: str, the destination where cifar100 dataset locates, if None, assume cifardir=root.
    @param download: bool, If true, downloads the dataset from the internet and
        puts it in cifardir directory. If dataset is already downloaded, it is not
        downloaded again.
    @param is_tiered, bool, If true, use tiered CIFAR100 splits.
    """
    def __init__(self, root, cifardir=None, download=True, is_tiered=False):
        self.root = root
        self.cifardir = cifardir or root
        self.download = not not download
        self.is_tiered = not not is_tiered
        train_batch = CIFAR100(self.cifardir, train=True, download=self.download)
        test_batch = CIFAR100(self.cifardir, train=False, download=self.download)
        self._classes = train_batch.classes
        self._class_to_idx = train_batch.class_to_idx
        self.test_start_idx = len(train_batch)
        self.data = numpy.concatenate((train_batch.data, test_batch.data), axis=0)
        self.targets = numpy.concatenate((train_batch.targets, test_batch.targets), axis=0)
        self.loading()


    def loading(self):
        self.train_cls_num, self.val_cls_num, self.test_cls_num = (60, 20, 20) if self.is_tiered else (64,16,20)
        with open(os.path.join(self.root, 'train_tiered.txt' if self.is_tiered else 'train.txt'), 'r') as f:
            self.train_cls = [x.strip() for x in f]
        with open(os.path.join(self.root, 'val_tiered.txt' if self.is_tiered else 'val.txt'), 'r') as f:
            self.val_cls = [x.strip() for x in f]
        with open(os.path.join(self.root, 'test_tiered.txt' if self.is_tiered else 'test.txt'), 'r') as f:
            self.test_cls = [x.strip() for x in f]
        self.classes = self.train_cls + self.val_cls + self.test_cls
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        # assert len(self.classes) == (self.train_cls_num + self.val_cls_num + self.test_cls_num)
        assert len(self.train_cls) == self.train_cls_num
        assert len(self.val_cls) == self.val_cls_num
        assert len(self.test_cls) == self.test_cls_num
        mappings = {}
        for i in range(len(self.targets)):
            k = self._classes[self.targets[i]]
            items = mappings.get(k, None)
            if items is None:
                mappings[k] = []
                items = mappings[k]
            items.append(i)
        self.mappings = mappings
        self.num_of_instances_per_class = dict(zip(self.classes, [len(self.mappings[c]) for c in self.classes]))
    
    def load_image(self,i):
        """Load PIL image.

        Args:
        @param i: int, the index of the target image

        Return:
        The corresponding PIL Image object.
        """
        return Image.fromarray(self.data[i]).convert("RGB")

class Cifar100Horizontal(Cifar100Virtual):
    """Partitioning cifar100 horizontally in instance level, in which some out of the 600 
    instances (per category) are used for classifier training and the remaining are used for 
    evaluation. The catetories that can participate this kind of partition is strictlly controlled
    by "category_pool_name":
        - "train": use the 64 meta-train categories.
        - "val": use the 16 meta-val categories.
        - "test": use the 20 meta-test categories.
        - "trainval": use both meta-train and meta-val categories.
        - "notrain": use both meta-val and meta-test categories.
        - "all": use all 100 categories.
        - None: use all 100 categories.
    
    The dataset is divided into two splits, i.e. "train" and "test". No "val" split is available.
    The number of training instances per category is #INSTANCE_IN_CLASS*factor.

    Args:
    @param root: str, the directory where the meta files and images of miniImagenet dataset locate.
    @param phase: str, the subset to use, must be one of "train" and "test".
    @param factor: float, the ratio of instance for training, must in range (0,1.0), default 0.8.
    @param category_pool_name: str, a choice of the categories to construct this dataset.
    @param label_indent: int, the reserved label space, default -1 (0 is reserved for background).
    @param cifardir: str, the destination where cifar100 dataset locates, if None, assume cifardir=root.
    @param download: bool, If true, downloads the dataset from the internet and
        puts it in cifardir directory. If dataset is already downloaded, it is not
        downloaded again.
    @param is_tiered, bool, If true, use tiered CIFAR100 splits.
    """
    def __init__(self, root, phase, factor=0.8, category_pool_name=None, label_indent=-1, cifardir=None, download=True, is_tiered=False):
        assert phase in ['train', 'test']
        assert factor > 0 and factor < 1
        if category_pool_name is not None:
            assert category_pool_name in ['train', 'val', 'test', 'trainval', 'notrain', 'all']
        else:
            category_pool_name = 'all'
        assert isinstance(label_indent, int)
        assert label_indent < 1
        super().__init__(root, cifardir=cifardir, download=download, is_tiered=is_tiered)
        self.phase = phase
        self.factor = factor
        self.category_pool_name = category_pool_name
        self.category_pool = self._category_pool
        self.label_indent = label_indent
        #
        instances = []
        for c in self.category_pool:
            i = int(self.factor * self.num_of_instances_per_class[c])
            instances.append(self.mappings[c][:i] if 'train' in self.phase else self.mappings[c][i:])
        self.indices = dict(zip(self.category_pool, numpy.cumsum([0]+[len(items) for items in instances]).tolist()[:-1]))
        self.instances = list(itertools.chain.from_iterable(instances))
        self.num_instances = len(self.instances)
        self.category2idx = dict(zip(self.category_pool, range(len(self.category_pool))))

    @property
    def _category_pool(self):
        if self.category_pool_name == 'all':
            return self.classes[:]
        elif self.category_pool_name == 'train':
            return self.train_cls[:]
        elif self.category_pool_name == 'val':
            return self.val_cls[:]
        elif self.category_pool_name == 'test':
            return self.test_cls[:]
        elif self.category_pool_name == 'trainval':
            return self.train_cls + self.val_cls
        elif self.category_pool_name == 'notrain':
            return self.val_cls + self.test_cls
        else:
            raise ValueError('not supported category pool name: %s' % self.category_pool_name)

    def __len__(self):
        return self.num_instances
    
    def __getitem__(self, index):
        i = self.instances[index]
        label = self._classes[self.targets[i]]
        image = self.load_image(i)
        return image, label, self.category2idx[label] - self.label_indent


class Cifar100(Cifar100Virtual):
    """
    Args:
    @param root: str, the directory where the meta files and images of miniImagenet dataset locate.
    @param phase: str, the subset to use, must be one of "train" (64), "val" (16), "trainval" (80) and "test" (20).
    @param label_indent: int, the reserved label space, default -1 (0 is reserved for background).
    @param cifardir: str, the destination where cifar100 dataset locates, if None, assume cifardir=root.
    @param download: bool, If true, downloads the dataset from the internet and
        puts it in cifardir directory. If dataset is already downloaded, it is not
        downloaded again.
    @param is_tiered, bool, If true, use tiered CIFAR100 splits.
    """
    def __init__(self, root, phase, label_indent=-1, cifardir=None, download=True, is_tiered=False):
        assert phase in ['train', 'val', 'trainval', 'test']
        assert isinstance(label_indent, int)
        assert label_indent < 1
        super().__init__(root, cifardir=cifardir, download=download, is_tiered=is_tiered)
        self.phase = phase
        self.category_pool = self._category_pool
        self.label_indent = label_indent
        #
        instances = [self.mappings[c] for c in self.category_pool]
        self.indices = dict(zip(self.category_pool, numpy.cumsum([0]+[len(items) for items in instances]).tolist()[:-1]))
        self.instances = list(itertools.chain.from_iterable(instances))
        self.num_instances = len(self.instances)
        self.category2idx = dict(zip(self.category_pool, range(len(self.category_pool))))
    
    @property
    def _category_pool(self):
        if self.phase == 'train':
            return self.train_cls[:]
        elif self.phase == 'val':
            return self.val_cls[:]
        elif self.phase == 'test':
            return self.test_cls[:]
        elif self.phase == 'trainval':
            return self.train_cls + self.val_cls
        else:
            raise ValueError('not supported dataset split: %s' % self.phase)
    
    def __len__(self):
        return self.num_instances
    
    def __getitem__(self, index):
        i = self.instances[index]
        label = self._classes[self.targets[i]]
        image = self.load_image(i)
        return image, label, self.category2idx[label] - self.label_indent