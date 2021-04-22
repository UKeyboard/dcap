import os
import csv
import pickle
import random
import numpy
import itertools
import torch
import torch.utils.data as torda
from PIL import Image
from dataset.lmdb import VisionLMDBDatabase

#
__all__ = ['MiniImagenetHorizontal', 'MiniImagenet', 'MiniImagenetLMDBHorizontal', 'MiniImagenetLMDB']

class MiniImagenetVirtual(torda.Dataset):
    """A virtual class for miniImagenet dataset.

    Args:  
    @param root: str, the directory where the meta files and images of miniImagenet dataset locate.
    @param version: str, the partition of miniImagenet. There exist two kinds of partition used in 
    the community, i.e. "protonet" and "matchnet". 
    @param imagenet: str, the destination where ImageNet dataset locates, if None, load images from root/images.
    """
    def __init__(self, root, version, imagenet=None):
        self.root = root
        self.version = version
        assert version in ['protonet', 'matchnet']
        self.imagenet = imagenet
        if imagenet is None:
            self.image_root = os.path.join(root, 'images')
        else:
            assert isinstance(imagenet, str)
            if not os.path.isdir(imagenet):
                raise ValueError('Bad ImageNet dataset location: %s' % imagenet)
            self.image_root = imagenet
        self.loading()


    def loading(self):
        fn = getattr(self, self.version+'_loading', None)
        assert fn is not None
        self.train_cls_num, self.val_cls_num, self.test_cls_num = 64, 16, 20
        self.mappings, self.classes = fn()
        assert len(self.classes) == (self.train_cls_num + self.val_cls_num + self.test_cls_num)
        self.train_cls, self.val_cls, self.test_cls = self.class_spliting()
        assert len(self.train_cls) == self.train_cls_num
        assert len(self.val_cls) == self.val_cls_num
        assert len(self.test_cls) == self.test_cls_num
        self.num_of_instances_per_class = dict(zip(self.classes, [len(self.mappings[c]) for c in self.classes]))

    def protonet_loading(self):
        """Refer to paper "Prototypical Networks for Few-shot Learning" for more details.
        """
        serialized_meta_file = os.path.join(self.root, 'protonet.pkl')
        if os.path.isfile(serialized_meta_file):
            with open(serialized_meta_file, 'rb') as f:
                mapping, classes = pickle.load(f)
        else:
            _mapping = {}
            with open(os.path.join(self.root, 'mapping.csv'), mode='r') as csvfin:
                csvreadin = csv.reader(csvfin, delimiter=',')
                next(csvreadin, None)
                for i, row in enumerate(csvreadin):
                    _mapping[row[0]] = row[1]
            #
            mapping = {}
            classes = []
            for metafile in map(lambda x: os.path.join(self.root, x), ['train.csv', 'val.csv', 'test.csv']):
                with open(metafile, mode='r') as csvfin:
                    csvreadin = csv.reader(csvfin, delimiter=',')
                    next(csvreadin, None)
                    for i, row in enumerate(csvreadin):
                        if row[1] not in mapping:
                            mapping[row[1]] = []
                            classes.append(row[1])
                        mapping[row[1]].append((_mapping[row[0]], row[1]))
            with open(serialized_meta_file, 'wb') as f:
                pickle.dump((mapping, classes), f)
        #
        return mapping, classes


    def matchnet_loading(self):
        """Refer to paper "Matching Networks for One Shot Learning" for more details.
        """
        serialized_meta_file = os.path.join(self.root, 'matchnet.pkl')
        if os.path.isfile(serialized_meta_file):
            with open(serialized_meta_file, 'rb') as f:
                mapping, classes = pickle.load(f)
        else:
            mapping = {}
            classes = []
            metafile = os.path.join(self.root, "miniimagenet.txt")
            with open(metafile, mode='r') as fin:
                label = ""
                while True:
                    line = fin.readline()
                    if line == "": break
                    line = line.strip()
                    if line == "": continue
                    if line.startswith("data/imagenet"):
                        label = os.path.basename(line[:-2])
                        if label not in mapping: 
                            mapping[label]=[]
                            classes.append(label)
                    else:
                        icls,_ = line.split(".")[0].split("_")
                        mapping[icls].append((line, icls))
            with open(serialized_meta_file, 'wb') as f:
                pickle.dump((mapping, classes), f)
        #
        return mapping, classes

    def class_spliting(self):
        a, b, c = self.train_cls_num, self.train_cls_num+self.val_cls_num, self.train_cls_num+self.val_cls_num+self.test_cls_num
        return self.classes[:a], self.classes[a:b], self.classes[b:c]
    
    def load_image(self,x):
        """Load PIL image.

        Args:
        @param x: str, the image name, e.g. 'n01614925_1001.JPEG'

        Return:
        The corresponding PIL Image object.
        """
        label,_ = x.split(".")[0].split("_")
        filename = os.path.join(self.image_root, label, x) # load image from /path/to/miniimagenet/images/category/name
        return Image.open(filename).convert("RGB")

class MiniImagenetHorizontal(MiniImagenetVirtual):
    """Partitioning miniImagenet horizontally in instance level, in which some out of the 600 
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
    @param version: str, the partition of miniImagenet. There exist two kinds of partition used in 
    the community, i.e. "protonet" and "matchnet".
    @param phase: str, the subset to use, must be one of "train" and "test".
    @param factor: float, the ratio of instance for training, must in range (0,1.0), default 0.8.
    @param category_pool_name: str, a choice of the categories to construct this dataset.
    @param label_indent: int, the reserved label space, default -1 (0 is reserved for background).
    @param imagenet: str, the destination where ImageNet dataset locates, if None, load images from root/images.
    """
    def __init__(self, root, version, phase, factor=0.8, category_pool_name=None, label_indent=-1, imagenet=None):
        assert phase in ['train', 'test']
        assert factor > 0 and factor < 1
        if category_pool_name is not None:
            assert category_pool_name in ['train', 'val', 'test', 'trainval', 'notrain', 'all']
        else:
            category_pool_name = 'all'
        assert isinstance(label_indent, int)
        assert label_indent < 1
        super().__init__(root, version, imagenet)
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
        image_name, label = self.instances[index]
        image = self.load_image(image_name)
        return image, label, self.category2idx[label] - self.label_indent


class MiniImagenet(MiniImagenetVirtual):
    """
    Args:
    @param root: str, the directory where the meta files and images of miniImagenet dataset locate.
    @param version: str, the partition of miniImagenet. There exist two kinds of partition used in 
    the community, i.e. "protonet" and "matchnet".
    @param phase: str, the subset to use, must be one of "train" (64), "val" (16), "trainval" (80) and "test" (20).
    @param label_indent: int, the reserved label space, default -1 (0 is reserved for background).
    @param imagenet: str, the destination where ImageNet dataset locates, if None, load images from root/images.
    """
    def __init__(self, root, version, phase, label_indent=-1, imagenet=None):
        assert phase in ['train', 'val', 'trainval', 'test']
        assert isinstance(label_indent, int)
        assert label_indent < 1
        super().__init__(root, version, imagenet)
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
        image_name, label = self.instances[index]
        image = self.load_image(image_name)
        return image, label, self.category2idx[label] - self.label_indent




# LMDB for miniImagenet
class MiniImagenetLMDBHorizontal(MiniImagenetHorizontal):
    """LMDB version of MiniImagenetHorizontal. Assume the miniImagenet images are
    serialized in the target `lmdb` database. A valid lmdb database is required.
    """
    def __init__(self, root, version, phase, factor=0.8, category_pool_name=None, label_indent=-1, lmdb=None):
        super().__init__(root, version, phase, factor, category_pool_name, label_indent, imagenet=None)
        self.lmdb = VisionLMDBDatabase(lmdb)
    
    def load_image(self, x):
        """Load PIL image.

        Args:
        @param x: str, the image name, e.g. 'n01614925_1001.JPEG'

        Return:
        The corresponding PIL Image object.
        """
        label,_ = x.split(".")[0].split("_")
        key = os.path.join(label, x).encode('ascii')
        return self.lmdb.getitem_by_key(key)


class MiniImagenetLMDB(MiniImagenet):
    """LMDB version of MiniImagenet. Assume the miniImagenet images are
    serialized in the target `lmdb` database. A valid lmdb database is required.
    """
    def __init__(self, root, version, phase, label_indent=-1, lmdb=None):
        super().__init__(root, version, phase, label_indent, imagenet=None)
        self.lmdb = VisionLMDBDatabase(lmdb)
    
    def load_image(self, x):
        """Load PIL image.

        Args:
        @param x: str, the image name, e.g. 'n01614925_1001.JPEG'

        Return:
        The corresponding PIL Image object.
        """
        label,_ = x.split(".")[0].split("_")
        key = os.path.join(label, x).encode('ascii')
        return self.lmdb.getitem_by_key(key)