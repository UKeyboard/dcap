import io
import os
import csv
import pickle
import random
import numpy
import itertools
import functools
import torch
import torch.utils.data as torda
from PIL import Image
from dataset.lmdb import VisionLMDBDatabase

#
__all__ = ['CUB200Horizontal', 'CUB200', 'CUB200LMDBHorizontal', 'CUB200LMDB']
    

class CUB200Virtual(torda.Dataset):
    """A virtual class for CUB-200-2011 dataset.

    Args:  
    @param root: str, the directory where the meta files and images of CUB-200-2011 dataset locate.
    @param cub200dir: str, the destination where CUB-200-2011 dataset locates, if None, load images from root/images.
    """
    def __init__(self, root, cub200dir=None, init_hook=None):
        self.root = root
        self.cub200dir = cub200dir or root
        self.image_root = os.path.join(self.cub200dir, 'images')
        self.init_hook = init_hook or CUB200Virtual.prepare_cub200
        self.loading()
    
    def prepare_cub200(self):
        with open(os.path.join(self.cub200dir, 'images.txt'), 'r') as f:
            self.data = [x.strip().split()[1] for x in f]
        with open(os.path.join(self.cub200dir, 'image_class_labels.txt'), 'r') as f:
            self.targets = [int(x.strip().split()[1])-1 for x in f] # convert to zero-based
        with open(os.path.join(self.cub200dir, 'classes.txt'), 'r') as f:
            classes = [x.strip().split() for x in f]
            self._classes = [c[1] for c in classes]
            self._class_to_idx = dict([(c[1], int(c[0])-1) for c in classes])

    def loading(self):
        self.init_hook(self)
        self.train_cls_num, self.val_cls_num, self.test_cls_num = 100, 50, 50
        with open(os.path.join(self.root, 'train.txt'), 'r') as f:
            self.train_cls = [x.strip() for x in f]
        with open(os.path.join(self.root, 'val.txt'), 'r') as f:
            self.val_cls = [x.strip() for x in f]
        with open(os.path.join(self.root, 'test.txt'), 'r') as f:
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
        @param i: int, the index of the target image.

        Return:
        The corresponding PIL Image object.
        """
        imname = self.data[i]
        filename = os.path.join(self.image_root, imname) # load image from /path/to/CUB-200-2011/images/category/name
        return Image.open(filename).convert("RGB")

class CUB200Horizontal(CUB200Virtual):
    """Partitioning CUB-200-2011 horizontally in instance level, in which some 
    instances (per category) are used for classifier training and the remaining are used for 
    evaluation. The catetories that can participate this kind of partition is strictlly controlled
    by "category_pool_name":
        - "train": use the 100 meta-train categories.
        - "val": use the 50 meta-val categories.
        - "test": use the 50 meta-test categories.
        - "trainval": use both meta-train and meta-val categories.
        - "notrain": use both meta-val and meta-test categories.
        - "all": use all 200 categories.
        - None: use all 200 categories.
    
    The dataset is divided into two splits, i.e. "train" and "test". No "val" split is available.
    The number of training instances per category is #INSTANCE_IN_CLASS*factor.

    Args:
    @param root: str, the directory where the meta files and images of CUB-200-2011 dataset locate.
    @param phase: str, the subset to use, must be one of "train" and "test".
    @param factor: float, the ratio of instance for training, must in range (0,1.0), default 0.8.
    @param category_pool_name: str, a choice of the categories to construct this dataset.
    @param label_indent: int, the reserved label space, default -1 (0 is reserved for background).
    @param cub200dir: str, the destination where CUB-200-2011 dataset locates, if None, load images from root/images.
    """
    def __init__(self, root, phase, factor=0.8, category_pool_name=None, label_indent=-1, cub200dir=None, init_hook=CUB200Virtual.prepare_cub200):
        assert phase in ['train', 'test']
        assert factor > 0 and factor < 1
        if category_pool_name is not None:
            assert category_pool_name in ['train', 'val', 'test', 'trainval', 'notrain', 'all']
        else:
            category_pool_name = 'all'
        assert isinstance(label_indent, int)
        assert label_indent < 1
        super().__init__(root, cub200dir, init_hook)
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


class CUB200(CUB200Virtual):
    """
    Args:
    @param root: str, the directory where the meta files and images of CUB-200-2011 dataset locate.
    @param phase: str, the subset to use, must be one of "train" (100), "val" (50), "trainval" (150) and "test" (50).
    @param label_indent: int, the reserved label space, default -1 (0 is reserved for background).
    @param cub200dir: str, the destination where CUB-200-2011 dataset locates, if None, load images from root/images.
    """
    def __init__(self, root, phase, label_indent=-1, cub200dir=None, init_hook=CUB200Virtual.prepare_cub200):
        assert phase in ['train', 'val', 'trainval', 'test']
        assert isinstance(label_indent, int)
        assert label_indent < 1
        super().__init__(root, cub200dir, init_hook)
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

class CUB200LMDBDatabase(VisionLMDBDatabase):
    annotations2idx = dict(zip(['class', 'bbox', 'seg', 'parts', 'attr'], [0,1,2,3,4]))
    def __init__(self, lmdb_file, targets=['class', 'bbox', 'seg', 'parts', 'attr']):
        """
        @param lmdb_file, str, the source LMDB database file.
        @param targets, Iterable, the target labels to retrieve:
            - 'class', image class label.
            - 'bbox', image bounding box label.
            - 'seg', image segmentation label. (only when segmentation in database)
            - 'parts', part locations label in image.
            - 'attr', attribute labels in image.
        """
        super().__init__(lmdb_file)
        #
        with self.db_env.begin(write=False) as txn:
            # load meta: __meta__/classes.txt
            class_txt = txn.get(b'__meta__/classes.txt')
            db_classes = [line.decode('ascii').strip().split()[1] for line in io.BytesIO(class_txt).readlines()]
            db_classes_to_idx = dict([(db_classes[i], i) for i in range(len(db_classes))])
            self.db_classes = db_classes
            self.db_classes_to_idx = db_classes_to_idx
            # load annotation: __annotation__/image_class_labels.txt
            image_class_labels_txt = txn.get(b'__annotation__/image_class_labels.txt')
            db_class_labels = [int(line.decode('ascii').strip().split()[1])-1 for line in io.BytesIO(image_class_labels_txt).readlines()]  # convert ot zero-indexed
            self.db_class_labels = db_class_labels
            # load annotation: __annotation__/bounding_boxes.txt
            bounding_boxes_txt = txn.get(b'__annotation__/bounding_boxes.txt')
            db_bboxes = [None] * len(self)
            for line in io.BytesIO(bounding_boxes_txt).readlines():
                try:
                    iid, x, y, w, h = line.decode('ascii').strip().split()
                except ValueError as e:
                    continue
                iid = int(iid)-1 # convert ot zero-indexed
                _bbox = tuple(map(float, (x,y,w,h)))
                _bboxes = db_bboxes[iid]
                if _bboxes is None:
                    db_bboxes[iid] = []
                    _bboxes = db_bboxes[iid]
                _bboxes.append(_bbox)
            self.db_bboxes = db_bboxes
            # load annotation: __annotation__/parts/part_locs.txt
            part_locs_txt = txn.get(b'__annotation__/parts/part_locs.txt')
            db_part_locs = [None] * len(self)
            for line in io.BytesIO(part_locs_txt).readlines():
                try:
                    iid, pid, x, y, is_visible = line.decode('ascii').strip().split()
                except ValueError as e:
                    continue
                iid, pid = int(iid)-1, int(pid)-1 # convert ot zero-indexed
                x,y = float(x), float(y)
                is_visible = bool(int(is_visible))
                _part = (pid, x, y, is_visible)
                _parts = db_part_locs[iid]
                if _parts is None:
                    db_part_locs[iid] = []
                    _parts = db_part_locs[iid]
                _parts.append(_part)
            self.db_part_locs = db_part_locs
            # load annotation: __annotation__/attributes/image_attribute_labels.txt
            image_attribute_labels_txt = txn.get(b'__annotation__/attributes/image_attribute_labels.txt')
            db_attribute_labels = [None] * len(self)
            for line in io.BytesIO(image_attribute_labels_txt).readlines():
                try:
                    iid, aid, is_present, cid, utime = line.decode('ascii').strip().split()
                except ValueError as e:
                    continue
                iid, aid, cid = int(iid)-1, int(aid)-1, int(cid)-1 # convert ot zero-indexed
                is_present, utime = bool(int(is_present)), float(utime)
                _attr = (aid, is_present, cid, utime)
                _attrs = db_attribute_labels[iid]
                if _attrs is None:
                    db_attribute_labels[iid] = []
                    _attrs = db_attribute_labels[iid]
                _attrs.append(_attr)
            self.db_attribute_labels = db_attribute_labels
        #
        # must in ['class', 'bbox', 'seg', 'parts', 'attr'] order
        self.annotations = [self.db_class_labels, self.db_bboxes, None, self.db_part_locs, self.db_attribute_labels]
        if targets is not None:
            # segmentation is not supported yet.
            # assert 'seg' not in targets
            if 'seg' in targets:
                raise ValueError('segmentation labels are not available yet, the supported labels are: %s' % list(filter(lambda x: x != 'seg', self.annotations2idx.keys())))
            if not all([t in self.annotations2idx for t in targets]):
                raise ValueError('the required annotation must be one or multi-selection of %s.'% self.annotations2idx.keys())
        self.targets = targets
            
    def __getitem__(self, index):
        if self.targets is not None:
            return self.getitem_by_key(self.db_keys[index]), *(self.annotations[self.annotations2idx[t]][index] for t in self.targets)
        else:
            return self.getitem_by_key(self.db_keys[index])


# LMDB for CUB-200-2011
class CUB200LMDBHorizontal(CUB200Horizontal):
    """LMDB version of CUB200Horizontal. Assume the CUB-200-2011 images are
    serialized in the target `lmdb` database. A valid lmdb database is required.
    """
    def __init__(self, root, phase, factor=0.8, category_pool_name=None, label_indent=-1, lmdb=None):
        super().__init__(root, phase, factor, category_pool_name, label_indent, cub200dir=None, init_hook=functools.partial(CUB200LMDBHorizontal.prepare_cub200, lmdb=lmdb))
    
    def prepare_cub200(self, lmdb):
        self.lmdb = CUB200LMDBDatabase(lmdb, targets=['class', 'bbox'])
        self.data = self.lmdb.db_keys
        self.targets = self.lmdb.db_class_labels
        self._classes = self.lmdb.db_classes
        self._class_to_idx = self.lmdb.db_classes_to_idx

    def load_image(self,i):
        """Load PIL image.

        Args:
        @param i: int, the index of the target image.

        Return:
        The corresponding PIL Image object.
        """
        key = self.data[i]
        return self.lmdb.getitem_by_key(key)

class CUB200LMDB(CUB200):
    """LMDB version of CUB200. Assume the CUB-200-2011 images are
    serialized in the target `lmdb` database. A valid lmdb database is required.
    """
    def __init__(self, root, phase, label_indent=-1, lmdb=None):
        super().__init__(root, phase, label_indent, cub200dir=None, init_hook=functools.partial(CUB200LMDB.prepare_cub200, lmdb=lmdb))
    
    def prepare_cub200(self, lmdb):
        self.lmdb = CUB200LMDBDatabase(lmdb, targets=['class', 'bbox'])
        self.data = self.lmdb.db_keys
        self.targets = self.lmdb.db_class_labels
        self._classes = self.lmdb.db_classes
        self._class_to_idx = self.lmdb.db_classes_to_idx

    def load_image(self,i):
        """Load PIL image.

        Args:
        @param i: int, the index of the target image.

        Return:
        The corresponding PIL Image object.
        """
        key = self.data[i]
        return self.lmdb.getitem_by_key(key)