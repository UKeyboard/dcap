import os
import numpy
import functools
import torch
import torch.utils.data as torda
import torchvision.transforms as ttf
from collections import namedtuple
from utils import identity
from dataset.miniimagenet import *
from dataset.tieredimagenet import *
from dataset.cifar import *
from dataset.cub200 import *

_MiniImagenetHorizontal = MiniImagenetHorizontal
_MiniImagenetLMDBHorizontal = MiniImagenetLMDBHorizontal
_MiniImagenet = MiniImagenet
_MiniImagenetLMDB = MiniImagenetLMDB
_TieredImagenetHorizontal = TieredImagenetHorizontal
_TieredImagenetLMDBHorizontal = TieredImagenetLMDBHorizontal
_TieredImagenet = TieredImagenet
_TieredImagenetLMDB = TieredImagenetLMDB
_Cifar100Horizontal = Cifar100Horizontal
_Cifar100 = Cifar100
_CUB200Horizontal = CUB200Horizontal
_CUB200 = CUB200
_CUB200LMDBHorizontal = CUB200LMDBHorizontal
_CUB200LMDB = CUB200LMDB

IMAGENET_DATASET_DIR = "/home/huzhou/imageNet"
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGENET_MEAN_STD = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
UNIFORM_MEAN_STD = dict(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
CIFAR_MEAN_STD = dict(mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261))
# CUB200_MEAN_STD = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

FewShotEpisodeMeta = namedtuple('FewShotEpisodeMeta', ['n_class', 'n_support', 'n_query', 'n_unlabel'])

def __label_remapping(labels):
    """Re-mapping a list of labels.

    Args:
    @param labels: list, a iterable list of labels.

    Return:
    torch.LongTensor, A iterable list of new labels.
    """
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else numpy.array(labels)
    # numpy.unique returns a sorted unique values in which case a small
    # label can only have a small label (close to zero) after re-mapping.
    # we use numpy.random to avoid this situation so that a small label 
    # can have a big label after re-mapping.
    U, RI = numpy.unique(labels, return_inverse=True)
    n = len(U)
    return torch.LongTensor((numpy.random.choice(n) + RI) % n)


dataset_transforms = {}
def _register_transform(dataset_name, transform_name):
    """A general register for setting transformation for datasets.

    Args:
    @param dataset_name: str or list, the dataset name(s), e.g. 'MiniImagenetFS' or ['MiniImagenetFS', 'TieredImagenetFS'].
    @param transform_name: str or list, the transformation name(s), e.g. 'train' or ['train', 'test'].
    """
    if isinstance(dataset_name, str): dataset_name = [dataset_name]
    if isinstance(transform_name, str): transform_name = [transform_name]
    assert all(map(lambda x: isinstance(x, str), dataset_name))
    assert all(map(lambda x: isinstance(x, str), transform_name))
    def wrapper(transform_fn):
        _transform = transform_fn()
        for i in dataset_name:
            dts = dataset_transforms.get(i, None)
            if dts is None:
                dataset_transforms[i] = {}
                dts = dataset_transforms[i]
            for j in transform_name: dts[j] = _transform
        return transform_fn
    return wrapper

@_register_transform([
    'MiniImagenetFS', 
    'TieredImagenetFS', 
    'MiniImagenetLMDBFS', 
    'TieredImagenetLMDBFS',
    'CUB200FS',
    'CUB200LMDBFS'], 
    'basic')
def __transform_helper():
    return ttf.Compose([
        ttf.Resize((84,84)),
        ttf.ToTensor(),
        ttf.Normalize(**IMAGENET_MEAN_STD)
    ]), __label_remapping

@_register_transform([
    'MiniImagenetFS', 
    'TieredImagenetFS', 
    'MiniImagenetLMDBFS', 
    'TieredImagenetLMDBFS',
    'CUB200FS',
    'CUB200LMDBFS'], 
    'basic-')
def __transform_helper():
    return ttf.Compose([
        ttf.Resize((84,84)),
        ttf.ToTensor(),
        ttf.Normalize(**UNIFORM_MEAN_STD)
    ]), __label_remapping

@_register_transform([
    'MiniImagenetHorizontal', 
    'TieredImagenetHorizontal', 
    'MiniImagenetLMDBHorizontal', 
    'TieredImagenetLMDBHorizontal',
    'CUB200Horizontal',
    'CUB200LMDBHorizontal'], 
    'basic')
def __transform_helper():
    return ttf.Compose([
        ttf.Resize((84,84)),
        ttf.ToTensor(),
        ttf.Normalize(**IMAGENET_MEAN_STD)
    ]), identity

@_register_transform([
    'MiniImagenetHorizontal', 
    'TieredImagenetHorizontal', 
    'MiniImagenetLMDBHorizontal', 
    'TieredImagenetLMDBHorizontal',
    'CUB200Horizontal',
    'CUB200LMDBHorizontal'], 
    'basic-')
def __transform_helper():
    return ttf.Compose([
        ttf.Resize((84,84)),
        ttf.ToTensor(),
        ttf.Normalize(**UNIFORM_MEAN_STD)
    ]), identity

@_register_transform([
    'MiniImagenetFS',
    'TieredImagenetFS', 
    'MiniImagenetLMDBFS', 
    'TieredImagenetLMDBFS',
    'CUB200FS',
    'CUB200LMDBFS'], 
    'aug_train+')
def __transform_helper():
    return ttf.Compose([
        ttf.RandomResizedCrop((84,84), scale=(0.2, 1.0)),
        ttf.RandomHorizontalFlip(),
        ttf.ToTensor(),
        ttf.Normalize(**IMAGENET_MEAN_STD)
    ]), __label_remapping

@_register_transform([
    'MiniImagenetFS',
    'TieredImagenetFS', 
    'MiniImagenetLMDBFS', 
    'TieredImagenetLMDBFS',
    'CUB200FS',
    'CUB200LMDBFS'], 
    'aug_train')
def __transform_helper():
    return ttf.Compose([
        ttf.Resize((84,84)),
        ttf.RandomHorizontalFlip(),
        ttf.ToTensor(),
        ttf.Normalize(**IMAGENET_MEAN_STD)
    ]), __label_remapping

@_register_transform([
    'MiniImagenetFS',
    'TieredImagenetFS', 
    'MiniImagenetLMDBFS', 
    'TieredImagenetLMDBFS',
    'CUB200FS',
    'CUB200LMDBFS'], 
    'aug_train-')
def __transform_helper():
    return ttf.Compose([
        ttf.Resize((84,84)),
        ttf.RandomHorizontalFlip(),
        ttf.ToTensor(),
        ttf.Normalize(**UNIFORM_MEAN_STD)
    ]), __label_remapping

@_register_transform([
    'MiniImagenetHorizontal', 
    'TieredImagenetHorizontal', 
    'MiniImagenetLMDBHorizontal', 
    'TieredImagenetLMDBHorizontal',
    'CUB200Horizontal',
    'CUB200LMDBHorizontal'], 
    'aug_train+')
def __transform_helper():
    return ttf.Compose([
        ttf.RandomResizedCrop((84,84), scale=(0.2, 1.0)),
        ttf.RandomHorizontalFlip(),
        ttf.ToTensor(),
        ttf.Normalize(**IMAGENET_MEAN_STD)
    ]), identity

@_register_transform([
    'MiniImagenetHorizontal', 
    'TieredImagenetHorizontal', 
    'MiniImagenetLMDBHorizontal', 
    'TieredImagenetLMDBHorizontal',
    'CUB200Horizontal',
    'CUB200LMDBHorizontal'], 
    'aug_train')
def __transform_helper():
    return ttf.Compose([
        ttf.Resize((84,84)),
        ttf.RandomHorizontalFlip(),
        ttf.ToTensor(),
        ttf.Normalize(**IMAGENET_MEAN_STD)
    ]), identity

@_register_transform([
    'MiniImagenetHorizontal', 
    'TieredImagenetHorizontal', 
    'MiniImagenetLMDBHorizontal', 
    'TieredImagenetLMDBHorizontal',
    'CUB200Horizontal',
    'CUB200LMDBHorizontal'], 
    'aug_train-')
def __transform_helper():
    return ttf.Compose([
        ttf.Resize((84,84)),
        ttf.RandomHorizontalFlip(),
        ttf.ToTensor(),
        ttf.Normalize(**UNIFORM_MEAN_STD)
    ]), identity

@_register_transform([
    'CIFARFS100',
    'CIFARFC100',
    ], 'basic')
def __transform_helper():
    return ttf.Compose([
        ttf.Resize((32,32)),
        ttf.ToTensor(),
        ttf.Normalize(**CIFAR_MEAN_STD)
    ]), __label_remapping

@_register_transform([
    'CIFARFS100',
    'CIFARFC100',
    ], 'aug_train')
def __transform_helper():
    return ttf.Compose([
        ttf.Resize((32,32)),
        ttf.RandomHorizontalFlip(),
        ttf.ToTensor(),
        ttf.Normalize(**CIFAR_MEAN_STD)
    ]), __label_remapping

@_register_transform([
    'CIFARFS100',
    'CIFARFC100',
    ], 'aug_train+')
def __transform_helper():
    return ttf.Compose([
        ttf.Resize((32,32)),
        ttf.RandomCrop(32, padding=4),
        ttf.RandomHorizontalFlip(),
        ttf.ToTensor(),
        ttf.Normalize(**CIFAR_MEAN_STD)
    ]), __label_remapping

@_register_transform([
    'CIFARFS100Horizontal',
    'CIFARFC100Horizontal',
    ], 'basic')
def __transform_helper():
    return ttf.Compose([
        ttf.Resize((32,32)),
        ttf.ToTensor(),
        ttf.Normalize(**CIFAR_MEAN_STD)
    ]), identity

@_register_transform([
    'CIFARFS100Horizontal',
    'CIFARFC100Horizontal',
    ], 'aug_train')
def __transform_helper():
    return ttf.Compose([
        ttf.Resize((32,32)),
        ttf.RandomHorizontalFlip(),
        ttf.ToTensor(),
        ttf.Normalize(**CIFAR_MEAN_STD)
    ]), identity

@_register_transform([
    'CIFARFS100Horizontal',
    'CIFARFC100Horizontal',
    ], 'aug_train+')
def __transform_helper():
    return ttf.Compose([
        ttf.Resize((32,32)),
        ttf.RandomCrop(32, padding=4),
        ttf.RandomHorizontalFlip(),
        ttf.ToTensor(),
        ttf.Normalize(**CIFAR_MEAN_STD)
    ]), identity

def default_collate_fn(batch, transform_fn_x, transform_fn_y):
    """The default collation function for building dataset.

    Args:
    @param batch: list, a list of data, each of which is composed of [image, label, label_idx].
    @param transform_fn_x: a callable object about how to prepare input images.
    @param transform_fn_y: a callable object about how to prepare targets.

    Return:
    The well-prepared batch data.
    """
    images, labels, _labels = [], [], []
    for x, y, _y in batch:
        images.append(x)
        labels.append(y)
        _labels.append(_y)
    #
    images = torch.stack([transform_fn_x(im) for im in images])
    _labels = torch.LongTensor(_labels)
    _new_labels = torch.stack(list(map(transform_fn_y, _labels))).reshape(-1)
    return images, _new_labels, _labels


def default_fewshot_collate_fn(batch, fsl_episode_definition, transform_fn_x, transform_fn_y):
    """The default collation function for building fewshot dataset.

    Args:
    @param batch: list, a list of data, each of which is composed of [image, label, label_idx].
    @param fsl_episode_definition: FewShotEpisodeMeta, a definition of a few-shot learning episode.
    @param transform_fn_x: a callable object about how to prepare input images.
    @param transform_fn_y: a callable object about how to prepare targets.

    Return:
    The well-prepared batch data.
    """
    n_class = fsl_episode_definition.n_class
    n_support = fsl_episode_definition.n_support
    n_query = fsl_episode_definition.n_query
    n_unlabel = fsl_episode_definition.n_unlabel
    n_shot = n_support + n_query + n_unlabel
    n_instance_per_episode = n_class * n_shot
    #
    assert len(batch) % n_instance_per_episode == 0
    # n_episode_per_batch = len(batch) / n_instance_per_episode
    images, labels, _labels = [], [], []
    for x, y, _y in batch:
        images.append(x)
        labels.append(y)
        _labels.append(_y)
    #
    images = torch.stack([transform_fn_x(im) for im in images])
    _labels = torch.LongTensor(_labels)
    _fsl_labels = []
    for episode_labels in _labels.view(-1, n_instance_per_episode):
        _fsl_labels.append(transform_fn_y(episode_labels))
    _fsl_labels = torch.stack(_fsl_labels).reshape(-1)
    return images, _fsl_labels, _labels


class FewShotBatchSampler(torda.Sampler):
    """Batch sampler for building few-shot learning dataset.
    Cannot guarantee each one instance in the base dataset is sampled at least once.

    Args:
    @param data_source: Dataset, the base dataset.
    @param num_batch: int, the number of batches.
    @param num_way: int, the number of classes per episode, e.g. 5 in 5-way 1-shot setting.
    @param num_shot: int, the number of instances per class in each episode.
    @param num_episode_per_batch: int, the number of episodes in one batch.
    """
    def __init__(self, data_source, num_batch, num_way, num_shot, num_episode_per_batch=1):
        self.dataset = data_source
        self.num_batch = num_batch
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_episode_per_batch = num_episode_per_batch
    
    def __len__(self):
        return self.num_batch

    def __iter__(self):
        for bi in range(self.num_batch):
            data = []
            for ei in range(self.num_episode_per_batch):
                task = []
                classes = numpy.random.choice(self.dataset.category_pool, self.num_way, replace=False)
                for c in classes:
                    instances = numpy.random.choice(self.dataset.num_of_instances_per_class[c], self.num_shot, replace=False) + self.dataset.indices[c]
                    task.append(torch.from_numpy(instances))
                task = torch.stack(task)
                data.append(task)
            data = torch.stack(data)
            yield data.reshape(-1) # bs * nway * kshot


class FewShotDistributedBatchSampler(torda.Sampler):
    """Batch sampler for building few-shot learning dataset in distributed setting.
    Cannot guarantee each one instance in the base dataset is sampled at least once.

    Args:
    @param data_source: Dataset, the base dataset.
    @param num_batch: int, the number of batches.
    @param num_way: int, the number of classes per episode, e.g. 5 in 5-way 1-shot setting.
    @param num_shot: int, the number of instances per class in each episode.
    @param num_episode_per_batch: int, the number of episodes in one batch.
    """
    def __init__(self, data_source, num_batch, num_way, num_shot, num_episode_per_batch=1):
        self.dataset = data_source
        self.num_batch = num_batch
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_episode_per_batch = num_episode_per_batch
        if not torch.distributed.is_available():
            raise RuntimeError("Requires distributed package to be available.")
        if not torch.distributed.is_initialized():
            raise RuntimeError("Requires distributed progress group initialized first.")
        self.num_replicas = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()
        self.total_episodes_per_batch = self.num_episode_per_batch * self.num_replicas
    
    def __len__(self):
        return self.num_batch

    def __iter__(self):
        for bi in range(self.num_batch):
            data = []
            for ei in range(self.total_episodes_per_batch):
                task = []
                classes = numpy.random.choice(self.dataset.category_pool, self.num_way, replace=False)
                for c in classes:
                    instances = numpy.random.choice(self.dataset.num_of_instances_per_class[c], self.num_shot, replace=False) + self.dataset.indices[c]
                    task.append(torch.from_numpy(instances))
                task = torch.stack(task)
                data.append(task)
            data = torch.stack(data)
            data = data[self.rank:self.total_episodes_per_batch:self.num_replicas]
            assert len(data) == self.num_episode_per_batch
            yield data.reshape(-1) # bs * nway * kshot


class FewShotDataLoader(torda.DataLoader):
    def __init__(self, dataset, n_batch, n_class, n_support, n_query, n_unlabel, batch_size, transform_fn_x, transform_fn_y, distributed=False, **kwargs):
        fsl_episode_definition = FewShotEpisodeMeta._make([n_class, n_support, n_query, n_unlabel])
        if not not distributed:
            batch_sampler = FewShotDistributedBatchSampler(dataset, n_batch, n_class, n_support + n_query + n_unlabel, num_episode_per_batch=batch_size)
        else:
            batch_sampler = FewShotBatchSampler(dataset, n_batch, n_class, n_support + n_query + n_unlabel, num_episode_per_batch=batch_size)
        collate_fn = functools.partial(default_fewshot_collate_fn, fsl_episode_definition=fsl_episode_definition, transform_fn_x=transform_fn_x, transform_fn_y=transform_fn_y)
        kwargs['batch_sampler'] = batch_sampler
        kwargs['collate_fn'] = collate_fn
        kwargs['batch_size'] = 1 # set batch_size=1 to work with default DataLoader
        try:
            super().__init__(dataset, **kwargs)
            self.fsl_episode_definition = fsl_episode_definition
        except ValueError as e:
            # Refine expection message "batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last"
            if 'batch_sampler' in e.message:
                raise ValueError("batch_sampler option is mutually exclusive with shuffle, sampler, and drop_last")
            else:
                raise e


datasets = {}
def _register_dataset(name):
    """A general register for datasets.

    Args:
    @param name: str or list, the dataset alias(s), e.g. 'miniimagenet' or ['mini', 'miniimagenet'].
    """
    if isinstance(name, str): name = [name]
    assert all(map(lambda x: isinstance(x, str), name))
    def wrapper(cls):
        for c in name:
            T = datasets.get(c, None)
            assert T is None
            datasets[c] = cls
        return cls
    return wrapper

@_register_dataset('miniImagenet')
class MiniImagenetFS(FewShotDataLoader):
    def __init__(self, version, phase, transform_name, n_batch, n_class, n_support, n_query, n_unlabel, batch_size=1, label_indent=-1, imagenet=None, distributed=False, **kwargs):
        dataset = _MiniImagenet(MINIIMAGENET_DATASET_DIR, version, phase, label_indent, imagenet)
        transform_fn_x, transform_fn_y = dataset_transforms[self.__class__.__name__][transform_name]
        super().__init__(dataset, n_batch, n_class, n_support, n_query, n_unlabel, batch_size, transform_fn_x, transform_fn_y, distributed=distributed **kwargs)

@_register_dataset('miniImagenetHorizontal')
class MiniImagenetHorizontal(torda.DataLoader):
    def __init__(self, version, phase, transform_name, factor=0.8, category_pool_name=None, label_indent=-1, imagenet=None, distributed=False, **kwargs):
        dataset = _MiniImagenetHorizontal(MINIIMAGENET_DATASET_DIR, version, phase, factor, category_pool_name, label_indent, imagenet)
        transform_fn_x, transform_fn_y = dataset_transforms[self.__class__.__name__][transform_name]
        collate_fn = functools.partial(default_collate_fn, transform_fn_x=transform_fn_x, transform_fn_y=transform_fn_y)
        if not not distributed:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            if not torch.distributed.is_initialized():
                raise RuntimeError("Requires distributed progress group initialized first.")
            sampler = torda.DistributedSampler(dataset)
        else:
            sampler = None
        kwargs['collate_fn'] = collate_fn
        kwargs['sampler'] = sampler
        kwargs['shuffle'] = kwargs.get('shuffle', False) and (sampler is None)
        super().__init__(dataset, **kwargs)

@_register_dataset('tieredImagenet')
class TieredImagenetFS(FewShotDataLoader):
    def __init__(self, phase, transform_name, label_indent, n_batch, n_class, n_support, n_query, n_unlabel, batch_size, imagenet=None, distributed=False, **kwargs):
        dataset = _TieredImagenet(TIEREDIMAGENET_DATASET_DIR, phase, label_indent, imagenet)
        transform_fn_x, transform_fn_y = dataset_transforms[self.__class__.__name__][transform_name]
        super().__init__(dataset, n_batch, n_class, n_support, n_query, n_unlabel, batch_size, transform_fn_x, transform_fn_y, distributed=distributed, **kwargs)

@_register_dataset('tieredImagenetHorizontal')
class TieredImagenetHorizontal(torda.DataLoader):
    def __init__(self, phase, transform_name, factor=0.8, category_pool_name=None, label_indent=-1, imagenet=None, distributed=False, **kwargs):
        dataset = _TieredImagenetHorizontal(TIEREDIMAGENET_DATASET_DIR, phase, factor, category_pool_name, label_indent, imagenet)
        transform_fn_x, transform_fn_y = dataset_transforms[self.__class__.__name__][transform_name]
        collate_fn = functools.partial(default_collate_fn, transform_fn_x=transform_fn_x, transform_fn_y=transform_fn_y)
        if not not distributed:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            if not torch.distributed.is_initialized():
                raise RuntimeError("Requires distributed progress group initialized first.")
            sampler = torda.DistributedSampler(dataset)
        else:
            sampler = None
        kwargs['collate_fn'] = collate_fn
        kwargs['sampler'] = sampler
        kwargs['shuffle'] = kwargs.get('shuffle', False) and (sampler is None)
        super().__init__(dataset, **kwargs)

@_register_dataset('cub200')
class CUB200FS(FewShotDataLoader):
    def __init__(self, phase, transform_name, label_indent, n_batch, n_class, n_support, n_query, n_unlabel, batch_size, cub200dir=None, distributed=False, **kwargs):
        dataset = _CUB200(CUB200_DATASET_DIR, phase, label_indent, cub200dir)
        transform_fn_x, transform_fn_y = dataset_transforms[self.__class__.__name__][transform_name]
        super().__init__(dataset, n_batch, n_class, n_support, n_query, n_unlabel, batch_size, transform_fn_x, transform_fn_y, distributed=distributed, **kwargs)

@_register_dataset('cub200Horizontal')
class CUB200Horizontal(torda.DataLoader):
    def __init__(self, phase, transform_name, factor=0.8, category_pool_name=None, label_indent=-1, cub200dir=None, distributed=False, **kwargs):
        dataset = _CUB200Horizontal(CUB200_DATASET_DIR, phase, factor, category_pool_name, label_indent, cub200dir)
        transform_fn_x, transform_fn_y = dataset_transforms[self.__class__.__name__][transform_name]
        collate_fn = functools.partial(default_collate_fn, transform_fn_x=transform_fn_x, transform_fn_y=transform_fn_y)
        if not not distributed:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            if not torch.distributed.is_initialized():
                raise RuntimeError("Requires distributed progress group initialized first.")
            sampler = torda.DistributedSampler(dataset)
        else:
            sampler = None
        kwargs['collate_fn'] = collate_fn
        kwargs['sampler'] = sampler
        kwargs['shuffle'] = kwargs.get('shuffle', False) and (sampler is None)
        super().__init__(dataset, **kwargs)

@_register_dataset('cifar-fs')
class CIFARFS100(FewShotDataLoader):
    def __init__(self, phase, transform_name, label_indent, n_batch, n_class, n_support, n_query, n_unlabel, batch_size, cifardir=None, distributed=False, **kwargs):
        dataset = _Cifar100(CIFAR_DATASET_DIR, phase, label_indent, cifardir, True, False)
        transform_fn_x, transform_fn_y = dataset_transforms[self.__class__.__name__][transform_name]
        super().__init__(dataset, n_batch, n_class, n_support, n_query, n_unlabel, batch_size, transform_fn_x, transform_fn_y, distributed=distributed, **kwargs)

@_register_dataset('cifar-fs-horizon')
class CIFARFS100Horizontal(torda.DataLoader):
    def __init__(self, phase, transform_name, factor=0.8, category_pool_name=None, label_indent=-1, cifardir=None, distributed=False, **kwargs):
        dataset = _Cifar100Horizontal(CIFAR_DATASET_DIR, phase, factor, category_pool_name, label_indent, cifardir, True, False)
        transform_fn_x, transform_fn_y = dataset_transforms[self.__class__.__name__][transform_name]
        collate_fn = functools.partial(default_collate_fn, transform_fn_x=transform_fn_x, transform_fn_y=transform_fn_y)
        if not not distributed:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            if not torch.distributed.is_initialized():
                raise RuntimeError("Requires distributed progress group initialized first.")
            sampler = torda.DistributedSampler(dataset)
        else:
            sampler = None
        kwargs['collate_fn'] = collate_fn
        kwargs['sampler'] = sampler
        kwargs['shuffle'] = kwargs.get('shuffle', False) and (sampler is None)
        super().__init__(dataset, **kwargs)

@_register_dataset('cifar-fc100')
class CIFARFC100(FewShotDataLoader):
    def __init__(self, phase, transform_name, label_indent, n_batch, n_class, n_support, n_query, n_unlabel, batch_size, cifardir=None, distributed=False, **kwargs):
        dataset = _Cifar100(CIFAR_DATASET_DIR, phase, label_indent, cifardir, True, True)
        transform_fn_x, transform_fn_y = dataset_transforms[self.__class__.__name__][transform_name]
        super().__init__(dataset, n_batch, n_class, n_support, n_query, n_unlabel, batch_size, transform_fn_x, transform_fn_y, distributed=distributed, **kwargs)

@_register_dataset('cifar-fc100-horizon')
class CIFARFC100Horizontal(torda.DataLoader):
    def __init__(self, phase, transform_name, factor=0.8, category_pool_name=None, label_indent=-1, cifardir=None, distributed=False, **kwargs):
        dataset = _Cifar100Horizontal(CIFAR_DATASET_DIR, phase, factor, category_pool_name, label_indent, cifardir, True, True)
        transform_fn_x, transform_fn_y = dataset_transforms[self.__class__.__name__][transform_name]
        collate_fn = functools.partial(default_collate_fn, transform_fn_x=transform_fn_x, transform_fn_y=transform_fn_y)
        if not not distributed:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            if not torch.distributed.is_initialized():
                raise RuntimeError("Requires distributed progress group initialized first.")
            sampler = torda.DistributedSampler(dataset)
        else:
            sampler = None
        kwargs['collate_fn'] = collate_fn
        kwargs['sampler'] = sampler
        kwargs['shuffle'] = kwargs.get('shuffle', False) and (sampler is None)
        super().__init__(dataset, **kwargs)

# register LMDB dataset
@_register_dataset('miniImagenet_lmdb')
class MiniImagenetLMDBFS(FewShotDataLoader):
    def __init__(self, version, phase, transform_name, n_batch, n_class, n_support, n_query, n_unlabel, batch_size=1, label_indent=-1, lmdb=None, distributed=False, **kwargs):
        dataset = _MiniImagenetLMDB(MINIIMAGENET_DATASET_DIR, version, phase, label_indent, lmdb)
        transform_fn_x, transform_fn_y = dataset_transforms[self.__class__.__name__][transform_name]
        super().__init__(dataset, n_batch, n_class, n_support, n_query, n_unlabel, batch_size, transform_fn_x, transform_fn_y, distributed=distributed, **kwargs)

@_register_dataset('miniImagenetHorizontal_lmdb')
class MiniImagenetLMDBHorizontal(torda.DataLoader):
    def __init__(self, version, phase, transform_name, factor=0.8, category_pool_name=None, label_indent=-1, lmdb=None, distributed=False, **kwargs):
        dataset = _MiniImagenetLMDBHorizontal(MINIIMAGENET_DATASET_DIR, version, phase, factor, category_pool_name, label_indent, lmdb)
        transform_fn_x, transform_fn_y = dataset_transforms[self.__class__.__name__][transform_name]
        collate_fn = functools.partial(default_collate_fn, transform_fn_x=transform_fn_x, transform_fn_y=transform_fn_y)
        if not not distributed:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            if not torch.distributed.is_initialized():
                raise RuntimeError("Requires distributed progress group initialized first.")
            sampler = torda.DistributedSampler(dataset)
        else:
            sampler = None
        kwargs['collate_fn'] = collate_fn
        kwargs['sampler'] = sampler
        kwargs['shuffle'] = kwargs.get('shuffle', False) and (sampler is None)
        super().__init__(dataset, **kwargs)

@_register_dataset('tieredImagenet_lmdb')
class TieredImagenetLMDBFS(FewShotDataLoader):
    def __init__(self, phase, transform_name, label_indent, n_batch, n_class, n_support, n_query, n_unlabel, batch_size, lmdb=None, distributed=False, **kwargs):
        dataset = _TieredImagenetLMDB(TIEREDIMAGENET_DATASET_DIR, phase, label_indent, lmdb)
        transform_fn_x, transform_fn_y = dataset_transforms[self.__class__.__name__][transform_name]
        super().__init__(dataset, n_batch, n_class, n_support, n_query, n_unlabel, batch_size, transform_fn_x, transform_fn_y, distributed=distributed, **kwargs)

@_register_dataset('tieredImagenetHorizontal_lmdb')
class TieredImagenetLMDBHorizontal(torda.DataLoader):
    def __init__(self, phase, transform_name, factor=0.8, category_pool_name=None, label_indent=-1, lmdb=None, distributed=False, **kwargs):
        dataset = _TieredImagenetLMDBHorizontal(TIEREDIMAGENET_DATASET_DIR, phase, factor, category_pool_name, label_indent, lmdb)
        transform_fn_x, transform_fn_y = dataset_transforms[self.__class__.__name__][transform_name]
        collate_fn = functools.partial(default_collate_fn, transform_fn_x=transform_fn_x, transform_fn_y=transform_fn_y)
        if not not distributed:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            if not torch.distributed.is_initialized():
                raise RuntimeError("Requires distributed progress group initialized first.")
            sampler = torda.DistributedSampler(dataset)
        else:
            sampler = None
        kwargs['collate_fn'] = collate_fn
        kwargs['sampler'] = sampler
        kwargs['shuffle'] = kwargs.get('shuffle', False) and (sampler is None)
        super().__init__(dataset, **kwargs)


@_register_dataset('cub200_lmdb')
class CUB200LMDBFS(FewShotDataLoader):
    def __init__(self, phase, transform_name, label_indent, n_batch, n_class, n_support, n_query, n_unlabel, batch_size, lmdb=None, distributed=False, **kwargs):
        dataset = _CUB200LMDB(CUB200_DATASET_DIR, phase, label_indent, lmdb)
        transform_fn_x, transform_fn_y = dataset_transforms[self.__class__.__name__][transform_name]
        super().__init__(dataset, n_batch, n_class, n_support, n_query, n_unlabel, batch_size, transform_fn_x, transform_fn_y, distributed=distributed, **kwargs)

@_register_dataset('cub200Horizontal_lmdb')
class CUB200LMDBHorizontal(torda.DataLoader):
    def __init__(self, phase, transform_name, factor=0.8, category_pool_name=None, label_indent=-1, lmdb=None, distributed=False, **kwargs):
        dataset = _CUB200LMDBHorizontal(CUB200_DATASET_DIR, phase, factor, category_pool_name, label_indent, lmdb)
        transform_fn_x, transform_fn_y = dataset_transforms[self.__class__.__name__][transform_name]
        collate_fn = functools.partial(default_collate_fn, transform_fn_x=transform_fn_x, transform_fn_y=transform_fn_y)
        if not not distributed:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            if not torch.distributed.is_initialized():
                raise RuntimeError("Requires distributed progress group initialized first.")
            sampler = torda.DistributedSampler(dataset)
        else:
            sampler = None
        kwargs['collate_fn'] = collate_fn
        kwargs['sampler'] = sampler
        kwargs['shuffle'] = kwargs.get('shuffle', False) and (sampler is None)
        super().__init__(dataset, **kwargs)

#
def get_dataset(name):
    T = datasets.get(name, None)
    if T is None:
        raise ValueError('Dataset named %s does not exist.' % name)
    return T