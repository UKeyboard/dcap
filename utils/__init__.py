import os
import torch
import torch.nn as nn
import torchvision
import importlib
import math
import pylab
import lmdb
import pickle
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

identity = lambda x: x

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_random_seeds(seed=0):
    """Init a random generator.
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def repeatc(c, n):
    """Repeat the character for number of n times.
    """
    return ''.join([c]*n)

def import_module(modulename, package=None):
    """
    Dynamically import module from package. 
    A python module is actually a python document, e.g. `*.py`.
    This function tries to load package/modulename.py to make
    all defined in the module file available.

    Args:\n
    - modulename: str, the module name, e.g. *.py.
    - package: str, the package name.

    Output:\n
    the loaded module or None if module not found
    """
    try:
        if package is not None:
            modulename = ("" if modulename[0] == "." else ".") + modulename
        mod = importlib.import_module(modulename, package)
        return mod
    except ModuleNotFoundError as e:
        return None
    except IndexError as e:
        return None

def import_x(x, module):
    """
    Dynamically import x from python module.

    Args:\n
    - x: str, the target 
    - module: module, the python module from which to import x.

    Output:\n
    x or None if not found
    """
    if x is None: return None
    if module is None: return None
    return getattr(module, x, None)

def init_weights(model):
    """
    Init weights and bias of an neurual network model.

    Args:\n
    - model, nn.Module, an neural network module instance.
    """
    if isinstance(model, nn.Module):
        m = model
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find("Norm2d") != -1:
            m.weight.data.fill_(1.)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find("Linear") != -1:
            m.weight.data.normal_(0.,0.01)
            if m.bias is not None: 
                m.bias.data.zero_()

def plot_batch(batch, normalization, target=None, force=False):
    """Plot a batch of images.
    Assume the images have already been normalized with normalization parameters.

    Args:
    @param batch: torch.FloatTensor, a batch of normalized images.
    @param normalization: dict, the normalization parameters.
    @param target: str, the target location.
    @param force: bool, overwrite exiting target if True.

    Return:
    numpy.ndarray, the well-origanized batch of data.
    """
    n, c, h, w = batch.shape 
    mean = torch.tensor(normalization['mean'])[None, :, None, None]
    std = torch.tensor(normalization['std'])[None, :, None, None]
    batch = ((batch * std + mean) * 255.0).numpy().astype('uint8').transpose((0,2,3,1)).reshape((n*h, w, c))
    if target is not None:
        if os.path.isfile(target):
            if not not force: pylab.imsave(target, batch)
        else:
            pylab.imsave(target, batch)
    return batch

def plot_episode(batch, normalization, info, target=None, force=False):
    """Plot a batch of images which are supposed to be organized as an episode.

    Args:
    @param batch: torch.FloatTensor, a batch of normalized images.
    @param normalization: dict, the normalization parameters.
    @param info: FewShotEpisodeMeta, the episode definition.
    @param target: str, the target location.
    @param force: bool, overwrite exiting target if True.

    Return:
    numpy.ndarray, the well-origanized batch of data.
    """
    n_c, n_s, n_q, n_u = info.n_class, info.n_support, info.n_query, info.n_unlabel
    n_i = n_s + n_q + n_u
    n_e = n_c * n_i
    N, C, H, W = batch.shape
    batch_size = int(N / n_e)
    batch = batch.view(batch_size, n_c, n_i, C, H, W)
    batch = batch.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, C, n_c * H, n_i * W)
    return plot_batch(batch, normalization, target, force)


def pil_loader(path):
    """A loader for loading PIL images.

    Args:
    @param path: str, the image path.

    Return:
    PIL.Image, the target image content.
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except IOError as e:
        return None


def accimage_loader(path):
    """A loader for loading acc images.

    Args:
    @param path: str, the image path.

    Return:
    accimage.Image (or PIL.Image if cannot load as acc image), the target image content.
    """
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def raw_loader(path, chunk_size=-1):
    """A loader for loading raw file content.

    Args:
    @param path: str, the file path.
    @param chunk_size: int, the chunk size, default -1.

    Return:
    the content of the file.
    """
    try:
        with open(path, 'rb') as f:
            if chunk_size is None or chunk_size < 1:
                # read all data once
                content = f.read()
            else:
                # read data chunk by chunk
                content = b''
                while True:
                    tmp = f.read(4096)
                    if not tmp: break
                    content = content + tmp
            return content
    except IOError as e:
        return None

def default_loader(path):
    """A default image loader.

    Args:
    @param path: str, the image path.

    Return:
    the target image content.
    """
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def dump_imagefolder_dataset_as_lmdb(dataset, lmdb_target, 
    key_generator = None,
    num_workers = 4,
    map_size = 1073741824, # default 1GB
    write_interval = -1,
    verbose = False,
    **kwargs
    ):
    """Dump an ImageFolder dataset into a LMDB dataset.

    Args:
    @param dataset: ImageFolder, the source dataset, please make sure the dataset generates PIL image.
    @param lmdb_target: str, the lmdb dataset file.
    @param key_generator: Callable[ImageFolder, int], function for preparing keys for each record in lmdb dataset. 
    If None, use relative path. A key_generator is a callable method with takes the dataset and the index of the 
    current item as input and outputs a tuple (key, path).
    @param num_workers: int, number of workers, default 4.
    @param map_size: int, the maximum size lmdb dataset can grow, default 1073741824 bytes (i.e. 1GB).
    @param write_interval: int, number of records cached before making a commit, default -1 (caching all records and making just one commit).
    @param verbose: bool, If True, show progress bar, default False.
    
    Other named parameters for LMDB open() method is also supported.
    """
    def default_imagefolder_dataset_key_generator(dataset, i):
        """Generate a key (str) for the i-th item in dataset.

        Args:
        @param dataset: ImageFolder, an ImageFolder object.
        @param i: int, the 0-based index value.

        Return
        str, the key
        """
        root = dataset.root
        path = dataset.samples[i][0]
        key = path.replace(root, '')
        if key.startswith('/'): key = key[1:]
        return key, path
    #
    assert isinstance(dataset, torchvision.datasets.ImageFolder)
    dloader = torch.utils.data.DataLoader(dataset, 
        batch_size=1, 
        shuffle=False,
        drop_last=False,
        num_workers=max(0,num_workers),
        collate_fn=identity
        )
    # TODO: check the validity of lmdb target
    __keys = []
    __files = []
    if key_generator is None: key_generator = default_imagefolder_dataset_key_generator
    db = lmdb.open(lmdb_target, subdir=False, map_size=map_size, readonly=False, **kwargs)
    txn = db.begin(write=True)
    for i, item in enumerate(tqdm(dloader) if not not verbose else dloader):
        image, label = item[0]
        key, path = key_generator(dataset, i)
        key = key.encode('ascii')
        __keys.append(key)
        __files.append(path)
        if not isinstance(image, bytes):
            assert isinstance(image, Image)
            image = image.tobytes()
        # TODO: check key < Environment.max_key_size()
        txn.put(key, image)
        if (write_interval > 0) and ((i+1) % write_interval == 0):
            txn.commit()
            # another begin after commit
            txn = db.begin(write=True)
    txn.put(b'__keys__', pickle.dumps(__keys))
    txn.put(b'__files__', pickle.dumps(__files))
    txn.put(b'__len__', int(len(__keys)).to_bytes(4, 'big')) # 4 bytes to hold the number of items (which should be less than 4294967296, i.e #ITEM < 4294967296). Use more bytes if #ITEM >= 4294967296.
    txn.commit()
    db.close()