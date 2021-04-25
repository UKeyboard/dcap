import torch
import torchvision
import lmdb
import pickle
from PIL import Image
from tqdm import tqdm

identity = lambda x: x

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
        print('Fail to read as PIL image: %s'% path)
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
        print('Fail to read a ACC image: %s' % path)
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
        print('Fail to read file: %s' % path)
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


if __name__ == '__main__':
    import os
    import sys
    from torchvision.datasets import ImageFolder
    import argparse
    parser = argparse.ArgumentParser("A script for converting ImageFolder dataset into LMDB database.")
    parser.add_argument("-f", "--folder", type=str, help="the base directory of ImageFolder, e.g. /path/to/imagenet/train", required=True)
    parser.add_argument("-o", '--out', type=str, help="the destination file which holds the output LMDB database, e.g. /path/to/somewhere/ILSVRC2012_img_train.lmdb", required=True)
    parser.add_argument('-j', '--jobs', type=int, default=4, help="the number of ImageFolder loading workers, default 4")
    parser.add_argument('-S', '--size', type=int, default=1073741824, help="the expected size of LMDB database in bytes, default 1073741824 (1GB). Set a huge number if you are not sure about that.")
    args = parser.parse_args()

    #
    dataset = ImageFolder(args.folder, loader=raw_loader)
    dump_imagefolder_dataset_as_lmdb(dataset, args.out, num_workers=args.jobs, map_size=args.size, write_interval=1024, verbose=True)