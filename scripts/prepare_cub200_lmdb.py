import os
import tarfile
import torch
import lmdb
import pickle
from PIL import Image
from tqdm import tqdm

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

def dump_cub200_dataset_as_lmdb(dataset, lmdb_target, 
    num_workers = 4,
    map_size = 1073741824, # default 1GB
    write_interval = -1,
    verbose = False,
    **kwargs
    ):
    """Dump CUB_200_2011 dataset into a LMDB dataset.

    Args:
    @param dataset: CUB200ImageFolder or CUB200TARFile, the source dataset.
    @param lmdb_target: str, the lmdb dataset file.
    @param num_workers: int, number of workers, default 4.
    @param map_size: int, the maximum size lmdb dataset can grow, default 1073741824 bytes (i.e. 1GB).
    @param write_interval: int, number of records cached before making a commit, default -1 (caching all records and making just one commit).
    @param verbose: bool, If True, show progress bar, default False.
    
    Other named parameters for LMDB open() method is also supported.
    """
    #
    assert isinstance(dataset, (CUB200ImageFolder, CUB200TARFile))
    is_tarfile = isinstance(dataset, CUB200TARFile)
    dloader = torch.utils.data.DataLoader(dataset, 
        batch_size=1, 
        shuffle=False,
        drop_last=False,
        num_workers=max(0,num_workers),
        collate_fn=lambda x: x
        )
    # TODO: check the validity of lmdb target
    __keys = []
    __files = []
    db = lmdb.open(lmdb_target, subdir=False, map_size=map_size, readonly=False, **kwargs)
    txn = db.begin(write=True)
    # meta and annotations
    txn.put(b'__meta__/images.txt', dataset.get_annotation('images.txt'))
    txn.put(b'__meta__/classes.txt', dataset.get_annotation('classes.txt'))
    txn.put(b'__meta__/parts/parts.txt', dataset.get_annotation('parts/parts.txt'))
    txn.put(b'__meta__/attributes/attributes.txt', dataset.get_annotation('/attributes.txt' if is_tarfile else '../attributes.txt'))
    txn.put(b'__meta__/attributes/certainties.txt', dataset.get_annotation('attributes/certainties.txt'))
    txn.put(b'__meta__/train_test_split.txt', dataset.get_annotation('train_test_split.txt'))
    txn.put(b'__annotation__/image_class_labels.txt', dataset.get_annotation('image_class_labels.txt'))
    txn.put(b'__annotation__/bounding_boxes.txt', dataset.get_annotation('bounding_boxes.txt'))
    txn.put(b'__annotation__/parts/part_locs.txt', dataset.get_annotation('parts/part_locs.txt'))
    txn.put(b'__annotation__/parts/part_click_locs.txt', dataset.get_annotation('parts/part_click_locs.txt'))
    txn.put(b'__annotation__/attributes/image_attribute_labels.txt', dataset.get_annotation('attributes/image_attribute_labels.txt'))
    txn.put(b'__annotation__/attributes/class_attribute_labels_continuous.txt', dataset.get_annotation('attributes/class_attribute_labels_continuous.txt'))
    #
    for i, item in enumerate(tqdm(dloader) if not not verbose else dloader):
        idx, key, imbytes = item[0]
        assert i == idx
        _key = key.encode('ascii')
        __keys.append(_key)
        __files.append(os.path.join('images', key))
        # if not isinstance(imbytes, bytes):
        #     assert isinstance(imbytes, Image)
        #     imbytes = imbytes.tobytes()
        # TODO: check key < Environment.max_key_size()
        txn.put(_key, imbytes)
        if (write_interval > 0) and ((i+1) % write_interval == 0):
            txn.commit()
            # another begin after commit
            txn = db.begin(write=True)
    txn.put(b'__keys__', pickle.dumps(__keys))
    txn.put(b'__files__', pickle.dumps(__files))
    txn.put(b'__len__', int(len(__keys)).to_bytes(4, 'big')) # 4 bytes to hold the number of items (which should be less than 4294967296, i.e #ITEM < 4294967296). Use more bytes if #ITEM >= 4294967296.
    txn.commit()
    db.close()

class CUB200ImageFolder(torch.utils.data.Dataset):
    """A toy CUB_200_2011 Dataset for loading images from image folders, e.g. /path/to/CUB_200_2011/.
    """
    _meta_images = 'images.txt'
    _meta_train_test_split = 'train_test_split.txt'
    _meta_classes = 'classes.txt'
    _meta_image_class_labels = 'image_class_labels.txt'
    _meta_bounding_boxes = 'bounding_boxes.txt'
    _meta_part_parts = 'parts/parts.txt'
    _meta_part_part_locs = 'parts/part_locs.txt'
    _meta_part_part_click_locs = 'parts/part_click_locs.txt'
    _meta_attr_attributes = '../attributes.txt'
    _meta_attr_certainties = 'attributes/certainties.txt'
    _meta_attr_image_attribute_labels = 'attributes/image_attribute_labels.txt'
    _meta_attr_class_attribute_labels_continuous = 'attributes/class_attribute_labels_continuous.txt'
    def __init__(self, root):
        assert os.path.isdir(root)
        assert os.path.isdir(os.path.join(root, 'images'))
        self.root = root
        with open(os.path.join(root, self._meta_images), 'r') as f:
            self.images = [item.strip().split()[1] for item in f.readlines()]
    
    def get_annotation(self, x):
        with open(os.path.join(self.root, x), 'rb') as f:
            data = f.read()
        return data
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        key = self.images[index]
        imfile = os.path.join(self.root, 'images', key)
        imbytes = raw_loader(imfile)
        return index, key, imbytes

class CUB200TARFile(torch.utils.data.Dataset):
    """A toy CUB_200_2011 Dataset for loading images from tar file, e.g. /path/to/CUB_200_2011.tar.gz
    """
    _meta_root = 'CUB_200_2011'
    _meta_images = 'images.txt'
    _meta_train_test_split = 'train_test_split.txt'
    _meta_classes = 'classes.txt'
    _meta_image_class_labels = 'image_class_labels.txt'
    _meta_bounding_boxes = 'bounding_boxes.txt'
    _meta_part_parts = 'parts/parts.txt'
    _meta_part_part_locs = 'parts/part_locs.txt'
    _meta_part_part_click_locs = 'parts/part_click_locs.txt'
    _meta_attr_attributes = '/attributes.txt'
    _meta_attr_certainties = 'attributes/certainties.txt'
    _meta_attr_image_attribute_labels = 'attributes/image_attribute_labels.txt'
    _meta_attr_class_attribute_labels_continuous = 'attributes/class_attribute_labels_continuous.txt'
    def __init__(self, ftar, mode):
        assert os.path.isfile(ftar)
        self.tarfile = ftar
        self.tar = tarfile.open(ftar, mode)
        #
        f = self.tar.extractfile(os.path.abspath(os.path.join('/', self._meta_root, self._meta_images))[1:])
        self.images = [item.decode('ascii').strip().split()[1] for item in f.readlines()]
        del f
        
    def get_annotation(self, x):
        f = self.tar.extractfile(os.path.abspath(os.path.join('/', self._meta_root, x))[1:])
        data = f.read()
        return data

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        key = self.images[index]
        imfile = os.path.abspath(os.path.join('/', self._meta_root, 'images', key))[1:]
        imbytes = self.tar.extractfile(imfile).read()
        return index, key, imbytes


if __name__ == '__main__':
    import os
    import sys
    import argparse
    parser = argparse.ArgumentParser("A script for converting CUB_200_2011 dataset into LMDB database.")
    parser.add_argument("-f", "--file", type=str, help="the original CUB_200_2011.tar.gz, e.g. /path/to/CUB_200_2011.tar.gz", required=True)
    # parser.add_argument("-f", "--folder", type=str, help="the base directory of CUB_200_2011, e.g. /path/to/CUB_200_2011/", required=True)
    parser.add_argument("-o", '--out', type=str, help="the destination file which holds the output LMDB database, e.g. /path/to/somewhere/CUB_200_2011.lmdb", required=True)
    parser.add_argument('-j', '--jobs', type=int, default=4, help="the number of loading workers, default 4")
    parser.add_argument('-S', '--size', type=int, default=1073741824, help="the expected size of LMDB database in bytes, default 1073741824 (1GB). Set a huge number if you are not sure about that.")
    args = parser.parse_args()

    # issue: cannot work in multi-thread mode 
    assert args.jobs < 2
    # dataset = CUB200ImageFolder(args.folder)
    dataset = CUB200TARFile(args.file, 'r:gz') # must in 'tar.gz' format
    dump_cub200_dataset_as_lmdb(dataset, args.out, num_workers=args.jobs, map_size=args.size, write_interval=1024, verbose=True)

