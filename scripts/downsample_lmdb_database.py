"""Given a image dataset serialized as LMDB database, downsample
all images in the dataset by resize them so that the short edge equals the specified size.
"""
import os
import sys
import lmdb
import pickle
import torchvision.transforms as ttf
from PIL import Image as PILImage
from tqdm import tqdm
from io import BytesIO

# VirtualLMDBDatabase and VisionLMDBDatabase are defined in dataset/lmdb.py
# We just copy the code and paste it here
class VirtualLMDBDatabase(object):
    """A virtual calss for LMDB database.

    Args:
    @param lmdb: str, the source LMDB database file.
    """
    def __init__(self, lmdb_file):
        assert isinstance(lmdb_file, str)
        assert os.path.isfile(lmdb_file)
        self.db_binary = lmdb_file
        self.db_env = lmdb.open(lmdb_file, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        with self.db_env.begin(write=False) as txn:
            # load meta: __len__
            db_length = txn.get(b'__len__')
            if db_length is None:
                raise ValueError('LMDB database must have __len__ item which indicates number of records in the database.')
            self.db_length = int().from_bytes(db_length, 'big') # ATTENTION! Do save __len__ as bytes in big version when making the database
            # load meta: __key__
            db_keys = txn.get(b'__keys__')
            if db_keys is None:
                raise ValueError('LMDB database must have __keys__ item which holds the keys for retrieving record.')
            self.db_keys = pickle.loads(db_keys)
            # load meta: __files__
            db_files = txn.get(b'__files__')
            if db_files is None:
                raise ValueError('LMDB database must have __files__ item which holds the paths of original data files.')
            self.db_files = pickle.loads(db_files)
        assert self.db_length == len(self.db_keys)
        self.db_iter_idx = -1
    
    def __len__(self):
        return self.db_length
    
    def __repr__(self):
        return "%s (%s)" % (self.__class__.__name__, self.db_binary)

    def __iter__(self):
        return self
    
    def __next__(self):
        self.db_iter_idx += 1
        if self.db_iter_idx >= len(self):
            raise StopIteration
        return self[self.db_iter_idx]
    
    def __getitem__(self, index):
        raise NotImplementedError


class VisionLMDBDatabase(VirtualLMDBDatabase):
    """ A basic LMDB database for images.
    """
    def __getitem__(self, index):
        return self.getitem_by_key(self.db_keys[index])

    def getitem_by_key(self, key):
        env = self.db_env
        with env.begin(write=False) as txn:
            byteflow = txn.get(key)
            if byteflow is None: return None
        return Image.open(io.BytesIO(byteflow)).convert('RGB')

class VisionLMDBDatabaseDownsampler(VisionLMDBDatabase):
    def __init__(self, lmdb_file, size, map_size, write_interval = -1, target_lmdb_file=None):
        assert isinstance(size, int)
        assert size > 0
        db_path, db_file = os.path.dirname(lmdb_file), os.path.basename(lmdb_file)
        assert db_file[-5:] == '.lmdb' # end with '.lmdb'
        new_db_file = target_lmdb_file or os.path.join(db_path, '%s_x%d.lmdb' % (db_file[:-5], size))
        if os.path.isfile(new_db_file):
            raise IOError("File already exists: %s" % new_db_file)
        super().__init__(lmdb_file)
        self.downsampler_transform = ttf.Resize(size)
        self.new_db_binary = new_db_file
        self.new_db_env = lmdb.open(new_db_file, subdir=False, map_size=map_size, readonly=False)
        self.write_interval = write_interval
        self.new_db_txn = self.new_db_env.begin(write=True)
    
    def __getitem__(self, index):
        key = self.db_keys[index]
        data = self.downsampler_transform(self.getitem_by_key(key))
        with BytesIO() as bytesO:
            data.save(bytesO, 'JPEG')
            data = bytesO.getvalue()
        self.new_db_txn.put(key, data)
        if (self.write_interval > 0) and ((index+1) % self.write_interval == 0):
            self.new_db_txn.commit()
            self.new_db_txn = self.new_db_env.begin(write=True)
        if index+1 == len(self):
            self.new_db_txn.put(b'__keys__', pickle.dumps(self.db_keys))
            self.new_db_txn.put(b'__files__', pickle.dumps(self.db_files))
            self.new_db_txn.put(b'__len__', int(self.db_length).to_bytes(4, 'big'))
            self.new_db_txn.commit()
            self.new_db_env.close()
        return None
    
if __name__ == "__main__":
    """
    Example:
    python downsample_lmdb_database.py --file=/path/to/imagenet/ILSVRC2012_img_train.lmdb --size=128
    """
    import argparse
    parser = argparse.ArgumentParser("A script for downsampling images dataset in LMDB format.")
    parser.add_argument("-f", "--file", type=str, help="the lmdb databse, e.g. /path/to/imagenet/ILSVRC2012_img_train.lmdb", required=True)
    parser.add_argument("-o", '--out', type=str, help="the destination file which holds the output LMDB database, e.g. /path/to/somewhere/ILSVRC2012_img_train_x128.lmdb", required=True)
    parser.add_argument('-s', '--size', type=int, help="the target short edge value, e.g. 128", required=True)
    parser.add_argument('-S', '--mapsize', type=int, default=1073741824, help="the expected size of LMDB database in bytes, default 1073741824 (1GB). Set a huge number if you are not sure about that.")
    args = parser.parse_args()

    #
    writer = VisionLMDBDatabaseDownsampler(args.file, args.size, args.mapsize, 1000, args.out)
    for _ in tqdm(writer): pass
    