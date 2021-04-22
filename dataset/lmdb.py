import os
import lmdb
import io
import pickle
from PIL import Image

__all__ = ['VisionLMDBDatabase']

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