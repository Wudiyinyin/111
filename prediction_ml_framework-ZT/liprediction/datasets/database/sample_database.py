# Copyright (c) 2021 Li Auto Company. All rights reserved.
import imp
import pickle
import zlib
from abc import ABCMeta, abstractmethod

import lmdb
from prediction_offline_feature_pb2 import FrameFeature as Sample
from prediction_offline_feature_pb2 import Metadata
from torch.utils import data

# from .proto.prediction.prediction_dataset_pb2 import Metadata, Sample


class SampleDataset(object):

    def __init__(self, database_folder, database_type, sample_list_file=None):
        self._db = SampleDatabase.create(database_folder, database_type)
        self._meta_data = self._db.meta_data()
        self._sample_name_list = self._meta_data.sample_name_list
        if sample_list_file is not None:
            with open(sample_list_file, 'r') as fin:
                lines = fin.readlines()
            self._sample_name_list = [line.strip() for line in lines]

    # so can access like dataset[idx]
    def __getitem__(self, idx):
        return self._db.get(self._sample_name_list[idx])

    # len(dataset)
    def __len__(self):
        return len(self._sample_name_list)

    def get_sample_name(self, idx):
        sample_name = self._sample_name_list[idx]
        return sample_name

    def get_sample(self, name):
        if not (name in self._sample_name_list):
            return None
        return self._db.get(name)

    def get_raw_sample(self, idx):
        sample_name = self._sample_name_list[idx]
        return (sample_name, self._db.get_raw(sample_name.encode('utf-8')))


class SampleDatabase(object, metaclass=ABCMeta):

    def __init__(self):
        pass

    @staticmethod
    def create(database_path, database_type='lmdb', write=False):
        if database_type == 'lmdb':
            return SampleDatabaseLmdb(database_path, write)
        elif database_type == 'sqlite3':
            return None
        else:
            return None

    @abstractmethod
    def meta_data(self):
        pass

    @abstractmethod
    def get(self, key):
        pass

    @abstractmethod
    def put(self, key, value):
        pass

    @abstractmethod
    def get_raw(self, key_raw):
        pass

    @abstractmethod
    def put_raw(self, key_raw, value_raw):
        pass


class SampleDatabaseLmdb(SampleDatabase):

    def __init__(self, database_path, write=False):
        self._lmdb_env = lmdb.open(
            database_path,
            map_size=32 * 1024 * 1024 * 1024,  # max 32G
            lock=write,
            readonly=(not write))
        self._database_path = database_path
        self._write = write

        # TODO(xlm) use list to keep order?
        self._sample_name_set = set()
        with self._lmdb_env.begin() as txn:
            meta_data_string = txn.get('meta_data'.encode('utf-8'))
            if meta_data_string is not None:
                meta_data = Metadata()
                meta_data.ParseFromString(meta_data_string)
                self._sample_name_set = set(meta_data.sample_name_list)

    def __del__(self):
        # write back the meta data
        if self._write:
            meta_data = Metadata()
            for sample_name in self._sample_name_set:
                meta_data.sample_name_list.append(sample_name)
            self.put_raw('meta_data', meta_data.SerializeToString())
            # print(f"sample database [{self._database_path}] meta_data saved!")

        # close dataset
        self._lmdb_env.close()
        # print(f"sample database [{self._database_path}] closed!")

    def meta_data(self):
        with self._lmdb_env.begin() as txn:
            meta_data_string = txn.get('meta_data'.encode('utf-8'))
            meta_data = Metadata()
            meta_data.ParseFromString(meta_data_string)
        return meta_data

    def get(self, key: str):
        with self._lmdb_env.begin() as txn:
            decompressor = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
            sample_bytes_compressed = txn.get(key.encode('utf-8'))
            if sample_bytes_compressed is None:
                raise Exception(f'{key} not found in database')
            sample_bytes = decompressor.decompress(sample_bytes_compressed)
            sample_bytes += decompressor.flush()  # sample is proto
            sample = Sample.FromString(sample_bytes)
        return sample

    def put(self, key: str, sample):
        if not self._write:
            return False

        with self._lmdb_env.begin(write=True) as txn:
            compressor = zlib.compressobj(9, wbits=16 + zlib.MAX_WBITS)
            sample_bytes = sample.SerializeToString()  # sample is proto
            sample_bytes_compressed = compressor.compress(sample_bytes)
            sample_bytes_compressed += compressor.flush()
            res = txn.put(key.encode('utf-8'), sample_bytes_compressed)
            self._sample_name_set.add(key)
        return res

    def put_list(self, key_list, sample_list):
        if not self._write:
            return False

        with self._lmdb_env.begin(write=True) as txn:
            for key, sample in zip(key_list, sample_list):
                compressor = zlib.compressobj(9, wbits=16 + zlib.MAX_WBITS)
                sample_bytes = sample.SerializeToString()  # sample is proto
                sample_bytes_compressed = compressor.compress(sample_bytes)
                sample_bytes_compressed += compressor.flush()
                txn.put(key.encode('utf-8'), sample_bytes_compressed)
                self._sample_name_set.add(key)
        return True

    def get_raw(self, key: str):
        with self._lmdb_env.begin(write=False) as txn:
            sample = txn.get(key.encode('utf-8'))
        return sample

    def put_raw(self, key: str, sample: bytes):
        if not self._write:
            return False

        with self._lmdb_env.begin(write=True) as txn:
            # lmdb put must be bytestring
            txn.put(key.encode('utf-8'), sample)
            self._sample_name_set.add(key)
        return True


class SampleDatabaseSqlite(SampleDatabase):

    def __init__(self):
        pass

    def __del__(self):
        pass

    def meta_data(self):
        pass

    def get(self, key):
        pass

    def put(self, key, value):
        pass


class PickleDatabase(object):

    def __init__(self, database_path, write=False, map_size=32 * 1024 * 1024 * 1024):
        self._lmdb_env = lmdb.open(database_path, map_size=map_size, lock=write, readonly=(not write))
        self._write = write
        self._sample_name_set = set()
        self._database_path = database_path

    def __del__(self):
        # write back the meta data
        if self._write:
            meta_data = Metadata()
            for sample_name in self._sample_name_set:
                meta_data.sample_name_list.append(sample_name)
            self.put_raw('meta_data', meta_data.SerializeToString())
            # print(f"pickle database [{self._database_path}] meta_data saved!")

        # close dataset
        self._lmdb_env.close()
        # print(f"pickle database [{self._database_path}] closed!")

    def meta_data(self):
        with self._lmdb_env.begin() as txn:
            meta_data_string = txn.get('meta_data'.encode('utf-8'))
            meta_data = Metadata()
            meta_data.ParseFromString(meta_data_string)
        return meta_data

    def get(self, key: str):
        with self._lmdb_env.begin() as txn:
            decompressor = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
            sample_bytes_compressed = txn.get(key.encode('utf-8'))
            if sample_bytes_compressed is None:
                raise Exception(f'{key} not found in database')
            sample_bytes = decompressor.decompress(sample_bytes_compressed)
            sample_bytes += decompressor.flush()
            sample = pickle.loads(sample_bytes)  # sample is object
        return sample

    def put(self, key: str, sample):
        if not self._write:
            return False

        with self._lmdb_env.begin(write=True) as txn:
            compressor = zlib.compressobj(9, wbits=16 + zlib.MAX_WBITS)
            sample_bytes = pickle.dumps(sample)  # sample is object
            sample_bytes_compressed = compressor.compress(sample_bytes)
            sample_bytes_compressed += compressor.flush()
            res = txn.put(key.encode('utf-8'), sample_bytes_compressed)
            self._sample_name_set.add(key)
        return res

    def put_list(self, key_list, sample_list):
        if not self._write:
            return False

        with self._lmdb_env.begin(write=True) as txn:
            for key, sample in zip(key_list, sample_list):
                compressor = zlib.compressobj(9, wbits=16 + zlib.MAX_WBITS)
                sample_bytes = pickle.dumps(sample)  # sample is object
                sample_bytes_compressed = compressor.compress(sample_bytes)
                sample_bytes_compressed += compressor.flush()
                txn.put(key.encode('utf-8'), sample_bytes_compressed)
                self._sample_name_set.add(key)
        return True

    def get_raw(self, key: str):
        with self._lmdb_env.begin() as txn:
            value = txn.get(key.encode('utf-8'))
        return value

    def put_raw(self, key: str, sample_raw: bytes):
        if not self._write:
            return False

        with self._lmdb_env.begin(write=True) as txn:
            # lmdb put must be bytestring
            txn.put(key.encode('utf-8'), sample_raw)
            self._sample_name_set.add(key)
        return
