# Copyright (c) 2021 Li Auto Company. All rights reserved.

import multiprocessing
import os
import pickle
import shutil
import zlib
from pathlib import Path
from typing import List

import tqdm
from datasets.database.sample_database import PickleDatabase, SampleDatabase
from numpy import ufunc
from prediction_offline_feature_pb2 import Metadata
from torch.utils.data import Dataset


class PredDataset(Dataset):

    def __init__(self, database_folder, database_type='lmdb', sample_list_file=None, transform=None):
        self.db = SampleDatabase.create(database_folder, database_type)
        with open(sample_list_file, 'r') as fin:
            lines = fin.readlines()
        self.sample_name_list = [line.strip() for line in lines]
        self.database_folder = database_folder
        self.database_type = database_type
        self.sample_list_file = sample_list_file
        self.transform = transform

    def __getitem__(self, idx):
        sample_name = self.sample_name_list[idx]
        sample = self.db.get(sample_name)
        if self.transform is not None:
            sample = self.transform(sample_name, sample)
        return sample

    def __len__(self):
        return len(self.sample_name_list)


class PredDatasetWithCache(PredDataset):

    def __init__(self,
                 database_folder,
                 sample_list_file=None,
                 transform=None,
                 cache_root=None,
                 cache_worker_num=-1,
                 compress_lvl=6,
                 using_cache=False,
                 cache_transform=None):
        super().__init__(database_folder, 'lmdb', sample_list_file, transform)
        self.cache_root = cache_root
        self.compress_lvl = compress_lvl
        self.cache_worker_num = cache_worker_num
        self.using_cache = using_cache
        self.cache_tmp_root = '/dev/shm/prediction_pytorch_tmp'
        self.cache_transform = cache_transform

        if self.using_cache:
            self.cache_db = PickleDatabase(cache_root, write=False, map_size=64 * 1024 * 1024 * 1024)

    def __getitem__(self, idx):
        if not self.using_cache:
            return super().__getitem__(idx)
        else:
            if self.cache_transform is not None:
                return self.cache_transform(self.cache_db.get(self.sample_name_list[idx]))
            else:
                return self.cache_db.get(self.sample_name_list[idx])

    def __len__(self):
        return super().__len__()

    def cache(self):
        print('Caching Begin !!!')

        # check if cache tmp already exists, if true, delete it
        if os.path.exists(self.cache_tmp_root):
            shutil.rmtree(self.cache_tmp_root)

        # create new database to store cache
        cache_database = PickleDatabase(self.cache_tmp_root, write=True, map_size=80 * 1024 * 1024 * 1024)

        proc_num = self.cache_worker_num
        if proc_num == -1:
            proc_num = multiprocessing.cpu_count() - 2
        split_size = int(self.__len__() / proc_num)

        manager = multiprocessing.Manager()
        queue = manager.Queue(maxsize=4 * proc_num)

        caching_process = []
        for i in range(proc_num):
            split_list = self.sample_name_list[i * split_size:(i + 1) * split_size]
            if i == proc_num - 1:
                split_list = self.sample_name_list[i * split_size:]

            proc = multiprocessing.Process(target=self.cache_list, args=(split_list, queue))
            proc.start()
            caching_process.append(proc)

        for _ in tqdm(range(self.__len__())):
            data = queue.get()
            cache_database.put_raw(data[0], data[1])

        for proc in caching_process:
            proc.join()

        # check if cache folder already exists, if exists, delete it and make a new one
        if os.path.exists(self.cache_root):
            shutil.rmtree(self.cache_root)

        # move cache tmp to cache folder
        print(f'Moving cache from {self.cache_tmp_root} to {self.cache_root}')
        shutil.move(self.cache_tmp_root, self.cache_root)

        print('Caching Done !!!')

    def cache_list(self, sample_list, queue):
        for sample_name in sample_list:
            sample = self.db.get(sample_name)
            if self.transform is not None:
                sample = self.transform(sample_name, sample)

            sample_bytes = pickle.dumps(sample)

            compressor = zlib.compressobj(self.compress_lvl, wbits=16 + zlib.MAX_WBITS)
            sample_bytes_compressed = compressor.compress(sample_bytes)
            sample_bytes_compressed += compressor.flush()

            queue.put((sample_name.encode('utf-8'), sample_bytes_compressed))


class PickleDataset(Dataset):

    def __init__(self, database_folders: Path, transform=None):
        self._sample_accum_num: List[int] = []
        self._db_file_name: List[Path] = []
        for db_folder in database_folders.iterdir():
            db = PickleDatabase(str(db_folder), write=False)
            sample_length: int = len(db.meta_data().sample_name_list)
            accum_length: int = self._sample_accum_num[-1] + sample_length if len(
                self._sample_accum_num) > 0 else sample_length

            self._sample_accum_num.append(accum_length)
            self._db_file_name.append(db_folder)

        self.transform = transform

    def __getitem__(self, idx):
        # Find the first database index that is less than the accumulate sample number
        db_idx: int = next(i for i, v in enumerate(self._sample_accum_num) if v > idx)
        sample_idx: int = idx - self._sample_accum_num[db_idx - 1] if db_idx > 0 else idx

        db = PickleDatabase(str(self._db_file_name[db_idx]), write=False)
        sample = db.get(db.meta_data().sample_name_list[sample_idx])
        sample = self.transform(sample) if self.transform is not None else sample

        # TODO(xlm) the following is temporary bugfix to avoid reprocess dataset
        #  if (not hasattr(sample, 'gt_traj')):
        #      B, L, F = 1, 0, 2
        #  else:
        #      B, L, F = sample.gt_traj.shape
        #  target_num = sample.actor_future_limit
        #  if L != target_num:
        #      # padding gt_traj to (1, target_num, 2)
        #      padding = torch.zeros((B, target_num - L, F), dtype=torch.float32)
        #      # [B, L, F] + [B, target_num-L, F] -> [B,target_num, F]
        #      sample.gt_traj = torch.cat([sample.gt_traj, padding], dim=-2)
        #      assert sample.gt_traj.shape == (1, target_num, 2)
        #
        #  if (not hasattr(sample, 'gt_traj_mask')) or sample.gt_traj_mask.shape != (1, target_num):
        #      padding_mask1 = torch.ones((B, L), dtype=torch.int32)
        #      padding_mask0 = torch.zeros((B, target_num - L), dtype=torch.int32)
        #      # [B, L] + [B, target_num-L] -> [B,target_num]
        #      sample.gt_traj_mask = torch.cat([padding_mask1, padding_mask0], dim=-1)
        #      assert sample.gt_traj_mask.shape == (1, target_num)
        #
        #  if (not hasattr(sample, 'gt_target_point')):
        #      B, L, F = 1, 0, 2
        #  else:
        #      B, L, F = sample.gt_target_point.shape
        #  target_num = 3
        #  if L != target_num:
        #      # [B, L]
        #      padding_ones_2d = torch.ones((B, L), dtype=torch.int32)
        #      # [B, target_num - L]
        #      padding_zeros_2d = torch.zeros((B, target_num - L), dtype=torch.int32)
        #      # [B, L, F]
        #      # padding_ones_3d = torch.ones((B, L, F), dtype=torch.int32)#.type_as(sample.actor_vector_mask)
        #      # [B, target_num - L, F]
        #      padding_zeros_3d = torch.zeros((B, target_num - L, F), dtype=torch.int32)
        #
        #      # [B, L] + [B, target_num - L] -> [B, target_num]
        #      sample.gt_target_point_mask = torch.cat([padding_ones_2d, padding_zeros_2d], dim=-1)
        #      assert sample.gt_target_point_mask.shape == (1, 3)
        #
        #      # [B, L, F] + [B, target_num - L, F] -> [B, target_num, F]
        #      sample.gt_target_point = torch.cat([sample.gt_target_point, padding_zeros_3d], dim=-2)
        #      assert sample.gt_target_point.shape == (1, 3, 2)
        #
        #      # [B, L] + [B, target_num - L] -> [B, target_num]
        #      sample.gt_target_grid_idx = torch.cat([sample.gt_target_grid_idx, padding_zeros_2d], dim=-1)
        #      assert sample.gt_target_grid_idx.shape == (1, 3)
        #
        #      # [B, L] + [B, target_num - L] -> [B, target_num]
        #      sample.gt_target_grid_over_border = \
        #       torch.cat([sample.gt_target_grid_over_border, padding_zeros_2d], dim=-1)
        #      assert sample.gt_target_grid_over_border.shape == (1, 3)
        #
        #      # [B, L, F] + [B, target_num - L, F] -> [B, target_num, F]
        #      sample.gt_target_grid_center = torch.cat([sample.gt_target_grid_center, padding_zeros_3d], dim=-2)
        #      assert sample.gt_target_grid_center.shape == (1, 3, 2)
        #
        #      # TODO(xlm) in case of over_grid_border the offset is wrong
        #      # [B, L, F] + [B, target_num - L, F] -> [B, target_num, F]
        #      sample.gt_target_grid_offset = torch.cat([sample.gt_target_grid_offset, padding_zeros_3d], dim=-2)
        #      assert sample.gt_target_grid_offset.shape == (1, 3, 2)

        # if hasattr(sample, 'gt_traj'):
        #     del sample.gt_traj
        # if hasattr(sample, 'gt_traj_mask'):
        #     del sample.gt_traj_mask
        # if hasattr(sample, 'gt_target_point'):
        #     del sample.gt_target_point
        # if hasattr(sample, 'gt_target_point_mask'):
        #     del sample.gt_target_point_mask
        # if hasattr(sample, 'gt_target_grid_idx'):
        #     del sample.gt_target_grid_idx
        # if hasattr(sample, 'gt_target_grid_over_border'):
        #     del sample.gt_target_grid_over_border
        # if hasattr(sample, 'gt_target_grid_center'):
        #     del sample.gt_target_grid_center
        # if hasattr(sample, 'gt_target_grid_offset'):
        #     del sample.gt_target_grid_offset
        # if hasattr(sample, 'candi_target_points'):
        #     del sample.candi_target_points

        return sample

    def __len__(self):
        return self._sample_accum_num[-1]


class InMemoryPickleDataset(Dataset):

    def __init__(self, database_folder, sample_list_file):
        self._db = PickleDatabase(database_folder)
        with open(sample_list_file, 'r') as fin:
            lines = fin.readlines()
        self._sample_name_list = [line.strip() for line in lines]

        print('Loading samples into memory, please wait')
        self._samples = []
        for sample_name in tqdm(self._sample_name_list):
            sample = self._db.get(sample_name)
            self._samples.append(sample)

    def __getitem__(self, idx):
        return self._samples[idx]

    def __len__(self):
        return len(self._sample_name_list)


if __name__ == '__main__':
    dataset = PredDatasetWithCache("/database/argoverse/val",
                                   "/database/argoverse/val_full_list.txt",
                                   transform=None,
                                   cache_root="/database/argoverse/transformed/intention_transform3/val",
                                   using_cache=True)

    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        if data.gt_pix >= 288 * 288:
            print(data.raw_sample_name, data.gt, data.gt_pix)
