# Copyright (c) 2021 Li Auto Company. All rights reserved.

import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path

import yaml
from datasets.database.sample_database import PickleDatabase, SampleDataset
from datasets.processer.feature_processer import (FeatureProcesser, plot_transformed_data)
from datasets.processer.space_processer import SpaceProcesser
from datasets.processer.time_processer import TimeProcesser
from joblib import Parallel, delayed
from tqdm import tqdm
from yamlinclude import YamlIncludeConstructor

# from prediction_offline_feature_pb2 import Metadata


class Processer():

    def __init__(self, config):
        self.config = config
        self.phase = config['phase']
        self.worker_num = config['cache_worker_num']
        self.use_multi_thread = self.worker_num != -2
        if self.worker_num == -1:
            self.worker_num = multiprocessing.cpu_count() - 2

    def run(self, input_database_folder, output_database_folder):
        # input database
        if not os.path.exists(input_database_folder):
            print(f"Input database folder [{input_database_folder}] not exists!!!")
            return
        self.input_dataset = SampleDataset(input_database_folder, 'lmdb')
        # print(f'Dataset sample number: {len(self.input_dataset)}')

        # output database
        self.output_database_folder = output_database_folder
        output_database_root = os.path.dirname(output_database_folder)
        if not os.path.exists(output_database_root):
            print(f"Output database root [{output_database_root}] not exists, Creating ...")
            os.makedirs(output_database_root)

        # processer
        self.time_processer = TimeProcesser(self.config[self.phase], self.input_dataset)
        self.space_processer = SpaceProcesser(self.config[self.phase])
        self.feature_processer = FeatureProcesser(self.config)

        start_time = time.time()
        if self.use_multi_thread:
            print(f"\nRun multi_thread [{self.phase}] with worker_num {self.worker_num} \
                    sample number: {len(self.input_dataset)} \
                    {input_database_folder} --> {output_database_folder}")
            self.run_multi_thread()
        else:
            print(f"\nRun single_thread [{self.phase}] with worker_num {self.worker_num} \
                    sample number: {len(self.input_dataset)} \
                    {input_database_folder} --> {output_database_folder}")
            self.run_single_thread()
        end_time = time.time()

        print('Done in {:.4f} seconds'.format(end_time - start_time))

    def run_single_thread(self):
        save_folder = self.output_database_folder
        saver = SampleSaver(save_folder)
        for i in tqdm(range(len(self.input_dataset))):
            self.cache_one_scene_map(i)
            samples = self.process_one_frame(i)
            for sample in samples:
                saver.save_sample(sample)
        del saver

    def run_multi_thread(self):
        self.cache_all_scene_map()

        proc_num = self.worker_num
        total_size = len(self.input_dataset)
        split_size = int(total_size / proc_num)
        index_list = range(total_size)

        manager = multiprocessing.Manager()
        queue = manager.Queue()  # Queue(maxsize=4 * proc_num)
        caching_process = []
        for i in range(proc_num):
            split_list = index_list[i * split_size:(i + 1) * split_size]
            if i == proc_num - 1:
                split_list = index_list[i * split_size:]

            proc = multiprocessing.Process(target=self.process_many_frame, args=(split_list, queue))
            proc.start()
            caching_process.append(proc)

        for _ in tqdm(range(total_size)):
            queue.get()

        for proc in caching_process:
            proc.join()

    def process_many_frame(self, indexs, queue=None):
        save_folder = self.output_database_folder + "_" + str(indexs[0]) + "-" + str(indexs[-1])
        saver = SampleSaver(save_folder)
        for index in indexs:
            samples = self.process_one_frame(index)
            for sample in samples:
                saver.save_sample(sample)

            # Queue pass Tensor will failed, so we just put(None) to mark finish
            queue.put(None)

        del saver

    def process_one_frame(self, i):
        # print(f"process_one_frame {i} sample_name {self.input_dataset.get_sample_name(i)}")

        sample_time_process = self.time_processer.process(i)
        if sample_time_process is None:
            return []
        # print(f"time_processer generate {self.gen_sample_name(sample_time_process)}")

        samples_space_process = self.space_processer.process(sample_time_process)
        if len(samples_space_process) < 1:
            return []
        # print(f"space_processer process {self.gen_sample_name(sample_time_process)} \
        #             with {len(samples_space_process)} samples")

        sample_results = []
        for sample_space_process in samples_space_process:
            # print(f"feature_processer process {self.gen_sample_name(sample_space_process)}")
            sample_transformed = self.feature_processer.process(sample_space_process)
            if sample_transformed is None:
                continue
            # print(f"feature_processer produce {sample_transformed.raw_sample_name}")
            sample_results.append(sample_transformed)

        return sample_results

    def gen_sample_name(self, frame):
        return f"{frame.scene_id}_{frame.seq_num}_{frame.agent_id}"

    def cache_all_scene_map(self):
        # print("Cache All scene map for single PB multi Frame parallel ...")
        # get all scene_map
        scene_id_to_map = {}
        for i in range(len(self.input_dataset)):
            sample_name = self.input_dataset.get_sample_name(i)
            scene_id, scene_seq_num_str = sample_name.split("_")
            if (int(scene_seq_num_str) == 0) and (scene_id not in scene_id_to_map):
                scene_id_to_map[scene_id] = self.input_dataset.get_sample(sample_name).map_polyline
        self.feature_processer.scene_id_to_map = scene_id_to_map

    def cache_one_scene_map(self, i: int):
        sample_name = self.input_dataset.get_sample_name(i)
        scene_id, scene_seq_num_str = sample_name.split("_")
        if int(scene_seq_num_str) == 0:
            # print(f"Cache Current scene map [{sample_name}] for multi PB single Frame parallel ...")
            scene_id_to_map = {}
            scene_id_to_map[scene_id] = self.input_dataset.get_sample(sample_name).map_polyline
            self.feature_processer.scene_id_to_map = scene_id_to_map

    # def handle_result(self, samples):
    #     self.pbar.update(1)

    #     if samples is None or len(samples) < 1:
    #         return

    #     self.total_sample_num += len(samples)
    #     print(f"handle_result {samples[0].raw_sample_name} {len(samples)}/{self.total_sample_num}")
    #     for sample_transformed in samples:
    #         self.output_database.put(sample_transformed.raw_sample_name, sample_transformed)
    #         self.meta_data.sample_name_list.append(sample_transformed.raw_sample_name)
    #         self.output_database.put("meta_data", self.meta_data.SerializeToString())
    #         # print("output_database done!", sample_transformed.raw_sample_name)
    #         if not self.use_multi_thread:
    #             plot_transformed_data(sample_transformed, True, "")


class SampleSaver():

    def __init__(self, output_database_folder):
        self.output_database = PickleDatabase(output_database_folder, write=True, map_size=80 * 1024 * 1024 * 1024)
        # self.meta_data = Metadata()

    def __del__(self):
        # self.output_database.put_raw("meta_data", self.meta_data.SerializeToString())
        del self.output_database

    def save_sample(self, sample_transformed):
        self.output_database.put(sample_transformed.raw_sample_name, sample_transformed)
        # self.meta_data.sample_name_list.append(sample_transformed.raw_sample_name)
        # self.output_database.put_raw("meta_data", self.meta_data.SerializeToString())
        # print("output_database done!", sample_transformed.raw_sample_name)
        # plot_transformed_data(sample_transformed, True, "")


if __name__ == '__main__':
    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print(f'{config_file} not exists!!')
    config_dir = os.path.dirname(config_file)

    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=config_dir)
    with open(config_file, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    input_database_root = config["processer"]['sample_database_folder']
    output_database_root = config["processer"]['cache_database_folder']

    # check if cache tmp already exists, if true, delete it
    if os.path.exists(output_database_root):
        output_database_folder_tmp = output_database_root + "_" + time.strftime("%Y-%m-%d_%H:%M:%S",
                                                                                time.localtime(time.time()))
        # shutil.rmtree(self.cache_tmp_root)
        print(f'Output folder already exists, Moving to {output_database_folder_tmp}')
        shutil.move(output_database_root, output_database_folder_tmp)

    if not os.path.exists(output_database_root):
        print(f"Output database root [{output_database_root}] not exists, Creating ...")
        os.makedirs(output_database_root)

    def process(input_database_path, output_database_root):
        processer = Processer(config["processer"])
        dbname = os.path.basename(input_database_path)
        output_database_path = os.path.join(output_database_root, dbname)
        processer.run(str(input_database_path), output_database_path)

    input_database_root: Path = Path(input_database_root)
    output_database_root: Path = Path(output_database_root)

    # multi process: -1 : use all_cpu - 1, 1 : use single process
    Parallel(n_jobs=config["processer"]['db_process_num'])(delayed(process)(input_database_path, output_database_root)
                                                           for input_database_path in input_database_root.iterdir())
