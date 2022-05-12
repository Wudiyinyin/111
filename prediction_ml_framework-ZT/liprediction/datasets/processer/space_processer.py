# Copyright (c) 2021 Li Auto Company. All rights reserved.
import copy
import math
import multiprocessing as mp
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from core.utils.utils import np_vector_norm
from datasets.database.sample_database import SampleDataset
from datasets.database.utils.plot_new_sample import show_sample
from tqdm import tqdm
from yamlinclude import YamlIncludeConstructor


class SpaceProcesser():

    def __init__(self, config) -> None:
        self.config = config

    def process_one(self, frame, obstacle_id):
        # set obstacle id as agent id
        frame.agent_id = obstacle_id
        return frame

    def process(self, frame, show=False):
        if frame is None:
            return []

        samples = []
        for obstacle in frame.obstacle_features:
            if self.config['select_agent_strategy'] == "predict_id":
                if obstacle.id not in frame.predict_id:
                    # print(f"{obstacle.id} not in frame.predict_id")
                    continue
            else:
                assert self.config['select_agent_strategy'] == "all", \
                    f"UNKNOW select_agent_strategy {self.config['select_agent_strategy']}"

            if len(obstacle.obs_vec) < self.config['min_history_size']:
                # print(f"skip {frame.scene_id}_{frame.seq_num}_{obstacle.id} by \
                #    {len(obstacle.obs_vec)} < min_history_size({self.config['min_history_size']})")
                continue

            if len(obstacle.obs_future_vec) < self.config['min_future_size']:
                # print(f"skip {frame.scene_id}_{frame.seq_num}_{obstacle.id} by \
                #    {len(obstacle.obs_future_vec)} < \
                #    min_future_size({self.config['min_future_size']})")
                continue

            if self.config['min_move_distance'] > 0:
                total_move_distance = 0
                for frame in obstacle.obs_vec:
                    start_x = frame.vec.points[0].x
                    start_y = frame.vec.points[0].y
                    end_x = frame.vec.points[1].x
                    end_y = frame.vec.points[1].y
                    total_move_distance += np_vector_norm(np.array([end_x - start_x, end_y - start_y],
                                                                   dtype=np.float32))
                for frame in obstacle.obs_future_vec:
                    start_x = frame.vec.points[0].x
                    start_y = frame.vec.points[0].y
                    end_x = frame.vec.points[1].x
                    end_y = frame.vec.points[1].y
                    total_move_distance += np_vector_norm(np.array([end_x - start_x, end_y - start_y],
                                                                   dtype=np.float32))
                if total_move_distance < self.config['min_move_distance']:
                    # print(f"skip {frame.scene_id}_{frame.seq_num}_{obstacle.id} by \
                    #     {total_move_distance} < {self.config['min_move_distance']} ")
                    continue

            copy_frame = copy.deepcopy(frame)
            sample = self.process_one(copy_frame, obstacle.id)
            assert sample is not None
            samples.append(sample)

            if show:
                agent_id = sample.agent_id
                use_global_map = False
                figure_title = "agent_with_local_map_" + sample.scene_id + "_" + str(
                    sample.seq_num) + "_" + str(agent_id)
                show_sample(figure_title, agent_id, use_global_map, frame)

        if self.config['select_agent_strategy'] == "predict_id" and self.config['check_predict_id']:
            actual_pred_ids = []
            for sample in samples:
                actual_pred_ids.append(sample.agent_id)

            for need_pred_id in frame.predict_id:
                assert need_pred_id in actual_pred_ids, \
                    f"{frame.scene_id}_{frame.seq_num} need_pred_id {need_pred_id} is not predict!"

        # print(f'{frame.scene_id}-{frame.seq_num} generate {len(samples)} samples')
        return samples


if __name__ == '__main__':
    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print(f'{config_file} not exists!!')
    config_dir = os.path.dirname(config_file)

    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=config_dir)
    with open(config_file, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = config['processer']

    sample_list_file = None
    if 'sample_list_file' in config and len(config['sample_list_file']) > 0:
        sample_list_file = config['sample_list_file']

    dataset = SampleDataset(config['sample_database_folder'], 'lmdb', sample_list_file)
    print(f'Dataset sample number: {len(dataset)}')

    space_processer = SpaceProcesser(config)
    for i in tqdm(range(len(dataset))):
        space_process_samples = space_processer.process(dataset[i], True)
        print(f'{i}-{dataset[i].scene_id}-{dataset[i].seq_num} generate {len(space_process_samples)} samples')
