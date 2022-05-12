# Copyright (c) 2021 Li Auto Company. All rights reserved.
import math
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from datasets.database.sample_database import SampleDataset
from datasets.database.utils.plot_new_sample import show_sample
from tqdm import tqdm
from yamlinclude import YamlIncludeConstructor


class TimeProcesser():

    def __init__(self, config, dataset) -> None:
        self.config = config
        self.dataset = dataset

    def get_future_obstacle(self, start_frame_idx, scene_id, start_seq_num, max_size) -> dict:
        frame_idx = start_frame_idx
        obstacle_traj = {}
        # obstacle_seq_num = {}
        total_frame_size = 0
        while True:
            # check end of database
            if frame_idx > len(self.dataset) - 1:
                break

            frame = self.dataset[frame_idx]

            # check enter another scene
            if frame.scene_id != scene_id:
                break

            # collect future frame
            for obstacle in frame.obstacle_features:
                if obstacle.id in obstacle_traj:
                    obstacle_traj[obstacle.id].append(obstacle.obs_vec[-1])
                    # print(obstacle.id, obstacle_seq_num[obstacle.id], "->", obstacle.obs_vec[-1].seq_num, " vs ",
                    #      frame.seq_num)
                    # assert obstacle.obs_vec[-1].seq_num == obstacle_seq_num[
                    #     obstacle.id][-1] + 1, "last obs_vec seq_num error!"
                    # obstacle_seq_num[obstacle.id].append(obstacle.obs_vec[-1].seq_num)
                else:
                    obstacle_traj[obstacle.id] = [obstacle.obs_vec[-1]]
                    # obstacle_seq_num[obstacle.id] = [obstacle.obs_vec[-1].seq_num]

            total_frame_size = (frame_idx - start_frame_idx) + 1
            if total_frame_size == max_size:
                break

            frame_idx += 1
        return obstacle_traj, total_frame_size

    def process(self, data_idx, show=False):
        # for i in tqdm(range(len(self.dataset))):
        frame = self.dataset[data_idx]
        # frame_name = str(frame.scene_id) + "_" + str(frame.seq_num)  # + "_" + str(agent_id)
        # print(frame_name)

        start_seq_num = self.config['scene_sample_start_seq_num']
        if frame.seq_num < start_seq_num:
            return None

        # select frame every N step from start_seq_num (N=skip_num)
        if (frame.seq_num - start_seq_num) % self.config['scene_sample_interval'] != 0:
            return None

        # handle history traj: skip frame which has too few history traj
        current_obstacle_id_to_idx = {obstacle.id: idx for idx, obstacle in enumerate(frame.obstacle_features)}
        agent = frame.obstacle_features[current_obstacle_id_to_idx[frame.agent_id]]
        agetn_history_size = len(agent.obs_vec)
        if agetn_history_size < self.config['min_history_size']:
            # print(f"agetn_history_size {agetn_history_size} < {self.config['min_history_size']}")
            return None

        # handle future traj
        if self.config['max_future_size'] > 0:
            # history:[0, 10] future:[11,90]
            future_obstacle, future_frame_size = self.get_future_obstacle(data_idx + 1, frame.scene_id, frame.seq_num,
                                                                          self.config['max_future_size'])

            if future_frame_size < self.config['min_future_size']:
                return None

            # write to proto
            for future_obstacle_id, future_obstacle_list in future_obstacle.items():
                if future_obstacle_id in current_obstacle_id_to_idx:
                    current_obstacle = frame.obstacle_features[current_obstacle_id_to_idx[future_obstacle_id]]
                    assert len(current_obstacle.obs_future_vec) == 0
                    for future_obstacle_vec in future_obstacle_list:
                        future_vec = current_obstacle.obs_future_vec.add()
                        future_vec.CopyFrom(future_obstacle_vec)

        # if len(frame.map_polyline.lanes) == 0 and frame.seq_num != 0:
        #     scene_first_frame_name = f"{frame.scene_id}_{0}"
        #     scene_first_frame = self.dataset.get_sample(scene_first_frame_name)
        #     assert scene_first_frame is not None
        #     # frame.map_polyline.CopyFrom(scene_first_frame.map_polyline)
        #     frame.map_polyline.MergeFrom(scene_first_frame.map_polyline)

        if show:
            # show adc with global map
            agent_id = -1
            use_global_map = True
            figure_title = "adc_with_global_map_" + frame.scene_id + "_" + str(frame.seq_num) + "_" + str(agent_id)
            show_sample(figure_title, agent_id, use_global_map, frame)

            # show obstacle with local map
            for obstacle in frame.obstacle_features:
                agent_id = obstacle.id
                use_global_map = False
                figure_title = "obstacle_with_local_map_" + frame.scene_id + "_" + str(
                    frame.seq_num) + "_" + str(agent_id)
                show_sample(figure_title, agent_id, use_global_map, frame)
        return frame


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

    time_processer = TimeProcesser(config, dataset)
    for i in tqdm(range(len(dataset))):
        time_process_frame = time_processer.process(i, True)
