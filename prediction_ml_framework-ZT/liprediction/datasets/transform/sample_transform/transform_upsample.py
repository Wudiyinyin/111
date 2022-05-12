# Copyright (c) 2021 Li Auto Company. All rights reserved.
import math
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from core.utils.utils import np_vector_norm
from datasets.database.sample_database import SampleDatabase
from datasets.database.utils.plot_sample import plot_vector
from datasets.transform.agent_closure import AgentClosure, AgentClosureBatch
from datasets.transform.lane_graph import LaneGraph
from tqdm import tqdm
from yamlinclude import YamlIncludeConstructor


class IntentionTransform8(object):

    def __init__(self, config):
        self.config = config

    def __call__(self, sample_name, sample):
        # print(f'sample_name:{sample_name}')
        rot_present = 'rot_present' in self.config and self.config['rot_present']

        #################
        # feature vector
        #################
        # agent
        agent_traj = sample.trajectories[0]
        agent_pos = np.array([[state.pos.x, state.pos.y] for state in agent_traj.states[:self.config['obs_horizon']]],
                             dtype=np.float32)
        if rot_present:
            agent_dir = np.array(
                [[math.cos(state.rot), math.sin(state.rot)] for state in agent_traj.states[:self.config['obs_horizon']]
                ],
                dtype=np.float32)

        agent_origin = agent_pos[-1]
        agent_rotm = self.rotmFromVect(agent_pos[0], agent_pos[-1])
        agent_rot = agent_rotm[:, 0]
        agent_pose = np.concatenate((agent_origin, agent_rot))

        agent_vector_start = agent_pos[:-1]
        agent_vector_end = agent_pos[1:]

        if rot_present:
            agent_vector = np.hstack((
                agent_vector_start,
                agent_vector_end,
                agent_dir[1:],
                np.full((len(agent_vector_start), 1), self.config['agent_tag'], dtype=np.float32),
            ))
        else:
            agent_vector = np.hstack((
                agent_vector_start,
                agent_vector_end,
                np.full((len(agent_vector_start), 1), self.config['agent_tag'], dtype=np.float32),
            ))

        agent_vector_mask = np.ones(self.config['obs_horizon'] - 1, dtype=np.int32)

        # obstacle
        obstacle_vector_list = []
        obstacle_vector_mask_list = []
        for obstacle_traj in sample.trajectories[1:]:
            if len(obstacle_traj.states) <= 1:
                continue

            obstacle_pos = np.array([[state.pos.x, state.pos.y] for state in obstacle_traj.states], dtype=np.float32)
            if rot_present:
                obstacle_dir = np.array([[math.cos(state.rot), math.sin(state.rot)] for state in obstacle_traj.states],
                                        dtype=np.float32)

            if (np_vector_norm(obstacle_pos[-1] - agent_origin) >
                    self.config['range']) and (np_vector_norm(obstacle_pos[0] - agent_origin) > self.config['range']):
                continue

            obstacle_vector_start = obstacle_pos[:-1]
            obstacle_vector_end = obstacle_pos[1:]

            if rot_present:
                obstacle_vector = np.hstack((
                    obstacle_vector_start,
                    obstacle_vector_end,
                    obstacle_dir[1:],
                    np.full((len(obstacle_vector_start), 1), self.config['obstacle_tag'], dtype=np.float32),
                ))
            else:
                obstacle_vector = np.hstack((
                    obstacle_vector_start,
                    obstacle_vector_end,
                    np.full((len(obstacle_vector_start), 1), self.config['obstacle_tag'], dtype=np.float32),
                ))
            obstacle_vector_len = len(obstacle_vector)
            obstacle_vector = np.vstack(
                (np.zeros((self.config['obs_horizon'] - 1 - len(obstacle_vector), self.config['vector_size']),
                          dtype=np.float32), obstacle_vector))

            obstacle_vector_mask = np.concatenate((np.zeros(self.config['obs_horizon'] - 1 - obstacle_vector_len,
                                                            dtype=np.int32), np.ones(obstacle_vector_len,
                                                                                     dtype=np.int32)))

            obstacle_vector_list.append(obstacle_vector)
            obstacle_vector_mask_list.append(obstacle_vector_mask)

        actor_limit = self.config['actor_limit']
        actor_vector_list = [agent_vector] + obstacle_vector_list
        # most 32 actors
        if len(actor_vector_list) > actor_limit:
            # print(f'drop actor: {len(actor_vector_list)-actor_limit}')
            actor_vector_list = actor_vector_list[:actor_limit]
        actor_vector = np.stack(actor_vector_list)  # [C, V, F]
        actor_vector = np.concatenate(
            (actor_vector,
             np.zeros((actor_limit - len(actor_vector), self.config['obs_horizon'] - 1, self.config['vector_size']),
                      dtype=np.float32)))

        actor_vector_mask_list = [agent_vector_mask] + obstacle_vector_mask_list
        # most 32 actors
        if len(actor_vector_mask_list) > actor_limit:
            actor_vector_mask_list = actor_vector_mask_list[:actor_limit]
        actor_vector_mask = np.stack(actor_vector_mask_list)  # [C, V]
        actor_vector_mask = np.concatenate((actor_vector_mask,
                                            np.zeros(
                                                (actor_limit - len(actor_vector_mask), self.config['obs_horizon'] - 1),
                                                dtype=np.int32)))

        actor_cluster_mask = np.concatenate((
            np.ones(
                len(  # [C,]
                    actor_vector_list),
                dtype=np.int32),
            np.zeros(actor_limit - len(actor_vector_list), dtype=np.int32)))

        # lane
        lane_graph = LaneGraph.FromProto(sample.lane_graph)
        nbs = lane_graph.getNearestNodes(agent_origin, self.config['range'])
        lane_graph, id_map = lane_graph.getSubGraph(nbs, construct_component=True, construct_kdtree=True)

        lane_vector_list = []
        lane_vector_mask_list = []
        for comp in lane_graph.comp_son:
            st = 0
            while st < len(comp):
                this_len = min(self.config['cluster_size'], (len(comp) - st))

                lane_feature = lane_graph.node_feature[comp[st:st + this_len]]
                lane_vector_start = lane_feature[:, [0, 1]] - lane_feature[:, [2, 3]] / 2
                lane_vector_end = lane_feature[:, [0, 1]] + lane_feature[:, [2, 3]] / 2

                if rot_present:
                    lane_vector = np.hstack((
                        lane_vector_start,
                        lane_vector_end,
                        np.zeros((len(lane_vector_start), 2), dtype=np.float32),
                        np.full((len(lane_vector_start), 1), self.config['lane_tag'], dtype=np.float32),
                    ))
                else:
                    lane_vector = np.hstack((
                        lane_vector_start,
                        lane_vector_end,
                        np.full((len(lane_vector_start), 1), self.config['lane_tag'], dtype=np.float32),
                    ))

                lane_vector_len = len(lane_vector)
                lane_vector = np.vstack((
                    np.zeros(  # [V,F]
                        (self.config['obs_horizon'] - 1 - len(lane_vector), self.config['vector_size']),
                        dtype=np.float32),
                    lane_vector))

                lane_vector_mask = np.concatenate(  # [V]
                    (np.zeros(self.config['obs_horizon'] - 1 - lane_vector_len,
                              dtype=np.int32), np.ones(lane_vector_len, dtype=np.int32)))

                lane_vector_list.append(lane_vector)
                lane_vector_mask_list.append(lane_vector_mask)
                st += this_len

        lane_limit = self.config['lane_limit']
        if len(lane_vector_list) > lane_limit:
            lane_vector_list = lane_vector_list[:lane_limit]
        lane_vector = np.stack(lane_vector_list)  # [C, V, F]
        lane_vector = np.concatenate(
            (lane_vector,
             np.zeros((lane_limit - len(lane_vector), self.config['obs_horizon'] - 1, self.config['vector_size']),
                      dtype=np.float32)))

        if len(lane_vector_mask_list) > lane_limit:
            lane_vector_mask_list = lane_vector_mask_list[:lane_limit]
        lane_vector_mask = np.stack(lane_vector_mask_list)  # [C, V]
        lane_vector_mask = np.concatenate((lane_vector_mask,
                                           np.zeros(
                                               (lane_limit - len(lane_vector_mask), self.config['obs_horizon'] - 1),
                                               dtype=np.int32)))
        lane_cluster_mask = np.concatenate((
            np.ones(
                len(  # [C]
                    lane_vector_list),
                dtype=np.int32),
            np.zeros(lane_limit - len(lane_vector_list), dtype=np.int32)))

        vector = np.concatenate((actor_vector, lane_vector))  # [C, V, F]
        vector_mask = np.concatenate((actor_vector_mask, lane_vector_mask))  # [C, V]
        cluster_mask = np.concatenate((actor_cluster_mask, lane_cluster_mask))  # [C]

        ###############
        # ground truth
        ###############
        # gt pos
        gt_pos = np.array([[state.pos.x, state.pos.y] for state in agent_traj.states[self.config['obs_horizon']:]],
                          dtype=np.float32)
        gt_pos = np.matmul(gt_pos - agent_origin, agent_rotm)

        # gt pix
        gt_pix_cord = self.posToCord(gt_pos[-1], self.config['high_reso'], self.config['high_reso_dim'])
        gt_pix = self.cordToIndex(gt_pix_cord, self.config['high_reso_dim'])
        gt_pix_center = self.cordToPos(gt_pix_cord, self.config['high_reso'], self.config['high_reso_dim'])
        gt_pix_offset = gt_pos[-1] - gt_pix_center

        ###################
        # transformed data
        ###################
        data = AgentClosure()
        data.vector = torch.from_numpy(vector).unsqueeze(0)
        data.vector_mask = torch.from_numpy(vector_mask).unsqueeze(0)
        data.cluster_mask = torch.from_numpy(cluster_mask).unsqueeze(0)
        data.agent_pose = torch.from_numpy(agent_pose).unsqueeze(0)

        data.gt_pos = torch.from_numpy(gt_pos).unsqueeze(0)
        data.gt_pix = torch.LongTensor([gt_pix])
        data.gt_pix_offset = torch.from_numpy(gt_pix_offset).unsqueeze(0)

        # for viz and debug
        data.raw_sample_name = sample_name
        data.high_reso = self.config['high_reso']
        data.high_reso_dim = self.config['high_reso_dim']
        data.low_reso = self.config['low_reso']
        data.low_reso_dim = self.config['low_reso_dim']
        data.actor_limit = actor_limit
        data.lane_limit = lane_limit

        return data

    def posToCord(self, pos, reso, dim):
        '''
                x                                 x                                 x(row)
                |                                 |                                 |
                |                                 |                                 |
         y <----------                            |                                 |
                |                                 |                                 |
                |                      y <---------                 y(col) <---------
           center_coord_true_reso  ->   corner_coord_true_reso  -> corner_coord_grid_reso
        '''
        cord = np.floor((pos + dim * reso / 2) / reso)
        if isinstance(cord, np.ndarray):
            return cord.astype(np.long)
        else:
            return int(cord)

    def cordToPos(self, cord, reso, dim):
        '''
            corner_coord_grid_reso  ->  center_coord_grid_reso  -> center_coord_true_reso
        '''
        pos = (cord - (dim - 1) / 2) * reso
        if isinstance(pos, np.ndarray):
            return pos.astype(np.float32)
        else:
            return float(pos)

    def cordToIndex(self, cord, dim):
        '''
            corner_coord_grid_reso
        '''
        if cord.ndim == 2:
            cord = cord.transpose()
        cord = np.clip(cord, 0, dim - 1)  # in case of cord exceed dim
        return cord[0] + cord[1] * dim

    def posToIndex(self, pos, reso, dim):
        return self.cordToIndex(self.posToCord(pos, reso, dim), dim)

    def rotmFromVect(self, v0, v1):
        rotv = v1 - v0
        norm = np_vector_norm(rotv)
        if norm < 1e-6:
            cos_a, sin_a = 1.0, 0.0
        else:
            cos_a, sin_a = rotv / norm

        rotm = np.array([[cos_a, -sin_a], [sin_a, +cos_a]], dtype=np.float32)

        return rotm


class IntentionTransform8RandomMask(object):

    def __init__(self, config):
        self.config = config
        self.len_candi = [1, 3, 5, 8, 12, 15, 19, 19]

    def __call__(self, data):
        agent_len = random.choice(self.len_candi)
        mask_len = self.config['cluster_size'] - agent_len
        if mask_len > 0 and ('mask' in self.config and self.config['mask']):
            data.vector_mask[0, 0, :mask_len - 1] = 0

        return data


class IntentionTransform8Batch(AgentClosureBatch):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def __inc__(cls, key, data):
        return 0

    @classmethod
    def __cat_dim__(cls, key, data):
        return 0

    @classmethod
    def __init_cumsum__(cls, key, data):
        return 0


def PlotIntentionTransform8(data, show=True):
    plt.figure(data.raw_sample_name)
    plt.axis('equal')

    agent_pose = data.agent_pose[0]
    agent_origin = agent_pose[0:2]
    agent_rot = agent_pose[2:4]
    agent_rotm = torch.FloatTensor([[agent_rot[0], -agent_rot[1]], [agent_rot[1], agent_rot[0]]])

    # heatmap
    if hasattr(data, 'pred_heatmap'):
        dim = data.high_reso_dim
        reso = data.high_reso
        heatmap = data.pred_heatmap.view(dim, dim)
        x = (np.arange(dim) - dim / 2) * reso + 0.5 * reso
        y = (np.arange(dim) - dim / 2) * reso + 0.5 * reso
        # Wistia, YlOrBr, Oranges, YlOrBr, YlOrRd, YlGnBu, YlGn, hot
        # ref: https://matplotlib.org/stable/tutorials/colors/colormaps.html
        plt.pcolormesh(x, y, heatmap, cmap='Wistia', alpha=1.0, shading='auto')
    else:
        dim = data.high_reso_dim
        reso = data.high_reso
        heatmap = torch.zeros(dim * dim)
        heatmap[data.gt_pix] = 1.0
        heatmap = heatmap.view(dim, dim)
        x = (np.arange(dim) - dim / 2) * reso + 0.5 * reso
        y = (np.arange(dim) - dim / 2) * reso + 0.5 * reso
        # Wistia, YlOrBr, Oranges, YlOrBr, YlOrRd, YlGnBu, YlGn, hot
        # ref: https://matplotlib.org/stable/tutorials/colors/colormaps.html
        plt.pcolormesh(x, y, heatmap, cmap='Wistia', alpha=1.0, shading='auto')

    # plot lane
    for i in range(data.actor_limit, data.actor_limit + data.lane_limit):
        lane_vector = data.vector[0, i]
        if lane_vector[-1, -1] == 0.0:  # this cluster is padding [ zero|normal, other|tag]
            continue

        st_idx = torch.where(lane_vector[:, -1] != 0.0)[0][0]
        lane_vector = lane_vector[st_idx:, :-1]

        lane_vector_start = torch.matmul(lane_vector[:, [0, 1]] - agent_origin, agent_rotm)
        lane_vector_end = torch.matmul(lane_vector[:, [2, 3]] - agent_origin, agent_rotm)
        plot_vector(lane_vector_start, lane_vector_end, 'gray', 0.3, 0.3, 1.5, 0.7)

    # plot actor
    for i in range(0, data.actor_limit):
        actor_vector = data.vector[0, i]  # [agent|other, zero|normal, other|tag]
        if actor_vector[-1, -1] == 0.0 and i > 0:  # this cluster is padding [zero|normal, other|tag]
            continue

        if i > 0:
            st_idx = torch.where(actor_vector[:, -1] != 0.0)[0][0]
        else:
            st_idx = 0

        actor_vector = actor_vector[st_idx:, :-1]

        actor_vector_start = torch.matmul(actor_vector[:, [0, 1]] - agent_origin, agent_rotm)
        actor_vector_end = torch.matmul(actor_vector[:, [2, 3]] - agent_origin, agent_rotm)
        color = 'green' if i == 0 else 'blue'
        plot_vector(actor_vector_start, actor_vector_end, color, 0.1, 1.0, 1.5, 0.7)

    # plot pred traj
    if hasattr(data, 'pred_traj'):
        for pred_traj in data.pred_traj:
            traj = torch.cat([torch.FloatTensor([[0, 0]]), pred_traj], dim=0)
            plot_vector(traj[:-1], traj[1:], 'yellow', 0.1, 1.0, 1.5, 0.7)

    # plot gt
    gt_pos = torch.cat([torch.FloatTensor([[0, 0]]), data.gt_pos[0]], dim=0)
    plot_vector(gt_pos[:-1], gt_pos[1:], 'red', 0.1, 1.0, 1.5, 0.7)

    if show:
        plt.show()

    return plt


if __name__ == '__main__':
    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print(f'{config_file} not exists!!')
    config_dir = os.path.dirname(config_file)
    output_dir = sys.argv[2]

    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=config_dir)
    with open(config_file, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    database = SampleDatabase.create(config['validation']['sample_database_folder'])

    transform = IntentionTransform8(config['transform'])

    with open(config['validation']['sample_list_file'], 'r') as fin:
        lines = fin.readlines()
    sample_name_list = [line.strip() for line in lines]

    for sample_name in tqdm(sample_name_list):
        data = transform(sample_name, database.get(sample_name))
        fig = PlotIntentionTransform8(data.to('cpu'), show=False)
        fig.savefig(f'{output_dir}/{data.raw_sample_name}.svg')
        break
