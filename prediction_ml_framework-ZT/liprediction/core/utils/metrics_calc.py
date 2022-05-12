# Copyright (c) 2021 Li Auto Company. All rights reserved.

import torch
from liprediction.datasets.processer.feature_processer import (ACTOR_FEATURE_END_X, ACTOR_FEATURE_END_Y,
                                                               ACTOR_FEATURE_VECTOR_HEADING, ACTOR_FEATURE_VELOCITY_X,
                                                               ACTOR_FEATURE_VELOCITY_Y)

TRAJ_X = 0
TRAJ_Y = 1
TRAJ_VELOCITY = 2
TRAJ_HEADING = 3

THRESHOLD_LON_8S = 6
THRESHOLD_LAT_8S = 3
LOWER_FACTOR = 0.5
UPPER_FACTOR = 1.0
LOWER_SPEED_THRESHOLD = 1.4
UPPER_SPEED_THRESHOLD = 11.0


def gt_vector_to_gt_traj(vector):
    '''
    Generate ground truth trajectory with features we want

    Args:
        vector: batch.actor_future_vector

    Returns:
        traj
    '''
    traj = torch.zeros([vector.shape[0], vector.shape[1], vector.shape[2], 4], dtype=torch.float).type_as(vector)
    traj[:, :, :, TRAJ_X] = vector[:, :, :, ACTOR_FEATURE_END_X]
    traj[:, :, :, TRAJ_Y] = vector[:, :, :, ACTOR_FEATURE_END_Y]
    traj[:, :, :, TRAJ_VELOCITY] = torch.sqrt(vector[:, :, :, ACTOR_FEATURE_VELOCITY_X]**2 +
                                              vector[:, :, :, ACTOR_FEATURE_VELOCITY_Y]**2)
    traj[:, :, :, TRAJ_VELOCITY] = vector[:, :, :, ACTOR_FEATURE_VECTOR_HEADING]
    return traj


def miss_judge(traj_predict, traj_gt, end_timestep_index):
    '''
    Judge miss for every cluster prediction

    Args:
        traj_predict
        traj_gt
        end_timestep_index: corresponding to timestep index of 3s, 5s and 8s

    Returns:
        miss.bool(): one-dimensional tensor of True & False
        MR: MissRate
    '''
    traj_predict = traj_predict.view(traj_predict.shape[0], traj_predict.shape[1], -1, 2)
    traj_gt = traj_gt.view(-1, traj_gt.shape[-2], traj_gt.shape[-1])
    traj_gt = traj_gt.unsqueeze(1).repeat(1, traj_predict.shape[1], 1, 1)
    # traj_predict : [B, K, T, F] , traj_gt : [B, K, T, F]
    dis_x = torch.abs(traj_gt[:, :, end_timestep_index, TRAJ_X] - traj_predict[:, :, end_timestep_index, TRAJ_X])
    dis_y = torch.abs(traj_gt[:, :, end_timestep_index, TRAJ_Y] - traj_predict[:, :, end_timestep_index, TRAJ_Y])
    dis_lon = torch.abs(dis_x * torch.cos(traj_gt[:, :, end_timestep_index, TRAJ_HEADING]) +
                        dis_y * torch.sin(traj_gt[:, :, end_timestep_index, TRAJ_HEADING]))
    dis_lat = torch.abs(dis_x * torch.sin(traj_gt[:, :, end_timestep_index, TRAJ_HEADING]) -
                        dis_y * torch.cos(traj_gt[:, :, end_timestep_index, TRAJ_HEADING]))

    speed = traj_gt[:, :, end_timestep_index, TRAJ_VELOCITY]
    low_speed = (torch.ones_like(speed) * LOWER_SPEED_THRESHOLD).type_as(speed)
    high_speed = (torch.ones_like(speed) * UPPER_SPEED_THRESHOLD).type_as(speed)
    low_factor = (torch.ones_like(speed) * LOWER_FACTOR).type_as(speed)
    high_factor = (torch.ones_like(speed) * UPPER_FACTOR).type_as(speed)
    factor = torch.ones_like(speed).type_as(speed)
    factor[speed < low_speed] = low_factor[speed < low_speed]
    factor[speed >= high_speed] = high_factor[speed >= high_speed]
    bool_index = torch.logical_and(speed >= low_speed, speed < high_speed)
    slope = (speed - low_speed) / (high_speed - low_speed)
    factor[bool_index] = (low_factor + slope * (high_factor - low_factor))[bool_index]
    miss_traj = torch.zeros_like(speed).int()
    miss_index = torch.logical_or(dis_lat >= factor * THRESHOLD_LAT_8S, dis_lon >= factor * THRESHOLD_LON_8S)
    miss_traj[miss_index] = 1
    miss = torch.zeros(speed.shape[0], dtype=torch.int64)
    all_miss_index = (torch.sum(miss_traj, dim=-1) == speed.shape[1])
    miss[all_miss_index] = 1
    MR = torch.sum(miss, dim=0) / miss.shape[0]
    return miss.bool(), MR


def minADE_calc(traj_predict, traj_gt, end_timestep_index):
    '''
    Calculate minADE for every cluster prediction

    Args:
        traj_predict
        traj_gt
        end_timestep_index: corresponding to timestep index of 3s, 5s and 8s

    Returns:
        minADE: one-dimensional tensor of minADE
    '''
    traj_predict = traj_predict.view(traj_predict.shape[0], traj_predict.shape[1], -1, 2)
    traj_gt = traj_gt.view(-1, traj_gt.shape[-2], traj_gt.shape[-1])
    traj_gt = traj_gt.unsqueeze(1).repeat(1, traj_predict.shape[1], 1, 1)
    dis_x = torch.abs(traj_gt[:, :, :, TRAJ_X] - traj_predict[:, :, :, 0])
    dis_y = torch.abs(traj_gt[:, :, :, TRAJ_Y] - traj_predict[:, :, :, 1])
    l2 = torch.sqrt(dis_x**2 + dis_y**2)
    aver_l2 = torch.sum(l2[:, :, 0:(end_timestep_index + 1)], dim=2) / traj_predict.shape[2]
    minADE, _ = torch.min(aver_l2, dim=-1)
    return minADE


# Return one-dimensional tensor of minFDE
def minFDE_calc(traj_predict, traj_gt, end_timestep_index):
    '''
    Calculate minFDE for every cluster prediction

    Args:
        traj_predict
        traj_gt
        end_timestep_index: corresponding to timestep index of 3s, 5s and 8s

    Returns:
        minFDE: one-dimensional tensor of minFDE
    '''
    traj_predict = traj_predict.view(traj_predict.shape[0], traj_predict.shape[1], -1, 2)
    traj_gt = traj_gt.view(-1, traj_gt.shape[-2], traj_gt.shape[-1])
    traj_gt = traj_gt.unsqueeze(1).repeat(1, traj_predict.shape[1], 1, 1)
    dis_x = torch.abs(traj_gt[:, :, end_timestep_index, TRAJ_X] - traj_predict[:, :, end_timestep_index, 0])
    dis_y = torch.abs(traj_gt[:, :, end_timestep_index, TRAJ_Y] - traj_predict[:, :, end_timestep_index, 1])
    final_l2 = torch.sqrt(dis_x**2 + dis_y**2)
    minFDE, _ = torch.min(final_l2, dim=-1)
    return minFDE
