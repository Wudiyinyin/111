# Copyright (c) 2021 Li Auto Company. All rights reserved.

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from core.utils.grid import gen_grid_center_pos, index_to_pos, pos_to_index
from core.utils.utils import normal_angle, rotm_from_vect
from datasets.database.sample_database import SampleDataset
from datasets.database.utils.plot_new_sample import plot_vector
from datasets.transform.agent_closure import AgentClosure
from tqdm import tqdm
from yamlinclude import YamlIncludeConstructor

ACTOR_FEATURE_START_X = 0
ACTOR_FEATURE_START_Y = 1
ACTOR_FEATURE_END_X = 2
ACTOR_FEATURE_END_Y = 3
ACTOR_FEATURE_POLYLINE_TYPE = 4
ACTOR_FEATURE_TYPE = 5
ACTOR_FEATURE_LENGTH = 6
ACTOR_FEATURE_WIDTH = 7
ACTOR_FEATURE_HEIGHT = 8
ACTOR_FEATURE_VECTOR_HEADING = 9
ACTOR_FEATURE_VELOCITY_X = 10
ACTOR_FEATURE_VELOCITY_Y = 11
ACTOR_FEATURE_VELOCITY_HDEADING = 12
ACTOR_FEATURE_IS_INTERPOLATION = 13
ACTOR_FEATURE_TIME_EMBED = 14
ACTOR_FEATURE_HIDDEN_MASK = 15
ACTOR_FEATURE_SIZE = 16

MAP_FEATURE_START_X = 0
MAP_FEATURE_START_Y = 1
MAP_FEATURE_END_X = 2
MAP_FEATURE_END_Y = 3
MAP_FEATURE_POLYLINE_TYPE = 4
MAP_FEATURE_LANE_CENTER_TYPE = 5
MAP_FEATURE_LANE_BOUNDARY_TYPE = 6
MAP_FEATURE_ROAD_BOUNDARY_ROAD_TYPE = 7
MAP_FEATURE_ROAD_BOUNDARY_EDGE_TYPE = 8
MAP_FEATURE_POLYGON_TYPE = 9
MAP_FEATURE_VECTOR_HEADING = 10
MAP_FEATURE_IS_VIRTUAL = 11
MAP_FEATURE_SPEED_LIMIT = 12
MAP_FEATURE_STOP_SIGN = 13
MAP_FEATURE_TIME_EMBED = 14
MAP_FEATURE_HIDDEN_MASK = 15
MAP_FEATURE_SIZE = 16


def gen_actor_feature(frame, time_embed, hidden_mask):
    feature = [0] * ACTOR_FEATURE_SIZE
    if frame is not None:
        feature[ACTOR_FEATURE_START_X] = frame.vec.points[0].x
        feature[ACTOR_FEATURE_START_Y] = frame.vec.points[0].y
        feature[ACTOR_FEATURE_END_X] = frame.vec.points[1].x
        feature[ACTOR_FEATURE_END_Y] = frame.vec.points[1].y
        feature[ACTOR_FEATURE_POLYLINE_TYPE] = frame.vec.polyline_type
        feature[ACTOR_FEATURE_TYPE] = frame.type
        feature[ACTOR_FEATURE_LENGTH] = frame.length
        feature[ACTOR_FEATURE_WIDTH] = frame.width
        feature[ACTOR_FEATURE_HEIGHT] = frame.height
        feature[ACTOR_FEATURE_VECTOR_HEADING] = frame.heading
        feature[ACTOR_FEATURE_VELOCITY_X] = frame.velocity.x
        feature[ACTOR_FEATURE_VELOCITY_Y] = frame.velocity.y
        feature[ACTOR_FEATURE_VELOCITY_HDEADING] = np.arctan2(frame.velocity.y, frame.velocity.x)
        feature[ACTOR_FEATURE_IS_INTERPOLATION] = frame.is_interpolation_vec
    feature[ACTOR_FEATURE_TIME_EMBED] = time_embed
    feature[ACTOR_FEATURE_HIDDEN_MASK] = hidden_mask
    return feature


def gen_actor_history_vector(obstacle, history_limit, feature_len):
    obstacle_curr_seq_num = obstacle.seq_num
    obstacle_raw_len = len(obstacle.obs_vec)
    # TODO check length (assert length > min_size)

    # select all sample
    start_idx = 0
    if history_limit < obstacle_raw_len:
        # select some sample
        start_idx = obstacle_raw_len - history_limit

    obstacle_len = 0
    obstacle_vector = []
    obstacle_select_seq_num = []
    # assert history traj is continuous
    for idx, frame in enumerate(obstacle.obs_vec):
        if idx < start_idx:
            continue

        # handle time embedding (start,...,current = -n,...,0)
        # time_embed = frame.seq_num - obstacle_curr_seq_num
        # time embed as [1,91] to avoid special value `-1` `0`, wich used as mask value
        time_embed = frame.seq_num
        hidden_mask = 1.0  # 1.0 valid 0.0: hidden
        # padding_mask = 1.0  # 1.0 valid 0.0: mask
        feature = gen_actor_feature(frame, time_embed, hidden_mask)
        obstacle_vector.append(feature)

        if len(obstacle_select_seq_num) > 0:
            assert frame.seq_num == obstacle_select_seq_num[-1] + 1, "seq_num must be continuous num"
        obstacle_select_seq_num.append(frame.seq_num)
        obstacle_len += 1

    assert obstacle_len <= history_limit
    assert obstacle_select_seq_num[-1] == obstacle_curr_seq_num
    # TODO check distance

    obstacle_vector = np.array(obstacle_vector, dtype=np.float32)
    assert obstacle_vector.shape == (obstacle_len, feature_len)
    # check time embed
    # assert obstacle_vector[-1, ACTOR_FEATURE_TIME_EMBED] == 0.0
    obstacle_pading = np.zeros((history_limit - obstacle_len, feature_len), dtype=np.float32)

    obstacle_vector_mask = np.ones(obstacle_len, dtype=np.int32)
    obstacle_padding_mask = np.zeros(history_limit - obstacle_len, dtype=np.int32)

    # |--pad--prev--|curr|--future--pad--|
    obstacle_vector_pad = np.vstack((obstacle_pading, obstacle_vector))
    assert obstacle_vector_pad.shape == (history_limit, feature_len)
    obstacle_vector_mask_pad = np.concatenate((obstacle_padding_mask, obstacle_vector_mask))
    assert obstacle_vector_mask_pad.shape == (history_limit,)

    return obstacle_vector_pad, obstacle_vector_mask_pad, obstacle_select_seq_num, obstacle_vector


def gen_actor_future_vector(obstacle, future_limit, feature_len):
    obstacle_curr_seq_num = obstacle.seq_num
    # obstacle_future_raw_len = len(obstacle.obs_future_vec)
    # TODO check length (assert length > min_size)

    # future
    obstacle_future_len = 0
    obstacle_future_vector = []
    obstacle_future_select_seq_num = []
    obstacle_future_padding_mask_tmp = []
    obstacle_future_last_seq_num = obstacle_curr_seq_num
    # future traj maybe not continuous
    for idx, frame in enumerate(obstacle.obs_future_vec):
        # check seq_num is continus
        if frame.seq_num != obstacle_future_last_seq_num + 1:
            padding_num = frame.seq_num - obstacle_future_last_seq_num - 1
            for i in range(padding_num):
                # handle time embedding (current,... = 1,...)
                curr_seq_num = obstacle_future_last_seq_num + 1 + i
                # time_embed = curr_seq_num - obstacle_curr_seq_num
                # time embed as [1,91] to avoid special value `-1` `0`, wich used as mask value
                time_embed = curr_seq_num
                hidden_mask = 0.0  # 1.0 valid 0.0: hidden
                feature = gen_actor_feature(None, time_embed, hidden_mask)
                obstacle_future_vector.append(feature)
                # this feature is padding
                obstacle_future_padding_mask_tmp.append(0)
                obstacle_future_select_seq_num.append(curr_seq_num)
                obstacle_future_last_seq_num = curr_seq_num
                obstacle_future_len += 1

        # handle time embedding (current,... = 1,...)
        # time_embed = frame.seq_num - obstacle_curr_seq_num
        # time embed as [1,91] to avoid special value `-1` `0`, wich used as mask value
        time_embed = frame.seq_num
        hidden_mask = 0.0  # 1.0 valid 0.0: hidden
        # padding_mask = 1.0  # 1.0 valid 0.0: mask
        feature = gen_actor_feature(frame, time_embed, hidden_mask)
        obstacle_future_vector.append(feature)
        # this feature is not padding
        obstacle_future_padding_mask_tmp.append(1)
        obstacle_future_select_seq_num.append(frame.seq_num)
        obstacle_future_last_seq_num = frame.seq_num
        obstacle_future_len += 1

    # TODO check distance
    assert obstacle_future_len <= future_limit
    # time embed start from 1.0
    # if len(obstacle_future_vector) > 0:
    #     assert obstacle_future_vector[0][ACTOR_FEATURE_TIME_EMBED] == 1

    if obstacle_future_len == 0:
        obstacle_future_vector = np.empty((obstacle_future_len, feature_len), dtype=np.float32)
        obstacle_future_vector_mask = np.empty((obstacle_future_len), dtype=np.int32)
    else:
        obstacle_future_vector = np.array(obstacle_future_vector, dtype=np.float32)
        obstacle_future_vector_mask = np.array(obstacle_future_padding_mask_tmp, dtype=np.int32)
    assert obstacle_future_vector.shape == (obstacle_future_len, feature_len)
    assert obstacle_future_vector_mask.shape == (obstacle_future_len,)

    obstacle_future_pading = np.zeros((future_limit - obstacle_future_len, feature_len), dtype=np.float32)
    obstacle_future_padding_mask = np.zeros(future_limit - obstacle_future_len, dtype=np.int32)

    # |--pad--prev--|curr|--future--pad--|
    obstacle_future_vector_pad = np.vstack((obstacle_future_vector, obstacle_future_pading))
    assert obstacle_future_vector_pad.shape == (future_limit, feature_len)
    obstacle_future_vector_mask_pad = np.concatenate((obstacle_future_vector_mask, obstacle_future_padding_mask))
    assert obstacle_future_vector_mask_pad.shape == (future_limit,)

    return obstacle_future_vector_pad, obstacle_future_vector_mask_pad, \
        obstacle_future_select_seq_num, obstacle_future_vector


def transform_actor_featrue(actor_vector_pad, actor_vector_mask_pad, agent_origin, agent_rotm):
    agent_angle = np.arctan2(agent_rotm[1, 0], agent_rotm[0, 0])

    # actor_vector_pad
    # start_x start_y
    actor_vector_pad[..., [ACTOR_FEATURE_START_X, ACTOR_FEATURE_START_Y]] = np.matmul(
        actor_vector_pad[..., [ACTOR_FEATURE_START_X, ACTOR_FEATURE_START_Y]] - agent_origin, agent_rotm)
    # end_x end_y
    actor_vector_pad[..., [ACTOR_FEATURE_END_X, ACTOR_FEATURE_END_Y]] = np.matmul(
        actor_vector_pad[..., [ACTOR_FEATURE_END_X, ACTOR_FEATURE_END_Y]] - agent_origin, agent_rotm)
    # vector heading
    actor_vector_pad[...,
                     [ACTOR_FEATURE_VECTOR_HEADING]] = normal_angle(actor_vector_pad[...,
                                                                                     [ACTOR_FEATURE_VECTOR_HEADING]] -
                                                                    agent_angle)
    # velocity_x velocity_y
    actor_vector_pad[..., [ACTOR_FEATURE_VELOCITY_X, ACTOR_FEATURE_VELOCITY_Y]] = np.matmul(
        actor_vector_pad[..., [ACTOR_FEATURE_VELOCITY_X, ACTOR_FEATURE_VELOCITY_Y]] - agent_origin, agent_rotm)
    # velocity heading
    actor_vector_pad[..., [ACTOR_FEATURE_VELOCITY_HDEADING]] = normal_angle(
        actor_vector_pad[..., [ACTOR_FEATURE_VELOCITY_HDEADING]] - agent_angle)

    # we should only set the converted field to 0, keep 'TIME_EMBED, HIDDEN_MASK' field unchanged,
    # np.set_printoptions(threshold=sys.maxsize)
    actor_vector_pad_fixed = actor_vector_pad[..., [
        ACTOR_FEATURE_START_X, ACTOR_FEATURE_START_Y, ACTOR_FEATURE_END_X, ACTOR_FEATURE_END_Y,
        ACTOR_FEATURE_VECTOR_HEADING, ACTOR_FEATURE_VELOCITY_X, ACTOR_FEATURE_VELOCITY_Y,
        ACTOR_FEATURE_VELOCITY_HDEADING
    ]]
    actor_vector_pad_fixed[actor_vector_mask_pad == 0] = 0.0
    actor_vector_pad[..., [
        ACTOR_FEATURE_START_X, ACTOR_FEATURE_START_Y, ACTOR_FEATURE_END_X, ACTOR_FEATURE_END_Y,
        ACTOR_FEATURE_VECTOR_HEADING, ACTOR_FEATURE_VELOCITY_X, ACTOR_FEATURE_VELOCITY_Y,
        ACTOR_FEATURE_VELOCITY_HDEADING
    ]] = actor_vector_pad_fixed

    # the following method not work
    # actor_vector_pad[actor_vector_mask_pad == 0][..., [
    #    ACTOR_FEATURE_START_X, ACTOR_FEATURE_START_Y, ACTOR_FEATURE_END_X, ACTOR_FEATURE_END_Y,
    #    ACTOR_FEATURE_VECTOR_HEADING, ACTOR_FEATURE_VELOCITY_X, ACTOR_FEATURE_VELOCITY_Y,
    #    ACTOR_FEATURE_VELOCITY_HDEADING
    # ]] = 0.0
    # actor_vector_pad[..., [
    #    ACTOR_FEATURE_START_X, ACTOR_FEATURE_START_Y, ACTOR_FEATURE_END_X, ACTOR_FEATURE_END_Y,
    #    ACTOR_FEATURE_VECTOR_HEADING, ACTOR_FEATURE_VELOCITY_X, ACTOR_FEATURE_VELOCITY_Y,
    #    ACTOR_FEATURE_VELOCITY_HDEADING
    # ]][actor_vector_mask_pad == 0] = 0.0
    return actor_vector_pad


def transform_map_featrue(map_vector_pad, map_vector_mask_pad, agent_origin, agent_rotm):
    # map_vector
    agent_angle = np.arctan2(agent_rotm[1, 0], agent_rotm[0, 0])
    # map_vector_pad
    # start_x start_y
    map_vector_pad[..., [MAP_FEATURE_START_X, MAP_FEATURE_START_Y]] = np.matmul(
        map_vector_pad[..., [MAP_FEATURE_START_X, MAP_FEATURE_START_Y]] - agent_origin, agent_rotm)
    # end_x end_y
    map_vector_pad[..., [MAP_FEATURE_END_X, MAP_FEATURE_END_Y]] = np.matmul(
        map_vector_pad[..., [MAP_FEATURE_END_X, MAP_FEATURE_END_Y]] - agent_origin, agent_rotm)
    # vector heading
    map_vector_pad[..., [MAP_FEATURE_VECTOR_HEADING]] = normal_angle(map_vector_pad[..., [MAP_FEATURE_VECTOR_HEADING]] -
                                                                     agent_angle)

    # we should only set the converted field to 0, keep 'TIME_EMBED, HIDDEN_MASK' field unchanged,
    map_vector_pad_fixed = map_vector_pad[
        ...,
        [MAP_FEATURE_START_X, MAP_FEATURE_START_Y, MAP_FEATURE_END_X, MAP_FEATURE_END_Y, MAP_FEATURE_VECTOR_HEADING]]
    map_vector_pad_fixed[map_vector_mask_pad == 0] = 0.0
    map_vector_pad[
        ...,
        [MAP_FEATURE_START_X, MAP_FEATURE_START_Y, MAP_FEATURE_END_X, MAP_FEATURE_END_Y, MAP_FEATURE_VECTOR_HEADING
        ]] = map_vector_pad_fixed
    return map_vector_pad


def transform_traj(gt_traj, gt_traj_mask, agent_origin, agent_rotm):
    # shape=[K, 2]  ((R_2x2)^T *(t_2xN-t_2x1))^T -> (t_Nx2 - t_1x2)*R_2x2
    gt_traj = np.matmul(gt_traj - agent_origin, agent_rotm)
    gt_traj[gt_traj_mask == 0] = 0.0
    return gt_traj


class FeatureProcesser():

    def __init__(self, config) -> None:
        self.config = config
        self.scene_id_to_map = None
        self.is_agent_cord = True

    def get_agent_polyline_ids(self, sample, agent_id):
        agent_polyline_ids = {'Obstacle': [], 'Lane': [], 'LaneBoundary': [], 'RoadBoundary': [], 'Polygon': []}
        for obstacle in sample.obstacle_features:
            if obstacle.id == agent_id:
                # print(f"obstacle {obstacle.id} has {len(obstacle.polyline_info)} polyline_info")
                for polyline in obstacle.polyline_info:
                    # print(f'  polyline_id: {polyline.polyline_id}  polyline_type: {polyline.type}')
                    if polyline.type == polyline.Obstacle:
                        agent_polyline_ids['Obstacle'].append(int(polyline.polyline_id))
                    elif polyline.type == polyline.Lane:
                        agent_polyline_ids['Lane'].append(polyline.polyline_id)
                    elif polyline.type == polyline.LaneBoundary:
                        agent_polyline_ids['LaneBoundary'].append(polyline.polyline_id)
                    elif polyline.type == polyline.RoadBoundary:
                        agent_polyline_ids['RoadBoundary'].append(polyline.polyline_id)
                    elif polyline.type == polyline.Polygon:
                        agent_polyline_ids['Polygon'].append(polyline.polyline_id)
        return agent_polyline_ids

    def process_obstacle(self, sample, agent_id, valid_polyline_ids, history_limit, feature_len, future_limit):
        # obstacle
        obstacle_vector_list = []
        obstacle_vector_mask_list = []
        obstacle_future_vector_list = []
        obstacle_future_vector_mask_list = []
        for obstacle in sample.obstacle_features:
            # check in local map
            if obstacle.id != agent_id and valid_polyline_ids and not (obstacle.id in valid_polyline_ids):
                continue

            (obstacle_vector_pad, obstacle_vector_mask_pad, obstacle_select_seq_num,
             obstacle_vector) = gen_actor_history_vector(obstacle, history_limit, feature_len)

            (obstacle_future_vector_pad, obstacle_future_vector_mask_pad, obstacle_future_select_seq_num,
             obstacle_future_vector) = gen_actor_future_vector(obstacle, future_limit, feature_len)

            # check seq_num continuous
            if len(obstacle_future_select_seq_num) > 0:
                assert obstacle_select_seq_num[-1] + 1 == obstacle_future_select_seq_num[0]

            if obstacle.id == agent_id:
                agent_vector = obstacle_vector
                agent_vector_pad = obstacle_vector_pad
                agent_vector_mask_pad = obstacle_vector_mask_pad
                # agent_future_vector = obstacle_future_vector
                # agent_future_vector_mask = obstacle_future_vector_mask
                agent_future_vector_pad = obstacle_future_vector_pad
                agent_future_vector_mask_pad = obstacle_future_vector_mask_pad

                # pose
                agent_start = agent_vector[-1, [ACTOR_FEATURE_START_X, ACTOR_FEATURE_START_Y]]  # [2]
                agent_origin = agent_vector[-1, [ACTOR_FEATURE_END_X, ACTOR_FEATURE_END_Y]]  # [2]
                agent_rotm = rotm_from_vect(agent_start, agent_origin)
                agent_rot = agent_rotm[:, 0]  # [2] cos sin
                # [4] (x,y,cos,sin)
                agent_pose = np.concatenate((agent_origin, agent_rot))
                agent_gt_traj_pad = agent_future_vector_pad[:, [ACTOR_FEATURE_END_X, ACTOR_FEATURE_END_Y]]
                agent_gt_traj_mask_pad = agent_future_vector_mask_pad
            else:
                obstacle_vector_list.append(obstacle_vector_pad)
                obstacle_vector_mask_list.append(obstacle_vector_mask_pad)
                obstacle_future_vector_list.append(obstacle_future_vector_pad)
                obstacle_future_vector_mask_list.append(obstacle_future_vector_mask_pad)

        actor_vector_list = [agent_vector_pad] + obstacle_vector_list
        actor_vector_mask_list = [agent_vector_mask_pad] + obstacle_vector_mask_list
        actor_future_vector_list = [agent_future_vector_pad] + obstacle_future_vector_list
        actor_future_vector_mask_list = [agent_future_vector_mask_pad] + obstacle_future_vector_mask_list

        return (actor_vector_list, actor_vector_mask_list, actor_future_vector_list, actor_future_vector_mask_list,
                agent_pose, agent_origin, agent_rotm, agent_gt_traj_pad, agent_gt_traj_mask_pad)

    def process_lane_center(self, sample, agent_id, valid_polyline_ids, vector_limit, feature_len, agent_origin,
                            agent_rotm):
        if (len(sample.map_polyline.lanes) == 0) and (self.scene_id_to_map is not None and
                                                      sample.scene_id in self.scene_id_to_map):
            all_lanes = self.scene_id_to_map[sample.scene_id].lanes
        else:
            all_lanes = sample.map_polyline.lanes

        lane_vector_list = []
        lane_vector_mask_list = []
        for lane in all_lanes:
            vectors = lane.lane_vector

            # TODO check length
            # lane_raw_len = len(vectors)

            # check in local map
            if valid_polyline_ids and not (lane.id in valid_polyline_ids):
                continue

            lane_len = 0
            lane_vector = []
            for idx, node in enumerate(vectors):
                time_embed = idx
                hidden_mask = 1.0  # 1.0: valid  0.0: hidden

                feature = [0] * MAP_FEATURE_SIZE
                feature[MAP_FEATURE_START_X] = node.vec.points[0].x
                feature[MAP_FEATURE_START_Y] = node.vec.points[0].y
                feature[MAP_FEATURE_END_X] = node.vec.points[1].x
                feature[MAP_FEATURE_END_Y] = node.vec.points[1].y
                feature[MAP_FEATURE_POLYLINE_TYPE] = node.vec.polyline_type
                feature[MAP_FEATURE_LANE_CENTER_TYPE] = node.lane_type
                feature[MAP_FEATURE_LANE_BOUNDARY_TYPE] = -1
                feature[MAP_FEATURE_ROAD_BOUNDARY_ROAD_TYPE] = -1
                feature[MAP_FEATURE_ROAD_BOUNDARY_EDGE_TYPE] = -1
                feature[MAP_FEATURE_POLYGON_TYPE] = -1
                feature[MAP_FEATURE_VECTOR_HEADING] = node.heading
                feature[MAP_FEATURE_IS_VIRTUAL] = node.is_virtual
                feature[MAP_FEATURE_SPEED_LIMIT] = node.speed_limit
                feature[MAP_FEATURE_STOP_SIGN] = node.stop_sign
                feature[MAP_FEATURE_TIME_EMBED] = time_embed
                feature[MAP_FEATURE_HIDDEN_MASK] = hidden_mask
                lane_vector.append(feature)

                lane_len += 1
                if lane_len >= vector_limit:
                    break

            lane_vector = np.array(lane_vector, dtype=np.float32)
            lane_padding = np.zeros((vector_limit - lane_len, feature_len), dtype=np.float32)
            assert lane_vector.shape[0] == lane_len
            assert lane_vector.shape[1] == feature_len

            lane_vector_mask = np.ones(lane_len, dtype=np.int32)
            lane_padding_mask = np.zeros(vector_limit - lane_len, dtype=np.int32)

            lane_vector_pad = np.vstack((lane_padding, lane_vector))
            assert lane_vector_pad.shape[0] == vector_limit
            assert lane_vector_pad.shape[1] == feature_len
            lane_vector_mask_pad = np.concatenate((lane_padding_mask, lane_vector_mask))
            assert lane_vector_mask_pad.shape[0] == vector_limit

            lane_vector_list.append(lane_vector_pad)
            lane_vector_mask_list.append(lane_vector_mask_pad)

        return lane_vector_list, lane_vector_mask_list

    def process_lane_boundary(self, sample, agent_id, valid_polyline_ids, vector_limit, feature_len, agent_origin,
                              agent_rotm):
        if (len(sample.map_polyline.lane_boundarys) == 0) and (self.scene_id_to_map is not None and
                                                               sample.scene_id in self.scene_id_to_map):
            all_lanes = self.scene_id_to_map[sample.scene_id].lane_boundarys
        else:
            all_lanes = sample.map_polyline.lane_boundarys

        lane_vector_list = []
        lane_vector_mask_list = []
        for lane in all_lanes:
            vectors = lane.lane_boundary_vec

            # TODO check length
            # lane_raw_len = len(vectors)

            # check in local map
            if valid_polyline_ids and not (lane.id in valid_polyline_ids):
                continue

            lane_len = 0
            lane_vector = []
            for idx, node in enumerate(vectors):
                time_embed = idx
                hidden_mask = 1.0  # 1.0: valid  0.0: hidden

                feature = [0] * MAP_FEATURE_SIZE
                feature[MAP_FEATURE_START_X] = node.vec.points[0].x
                feature[MAP_FEATURE_START_Y] = node.vec.points[0].y
                feature[MAP_FEATURE_END_X] = node.vec.points[1].x
                feature[MAP_FEATURE_END_Y] = node.vec.points[1].y
                feature[MAP_FEATURE_POLYLINE_TYPE] = node.vec.polyline_type
                feature[MAP_FEATURE_LANE_CENTER_TYPE] = -1
                feature[MAP_FEATURE_LANE_BOUNDARY_TYPE] = node.type
                feature[MAP_FEATURE_ROAD_BOUNDARY_ROAD_TYPE] = -1
                feature[MAP_FEATURE_ROAD_BOUNDARY_EDGE_TYPE] = -1
                feature[MAP_FEATURE_POLYGON_TYPE] = -1
                feature[MAP_FEATURE_VECTOR_HEADING] = node.heading
                feature[MAP_FEATURE_IS_VIRTUAL] = -1
                feature[MAP_FEATURE_SPEED_LIMIT] = -1
                feature[MAP_FEATURE_STOP_SIGN] = node.stop_sign
                feature[MAP_FEATURE_TIME_EMBED] = time_embed
                feature[MAP_FEATURE_HIDDEN_MASK] = hidden_mask
                lane_vector.append(feature)

                lane_len += 1
                if lane_len >= vector_limit:
                    break

            lane_vector = np.array(lane_vector, dtype=np.float32)
            lane_padding = np.zeros((vector_limit - lane_len, feature_len), dtype=np.float32)
            assert lane_vector.shape[0] == lane_len
            assert lane_vector.shape[1] == feature_len

            lane_vector_mask = np.ones(lane_len, dtype=np.int32)
            lane_padding_mask = np.zeros(vector_limit - lane_len, dtype=np.int32)

            lane_vector_pad = np.vstack((lane_padding, lane_vector))
            assert lane_vector_pad.shape[0] == vector_limit
            assert lane_vector_pad.shape[1] == feature_len
            lane_vector_mask_pad = np.concatenate((lane_padding_mask, lane_vector_mask))
            assert lane_vector_mask_pad.shape[0] == vector_limit

            lane_vector_list.append(lane_vector_pad)
            lane_vector_mask_list.append(lane_vector_mask_pad)

        return lane_vector_list, lane_vector_mask_list

    def process_road_boundary(self, sample, agent_id, valid_polyline_ids, vector_limit, feature_len, agent_origin,
                              agent_rotm):
        if (len(sample.map_polyline.road_boundarys) == 0) and (self.scene_id_to_map is not None and
                                                               sample.scene_id in self.scene_id_to_map):
            all_lanes = self.scene_id_to_map[sample.scene_id].road_boundarys
        else:
            all_lanes = sample.map_polyline.road_boundarys

        lane_vector_list = []
        lane_vector_mask_list = []
        for lane in all_lanes:
            vectors = lane.road_boundary_vec

            # TODO check length
            # lane_raw_len = len(vectors)

            # check in local map
            if valid_polyline_ids and not (lane.id in valid_polyline_ids):
                continue

            lane_len = 0
            lane_vector = []
            for idx, node in enumerate(vectors):
                time_embed = idx
                hidden_mask = 1.0  # 1.0: valid  0.0: hidden

                feature = [0] * MAP_FEATURE_SIZE
                feature[MAP_FEATURE_START_X] = node.vec.points[0].x
                feature[MAP_FEATURE_START_Y] = node.vec.points[0].y
                feature[MAP_FEATURE_END_X] = node.vec.points[1].x
                feature[MAP_FEATURE_END_Y] = node.vec.points[1].y
                feature[MAP_FEATURE_POLYLINE_TYPE] = node.vec.polyline_type
                feature[MAP_FEATURE_LANE_CENTER_TYPE] = -1
                feature[MAP_FEATURE_LANE_BOUNDARY_TYPE] = -1
                feature[MAP_FEATURE_ROAD_BOUNDARY_ROAD_TYPE] = node.road_type
                feature[MAP_FEATURE_ROAD_BOUNDARY_EDGE_TYPE] = node.boundary_edge_type
                feature[MAP_FEATURE_POLYGON_TYPE] = -1
                feature[MAP_FEATURE_VECTOR_HEADING] = np.arctan2(node.vec.points[1].y - node.vec.points[0].y,
                                                                 node.vec.points[1].x - node.vec.points[0].x)
                feature[MAP_FEATURE_IS_VIRTUAL] = -1
                feature[MAP_FEATURE_SPEED_LIMIT] = -1
                feature[MAP_FEATURE_STOP_SIGN] = -1
                feature[MAP_FEATURE_TIME_EMBED] = time_embed
                feature[MAP_FEATURE_HIDDEN_MASK] = hidden_mask
                lane_vector.append(feature)

                lane_len += 1
                if lane_len >= vector_limit:
                    break

            lane_vector = np.array(lane_vector, dtype=np.float32)
            lane_padding = np.zeros((vector_limit - lane_len, feature_len), dtype=np.float32)
            assert lane_vector.shape[0] == lane_len
            assert lane_vector.shape[1] == feature_len

            lane_vector_mask = np.ones(lane_len, dtype=np.int32)
            lane_padding_mask = np.zeros(vector_limit - lane_len, dtype=np.int32)

            lane_vector_pad = np.vstack((lane_padding, lane_vector))
            assert lane_vector_pad.shape[0] == vector_limit
            assert lane_vector_pad.shape[1] == feature_len
            lane_vector_mask_pad = np.concatenate((lane_padding_mask, lane_vector_mask))
            assert lane_vector_mask_pad.shape[0] == vector_limit

            lane_vector_list.append(lane_vector_pad)
            lane_vector_mask_list.append(lane_vector_mask_pad)

        return lane_vector_list, lane_vector_mask_list

    def process_polygon(self, sample, agent_id, valid_polyline_ids, vector_limit, feature_len, agent_origin,
                        agent_rotm):
        if (len(sample.map_polyline.map_polylines) == 0) and (self.scene_id_to_map is not None and
                                                              sample.scene_id in self.scene_id_to_map):
            all_lanes = self.scene_id_to_map[sample.scene_id].map_polylines
        else:
            all_lanes = sample.map_polyline.map_polylines

        lane_vector_list = []
        lane_vector_mask_list = []
        for lane in all_lanes:
            vectors = lane.polygon_vec

            # TODO check length
            # lane_raw_len = len(vectors)

            # check in local map
            if valid_polyline_ids and not (lane.id in valid_polyline_ids):
                continue

            lane_len = 0
            lane_vector = []
            for idx, node in enumerate(vectors):
                time_embed = idx
                hidden_mask = 1.0  # 1.0: valid  0.0: hidden

                feature = [0] * MAP_FEATURE_SIZE
                feature[MAP_FEATURE_START_X] = node.vec.points[0].x
                feature[MAP_FEATURE_START_Y] = node.vec.points[0].y
                feature[MAP_FEATURE_END_X] = node.vec.points[1].x
                feature[MAP_FEATURE_END_Y] = node.vec.points[1].y
                feature[MAP_FEATURE_POLYLINE_TYPE] = node.vec.polyline_type
                feature[MAP_FEATURE_LANE_CENTER_TYPE] = -1
                feature[MAP_FEATURE_LANE_BOUNDARY_TYPE] = -1
                feature[MAP_FEATURE_ROAD_BOUNDARY_ROAD_TYPE] = -1
                feature[MAP_FEATURE_ROAD_BOUNDARY_EDGE_TYPE] = -1
                feature[MAP_FEATURE_POLYGON_TYPE] = node.type
                feature[MAP_FEATURE_VECTOR_HEADING] = np.arctan2(node.vec.points[1].y - node.vec.points[0].y,
                                                                 node.vec.points[1].x - node.vec.points[0].x)
                feature[MAP_FEATURE_IS_VIRTUAL] = -1
                feature[MAP_FEATURE_SPEED_LIMIT] = -1
                feature[MAP_FEATURE_STOP_SIGN] = -1
                feature[MAP_FEATURE_TIME_EMBED] = time_embed
                feature[MAP_FEATURE_HIDDEN_MASK] = hidden_mask
                lane_vector.append(feature)

                lane_len += 1
                if lane_len >= vector_limit:
                    break

            lane_vector = np.array(lane_vector, dtype=np.float32)
            lane_padding = np.zeros((vector_limit - lane_len, feature_len), dtype=np.float32)
            assert lane_vector.shape[0] == lane_len
            assert lane_vector.shape[1] == feature_len

            lane_vector_mask = np.ones(lane_len, dtype=np.int32)
            lane_padding_mask = np.zeros(vector_limit - lane_len, dtype=np.int32)

            lane_vector_pad = np.vstack((lane_padding, lane_vector))
            assert lane_vector_pad.shape[0] == vector_limit
            assert lane_vector_pad.shape[1] == feature_len
            lane_vector_mask_pad = np.concatenate((lane_padding_mask, lane_vector_mask))
            assert lane_vector_mask_pad.shape[0] == vector_limit

            lane_vector_list.append(lane_vector_pad)
            lane_vector_mask_list.append(lane_vector_mask_pad)

        return lane_vector_list, lane_vector_mask_list

    def merge_polylines(self, polyline_vector_list, polyline_vector_mask_list, polyline_limit, vector_limit,
                        feature_len):
        '''
            polyline_limit: C_max
            vector_limit: L_max
            feature_len: F_max
            polyline_vector_list: list[ [L_max, F_max] ]
            polyline_vector_mask_list: list[ [L_max] ]
            return  merge_vectors [C_max, L_max, F_max], vector_mask:[C_max, L_max] cluster_mask: [C_max]
        '''
        if len(polyline_vector_list) > polyline_limit:
            # print(f'drop some polylines: {len(polyline_vector_list)-polyline_limit}')
            polyline_vector_list = polyline_vector_list[:polyline_limit]
            # TODO make sure the drop poyline is far away from agent
        polyline_vector_len = len(polyline_vector_list)
        if polyline_vector_len == 0:
            return np.zeros((polyline_limit, vector_limit, feature_len), dtype=np.float32),\
                   np.zeros((polyline_limit, vector_limit), dtype=np.float32),\
                   np.zeros((polyline_limit), dtype=np.float32)

        # [N_cluster, N_vector, N_feature]
        polyline_vector = np.stack(polyline_vector_list)
        assert polyline_vector.shape[0] == polyline_vector_len
        assert polyline_vector.shape[1] == vector_limit
        assert polyline_vector.shape[2] == feature_len
        # [N_padding, N_vector, N_feature]
        polyline_padding = np.zeros((polyline_limit - polyline_vector_len, vector_limit, feature_len), dtype=np.float32)
        # [N_cluster+N_padding, N_vector, N_feature]
        polyline_vector_pad = np.concatenate((polyline_vector, polyline_padding))

        if len(polyline_vector_mask_list) > polyline_limit:
            # print(f'drop some polylines: {len(polyline_vector_list)-polyline_limit}')
            polyline_vector_mask_list = polyline_vector_mask_list[:polyline_limit]
            # TODO make sure the drop polyline is far away from agent
        polyline_vector_mask_len = len(polyline_vector_mask_list)

        # [N_cluster, N_vector]
        polyline_vector_mask = np.stack(polyline_vector_mask_list)
        # [N_padding, N_vector]
        polyline_padding_mask = np.zeros((polyline_limit - polyline_vector_mask_len, vector_limit), dtype=np.int32)
        # [N_cluster+N_padding, N_vector]
        polyline_vector_mask_pad = np.concatenate((polyline_vector_mask, polyline_padding_mask))
        assert polyline_vector.shape[:-1] == polyline_vector_mask.shape
        assert polyline_padding.shape[:-1] == polyline_padding_mask.shape
        assert polyline_vector_pad.shape[:-1] == polyline_vector_mask_pad.shape
        assert polyline_vector_len == polyline_vector_mask_len

        # [N_cluster+N_padding]
        polyline_cluster_mask_pad = np.concatenate(
            (np.ones(polyline_vector_len, dtype=np.int32), np.zeros(polyline_limit - polyline_vector_len,
                                                                    dtype=np.int32)))
        return polyline_vector_pad, polyline_vector_mask_pad, polyline_cluster_mask_pad

    def process(self, sample, show=False):
        if sample is None:
            return None

        # F
        lane_feature_len = self.config['lane_feature_len']
        actor_feature_len = self.config['actor_feature_len']
        # L
        lane_vector_limit = self.config['lane_vector_limit']
        actor_history_limit = self.config['actor_history_limit']
        actor_future_limit = self.config['actor_future_limit']
        # C
        actor_limit = self.config['actor_limit']
        lane_center_polyline_limit = self.config['lane_center_polyline_limit']
        lane_boundary_polyline_limit = self.config['lane_boundary_polyline_limit']
        road_boundary_polyline_limit = self.config['road_boundary_polyline_limit']
        polygon_polyline_limit = self.config['polygon_polyline_limit']

        # extract actor and map feature
        agent_id = sample.agent_id
        agent_polyline_ids = self.get_agent_polyline_ids(sample, agent_id)

        (actor_vector_list, actor_vector_mask_list, actor_future_vector_list, actor_future_vector_mask_list, agent_pose,
         agent_origin, agent_rotm, agent_gt_traj_pad,
         agent_gt_traj_mask_pad) = self.process_obstacle(sample, agent_id, agent_polyline_ids['Obstacle'],
                                                         actor_history_limit, actor_feature_len, actor_future_limit)

        (actor_vector_pad, actor_vector_mask_pad,
         actor_cluster_mask_pad) = self.merge_polylines(actor_vector_list, actor_vector_mask_list, actor_limit,
                                                        actor_history_limit, actor_feature_len)

        (actor_future_vector_pad, actor_future_vector_mask_pad,
         actor_future_cluster_mask_pad) = self.merge_polylines(actor_future_vector_list, actor_future_vector_mask_list,
                                                               actor_limit, actor_future_limit, actor_feature_len)

        (lane_center_vector_list,
         lane_center_vector_mask_list) = self.process_lane_center(sample, agent_id, agent_polyline_ids['Lane'],
                                                                  lane_vector_limit, lane_feature_len, agent_origin,
                                                                  agent_rotm)

        (lane_center_vector_pad, lane_center_vector_mask_pad,
         lane_center_cluster_mask_pad) = self.merge_polylines(lane_center_vector_list, lane_center_vector_mask_list,
                                                              lane_center_polyline_limit, lane_vector_limit,
                                                              lane_feature_len)

        (lane_boundary_vector_list, lane_boundary_vector_mask_list) = self.process_lane_boundary(
            sample, agent_id, agent_polyline_ids['LaneBoundary'], lane_vector_limit, lane_feature_len, agent_origin,
            agent_rotm)

        (lane_boundary_vector_pad, lane_boundary_vector_mask_pad,
         lane_boundary_cluster_mask_pad) = self.merge_polylines(lane_boundary_vector_list,
                                                                lane_boundary_vector_mask_list,
                                                                lane_boundary_polyline_limit, lane_vector_limit,
                                                                lane_feature_len)

        (road_boundary_vector_list, road_boundary_vector_mask_list) = self.process_road_boundary(
            sample, agent_id, agent_polyline_ids['RoadBoundary'], lane_vector_limit, lane_feature_len, agent_origin,
            agent_rotm)

        (road_boundary_vector_pad, road_boundary_vector_mask_pad,
         road_boundary_cluster_mask_pad) = self.merge_polylines(road_boundary_vector_list,
                                                                road_boundary_vector_mask_list,
                                                                road_boundary_polyline_limit, lane_vector_limit,
                                                                lane_feature_len)

        (polygon_vector_list,
         polygon_vector_mask_list) = self.process_polygon(sample, agent_id, agent_polyline_ids['Polygon'],
                                                          lane_vector_limit, lane_feature_len, agent_origin, agent_rotm)
        (polygon_vector_pad, polygon_vector_mask_pad,
         polygon_cluster_mask_pad) = self.merge_polylines(polygon_vector_list, polygon_vector_mask_list,
                                                          polygon_polyline_limit, lane_vector_limit, lane_feature_len)

        # merge actor lane_center lane_boundary
        # [ [C1, L, F], [C2, L, F], ... ] -> [sum_C, L, F]
        map_vector = np.concatenate(
            (lane_center_vector_pad, lane_boundary_vector_pad, road_boundary_vector_pad, polygon_vector_pad))
        assert map_vector.shape == (lane_center_polyline_limit + lane_boundary_polyline_limit +
                                    road_boundary_polyline_limit + polygon_polyline_limit, lane_vector_limit,
                                    lane_feature_len)

        # [ [C1, L], [C2, L], ... ] -> [sum_C, L]
        map_vector_mask = np.concatenate((lane_center_vector_mask_pad, lane_boundary_vector_mask_pad,
                                          road_boundary_vector_mask_pad, polygon_vector_mask_pad))
        assert map_vector_mask.shape == (lane_center_polyline_limit + lane_boundary_polyline_limit +
                                         road_boundary_polyline_limit + polygon_polyline_limit, lane_vector_limit)

        # [ [C1], [C2], ... ] -> [sum_C]
        map_cluster_mask = np.concatenate((lane_center_cluster_mask_pad, lane_boundary_cluster_mask_pad,
                                           road_boundary_cluster_mask_pad, polygon_cluster_mask_pad))
        assert map_cluster_mask.shape == (lane_center_polyline_limit + lane_boundary_polyline_limit +
                                          road_boundary_polyline_limit + polygon_polyline_limit,)

        map_cluster_slice_num = [
            0, lane_center_polyline_limit, lane_boundary_polyline_limit, road_boundary_polyline_limit,
            polygon_polyline_limit
        ]
        map_cluster_slice_idx = np.array(map_cluster_slice_num, dtype=np.int32).cumsum()

        # ground truth traj
        gt_traj_pad = agent_gt_traj_pad  # [future_limit, 2] map cord
        gt_traj_mask_pad = agent_gt_traj_mask_pad  # [future_limit,]
        max_future_size = self.config[self.config['phase']]['max_future_size']
        if max_future_size > 0:
            assert len(gt_traj_pad) == max_future_size, f" error {len(gt_traj_pad)} == {max_future_size}"

        # handle cord
        if self.is_agent_cord:
            actor_vector_pad = transform_actor_featrue(actor_vector_pad, actor_vector_mask_pad, agent_origin,
                                                       agent_rotm)
            actor_future_vector_pad = transform_actor_featrue(actor_future_vector_pad, actor_future_vector_mask_pad,
                                                              agent_origin, agent_rotm)

            map_vector = transform_map_featrue(map_vector, map_vector_mask, agent_origin, agent_rotm)

            gt_traj_pad = transform_traj(gt_traj_pad, gt_traj_mask_pad, agent_origin, agent_rotm)

        # extract ground truth info
        grid_reso, grid_size = self.config['high_reso'], self.config['high_reso_dim']
        # genereate target points [3] 3s,5s,8s
        gt_target_point_mask = gt_traj_mask_pad[[29, 49, 79]]  # [3,]
        gt_target_point = gt_traj_pad[[29, 49, 79], :]  # [3, 2]
        # generate grid info
        gt_target_grid_idx, gt_target_grid_over_border = pos_to_index(gt_target_point, grid_reso,
                                                                      grid_size)  # [3,] [3,]
        gt_target_grid_center = index_to_pos(gt_target_grid_idx, grid_reso, grid_size)  # [3,2]
        # TODO(xlm) in case of over_grid_border the offset is wrong
        gt_target_grid_offset = gt_target_point - gt_target_grid_center  # [3,2]

        # generate candi target points
        # [d, d, 2]
        candi_target_points_x, candi_target_points_y, cord_xy_matrix = gen_grid_center_pos(grid_reso, grid_size)
        # [d, d, 2] -> [d*d, 2]
        candi_target_points = cord_xy_matrix.reshape(-1, 2)

        # transformed data
        data = AgentClosure()
        data.actor_vector = torch.from_numpy(actor_vector_pad).unsqueeze(0)  # [1, C, L, F]
        data.actor_vector_mask = torch.from_numpy(actor_vector_mask_pad).unsqueeze(0)  # [1, C, L]
        data.actor_cluster_mask = torch.from_numpy(actor_cluster_mask_pad).unsqueeze(0)  # [1, C]
        assert data.actor_vector.shape == (1, actor_limit, actor_history_limit, actor_feature_len)
        assert data.actor_vector_mask.shape == (1, actor_limit, actor_history_limit)
        assert data.actor_cluster_mask.shape == (1, actor_limit)

        data.actor_future_vector = torch.from_numpy(actor_future_vector_pad).unsqueeze(0)  # [1, C, L, F]
        data.actor_future_vector_mask = torch.from_numpy(actor_future_vector_mask_pad).unsqueeze(0)  # [1, C, L]
        data.actor_future_cluster_mask = torch.from_numpy(actor_future_cluster_mask_pad).unsqueeze(0)  # [1, C]
        assert data.actor_future_vector.shape == (1, actor_limit, actor_future_limit, actor_feature_len)
        assert data.actor_future_vector_mask.shape == (1, actor_limit, actor_future_limit)
        assert data.actor_future_cluster_mask.shape == (1, actor_limit)

        data.map_vector = torch.from_numpy(map_vector).unsqueeze(0)  # [1, C, L, F]
        data.map_vector_mask = torch.from_numpy(map_vector_mask).unsqueeze(0)  # [1, C, L]
        data.map_cluster_mask = torch.from_numpy(map_cluster_mask).unsqueeze(0)  # [1, C]
        data.map_cluster_slice_idx = torch.from_numpy(map_cluster_slice_idx).unsqueeze(0)  # [1, 5]
        map_limit = lane_center_polyline_limit + lane_boundary_polyline_limit + \
            road_boundary_polyline_limit + polygon_polyline_limit
        assert data.map_vector.shape == (1, map_limit, lane_vector_limit, lane_feature_len)
        assert data.map_vector_mask.shape == (1, map_limit, lane_vector_limit)
        assert data.map_cluster_mask.shape == (1, map_limit)
        assert data.map_cluster_slice_idx.shape == (1, 5)

        data.agent_pose = torch.from_numpy(agent_pose).unsqueeze(0)  # [1, 4]
        assert data.agent_pose.shape == (1, 4)

        data.gt_traj = torch.from_numpy(gt_traj_pad).unsqueeze(0)  # [1, K, 2]
        data.gt_traj_mask = torch.from_numpy(gt_traj_mask_pad).unsqueeze(0)  # [1, K]
        assert data.gt_traj.shape == (1, actor_future_limit, 2)
        assert data.gt_traj_mask.shape == (1, actor_future_limit)

        data.gt_target_point_mask = torch.from_numpy(gt_target_point_mask).unsqueeze(0)  # [1, 3]
        data.gt_target_point = torch.from_numpy(gt_target_point).unsqueeze(0)  # [1, 3, 2]
        data.gt_target_grid_idx = torch.from_numpy(gt_target_grid_idx).unsqueeze(0)  # [1, 3]
        data.gt_target_grid_over_border = torch.from_numpy(gt_target_grid_over_border).unsqueeze(0)  # [1, 3]
        data.gt_target_grid_center = torch.from_numpy(gt_target_grid_center).unsqueeze(0)  # [1, 3, 2]
        # TODO(xlm) in case of over_grid_border the offset is wrong
        data.gt_target_grid_offset = torch.from_numpy(gt_target_grid_offset).unsqueeze(0)  # [1, 3, 2]
        assert data.gt_target_point_mask.shape == (1, 3)
        assert data.gt_target_point.shape == (1, 3, 2)
        assert data.gt_target_grid_idx.shape == (1, 3)
        assert data.gt_target_grid_over_border.shape == (1, 3)
        assert data.gt_target_grid_center.shape == (1, 3, 2)
        assert data.gt_target_grid_offset.shape == (1, 3, 2)

        data.candi_target_points = torch.from_numpy(candi_target_points).unsqueeze(0)  # [1, d*d, 2]
        data.candi_target_points_x = candi_target_points_x  # [d]
        data.candi_target_points_y = candi_target_points_y  # [d]
        assert data.candi_target_points.shape == (1, grid_size * grid_size, 2)
        assert data.candi_target_points_x.shape == (grid_size,)
        assert data.candi_target_points_y.shape == (grid_size,)

        # for debug
        data.raw_sample_name = sample.scene_id + "_" + str(sample.seq_num) + "_" + str(agent_id)
        data.high_reso = self.config['high_reso']
        data.high_reso_dim = self.config['high_reso_dim']
        data.lane_feature_len = lane_feature_len
        data.actor_feature_len = actor_feature_len
        data.actor_history_limit = actor_history_limit
        data.actor_future_limit = actor_future_limit
        data.lane_vector_limit = lane_vector_limit
        data.actor_limit = actor_limit
        data.lane_center_polyline_limit = lane_center_polyline_limit
        data.lane_boundary_polyline_limit = lane_boundary_polyline_limit
        data.road_boundary_polyline_limit = road_boundary_polyline_limit
        data.polygon_polyline_limit = polygon_polyline_limit

        if show:
            plot_transformed_data(data, True)
        return data


def plot_transformed_data(data, show=True, save_path=None, is_agent_cord=True):
    plt.figure(data.raw_sample_name)
    plt.axis('equal')

    agent_pose = data.agent_pose[0]
    agent_origin = agent_pose[0:2]
    agent_rot = agent_pose[2:4]
    agent_rotm = torch.FloatTensor([[agent_rot[0], -agent_rot[1]], [agent_rot[1], agent_rot[0]]])
    # agent_rotm = torch.DoubleTensor([[agent_rot[0], -agent_rot[1]], [agent_rot[1], agent_rot[0]]])

    # heatmap
    if hasattr(data, 'pred_heatmap'):
        dim = data.high_reso_dim
        # reso = data.high_reso
        heatmap = data.pred_heatmap.view(dim, dim)
        # x, y, _ = gen_grid_center_pos(reso, dim)
        # Wistia, YlOrBr, Oranges, YlOrBr, YlOrRd, YlGnBu, YlGn, hot
        # ref: https://matplotlib.org/stable/tutorials/colors/colormaps.html
        plt.pcolormesh(data.candi_target_points_x,
                       data.candi_target_points_y,
                       heatmap,
                       cmap='Wistia',
                       alpha=1.0,
                       shading='auto')
    else:
        if hasattr(data, 'candi_target_points_x'):
            dim = data.high_reso_dim
            # reso = data.high_reso
            heatmap = torch.zeros(dim * dim)
            if hasattr(data, 'gt_target_grid_idx'):
                for i in range(data.gt_target_grid_idx[0].size()[0]):
                    if data.gt_target_point_mask[0][i] > 0:
                        heatmap[data.gt_target_grid_idx[0][i]] = 0.4 + i * 0.3
            heatmap = heatmap.view(dim, dim)
            # x, y, _ = gen_grid_center_pos(reso, dim)
            # Wistia, YlOrBr, Oranges, YlOrBr, YlOrRd, YlGnBu, YlGn, hot
            # ref: https://matplotlib.org/stable/tutorials/colors/colormaps.html
            plt.pcolormesh(data.candi_target_points_x,
                           data.candi_target_points_y,
                           heatmap,
                           cmap='Wistia',
                           alpha=1.0,
                           shading='auto')
        if hasattr(data, 'gt_target_grid_center'):
            for i in range(data.gt_target_grid_center[0].size()[0]):
                if data.gt_target_point_mask[0][i] > 0:
                    plt.plot([data.gt_target_grid_center[0][i, 0], data.gt_target_point[0][i, 0]],
                             [data.gt_target_grid_center[0][i, 1], data.gt_target_point[0][i, 1]],
                             "y:",
                             linewidth=1)

    # plot lane center
    lane_center_start_idx = data.map_cluster_slice_idx[0, 0]
    lane_center_end_idx = data.map_cluster_slice_idx[0, 1]
    lane_center_vector = data.map_vector[0, lane_center_start_idx:lane_center_end_idx, ...]
    lane_center_vector_mask = data.map_vector_mask[0, lane_center_start_idx:lane_center_end_idx, ...]
    lane_center_cluster_mask = data.map_cluster_mask[0, lane_center_start_idx:lane_center_end_idx, ...]
    # vector is [batch_size, agent|obstacle|zeros, zeros|normal, other|tag]
    assert lane_center_vector.size() == (data.lane_center_polyline_limit, data.lane_vector_limit, data.lane_feature_len)
    # vector_mask is [batch_size, ones|zeros, (zeros|ones, zeros|zeros)]
    assert lane_center_vector_mask.size() == (data.lane_center_polyline_limit, data.lane_vector_limit)
    # cluseter_mask is [batch_size, ones|zeros]
    assert lane_center_cluster_mask.size() == (data.lane_center_polyline_limit,)
    for i in range(0, data.lane_center_polyline_limit):
        lane_vector = lane_center_vector[i]  # [zeros|normal, other|tag]
        lane_vector_mask = lane_center_vector_mask[i]  # [zeros|ones] or [zeros|zeros]
        lane_vector_cluseter_mask = lane_center_cluster_mask[i]  # one or zero

        # if lane_vector[-1, -1] == 0.0:
        if lane_vector_cluseter_mask == 0:
            continue

        # st_idx = torch.where(lane_vector[:, -1] != 0.0)[0][0]
        st_idx = torch.where(lane_vector_mask != 0)[0][0]
        lane_vector = lane_vector[st_idx:, :-1]

        if is_agent_cord:
            lane_vector_start = lane_vector[:, [0, 1]]
            lane_vector_end = lane_vector[:, [2, 3]]
        else:
            lane_vector_start = torch.matmul(lane_vector[:, [0, 1]] - agent_origin, agent_rotm)
            lane_vector_end = torch.matmul(lane_vector[:, [2, 3]] - agent_origin, agent_rotm)
        plot_vector(lane_vector_start, lane_vector_end, 'gray', 0.3, 0.5, 1.5, 0.7)

    # plot lane boundary
    lane_boundary_start_idx = data.map_cluster_slice_idx[0, 1]
    lane_boundary_end_idx = data.map_cluster_slice_idx[0, 2]
    lane_boundary_vector = data.map_vector[0, lane_boundary_start_idx:lane_boundary_end_idx, ...]
    lane_boundary_vector_mask = data.map_vector_mask[0, lane_boundary_start_idx:lane_boundary_end_idx, ...]
    lane_boundary_cluster_mask = data.map_cluster_mask[0, lane_boundary_start_idx:lane_boundary_end_idx, ...]
    # vector is [batch_size, agent|obstacle|zeros, zeros|normal, other|tag]
    assert lane_boundary_vector.size() == (data.lane_boundary_polyline_limit, data.lane_vector_limit,
                                           data.lane_feature_len)
    # vector_mask is [batch_size, ones|zeros, (zeros|ones, zeros|zeros)]
    assert lane_boundary_vector_mask.size() == (data.lane_boundary_polyline_limit, data.lane_vector_limit)
    # cluseter_mask is [batch_size, ones|zeros]
    assert lane_boundary_cluster_mask.size() == (data.lane_boundary_polyline_limit,)
    for i in range(0, data.lane_boundary_polyline_limit):
        lane_vector = lane_boundary_vector[i]  # [zeros|normal, other|tag]
        lane_vector_mask = lane_boundary_vector_mask[i]  # [zeros|ones] or [zeros|zeros]
        lane_vector_cluseter_mask = lane_boundary_cluster_mask[i]  # one or zero

        # if lane_vector[-1, -1] == 0.0:
        if lane_vector_cluseter_mask == 0:
            continue

        # st_idx = torch.where(lane_vector[:, -1] != 0.0)[0][0]
        st_idx = torch.where(lane_vector_mask != 0)[0][0]
        lane_vector = lane_vector[st_idx:, :-1]

        if is_agent_cord:
            lane_vector_start = lane_vector[:, [0, 1]]
            lane_vector_end = lane_vector[:, [2, 3]]
        else:
            lane_vector_start = torch.matmul(lane_vector[:, [0, 1]] - agent_origin, agent_rotm)
            lane_vector_end = torch.matmul(lane_vector[:, [2, 3]] - agent_origin, agent_rotm)
        plot_vector(lane_vector_start, lane_vector_end, 'gray', 0.1, 0.5, 1.5, 0.7)

    # plot road boundary
    road_boundary_start_idx = data.map_cluster_slice_idx[0, 2]
    road_boundary_end_idx = data.map_cluster_slice_idx[0, 3]
    road_boundary_vector = data.map_vector[0, road_boundary_start_idx:road_boundary_end_idx, ...]
    road_boundary_vector_mask = data.map_vector_mask[0, road_boundary_start_idx:road_boundary_end_idx, ...]
    road_boundary_cluster_mask = data.map_cluster_mask[0, road_boundary_start_idx:road_boundary_end_idx, ...]
    # vector is [batch_size, agent|obstacle|zeros, zeros|normal, other|tag]
    assert road_boundary_vector.size() == (data.road_boundary_polyline_limit, data.lane_vector_limit,
                                           data.lane_feature_len)
    # vector_mask is [batch_size, ones|zeros, (zeros|ones, zeros|zeros)]
    assert road_boundary_vector_mask.size() == (data.road_boundary_polyline_limit, data.lane_vector_limit)
    # cluseter_mask is [batch_size, ones|zeros]
    assert road_boundary_cluster_mask.size() == (data.road_boundary_polyline_limit,)
    for i in range(0, data.road_boundary_polyline_limit):
        lane_vector = road_boundary_vector[i]  # [zeros|normal, other|tag]
        lane_vector_mask = road_boundary_vector_mask[i]  # [zeros|ones] or [zeros|zeros]
        lane_vector_cluseter_mask = road_boundary_cluster_mask[i]  # one or zero

        # if lane_vector[-1, -1] == 0.0:
        if lane_vector_cluseter_mask == 0:
            continue

        # st_idx = torch.where(lane_vector[:, -1] != 0.0)[0][0]
        st_idx = torch.where(lane_vector_mask != 0)[0][0]
        lane_vector = lane_vector[st_idx:, :-1]

        if is_agent_cord:
            lane_vector_start = lane_vector[:, [0, 1]]
            lane_vector_end = lane_vector[:, [2, 3]]
        else:
            lane_vector_start = torch.matmul(lane_vector[:, [0, 1]] - agent_origin, agent_rotm)
            lane_vector_end = torch.matmul(lane_vector[:, [2, 3]] - agent_origin, agent_rotm)
        plot_vector(lane_vector_start, lane_vector_end, 'black', 0.1, 1.0, 1.5, 0.7)

    # plot polygon
    polygon_start_idx = data.map_cluster_slice_idx[0, 3]
    polygon_end_idx = data.map_cluster_slice_idx[0, 4]
    polygon_vector = data.map_vector[0, polygon_start_idx:polygon_end_idx, ...]
    polygon_vector_mask = data.map_vector_mask[0, polygon_start_idx:polygon_end_idx, ...]
    polygon_cluster_mask = data.map_cluster_mask[0, polygon_start_idx:polygon_end_idx, ...]
    # vector is [batch_size, agent|obstacle|zeros, zeros|normal, other|tag]
    assert polygon_vector.size() == (data.polygon_polyline_limit, data.lane_vector_limit, data.lane_feature_len)
    # vector_mask is [batch_size, ones|zeros, (zeros|ones, zeros|zeros)]
    assert polygon_vector_mask.size() == (data.polygon_polyline_limit, data.lane_vector_limit)
    # cluseter_mask is [batch_size, ones|zeros]
    assert polygon_cluster_mask.size() == (data.polygon_polyline_limit,)
    for i in range(0, data.polygon_polyline_limit):
        lane_vector = polygon_vector[i]  # [zeros|normal, other|tag]
        lane_vector_mask = polygon_vector_mask[i]  # [zeros|ones] or [zeros|zeros]
        lane_cluseter_mask = polygon_cluster_mask[i]  # one or zero

        # if lane_vector[-1, -1] == 0.0:
        if lane_cluseter_mask == 0:
            continue

        # st_idx = torch.where(lane_vector[:, -1] != 0.0)[0][0]
        st_idx = torch.where(lane_vector_mask != 0)[0][0]
        lane_vector = lane_vector[st_idx:, :-1]

        if is_agent_cord:
            lane_vector_start = lane_vector[:, [0, 1]]
            lane_vector_end = lane_vector[:, [2, 3]]
        else:
            lane_vector_start = torch.matmul(lane_vector[:, [0, 1]] - agent_origin, agent_rotm)
            lane_vector_end = torch.matmul(lane_vector[:, [2, 3]] - agent_origin, agent_rotm)
        plot_vector(lane_vector_start, lane_vector_end, 'gray', 0.1, 1.0, 1.5, 0.7)

    # plot actor
    actor_vector = data.actor_vector[0]
    actor_vector_mask = data.actor_vector_mask[0]
    actor_cluster_mask = data.actor_cluster_mask[0]
    # vector is [batch_size, agent|obstacle|zero, zero|normal, other|tag]
    assert actor_vector.size() == (data.actor_limit, data.actor_history_limit, data.actor_feature_len)
    # vector_mask is [batch_size, ones|zeros, (zeros|ones, zeros|zeros)]
    assert actor_vector_mask.size() == (data.actor_limit, data.actor_history_limit)
    # cluseter_mask is [batch_size, ones|zeros]
    assert actor_cluster_mask.size() == (data.actor_limit,)
    for i in range(0, data.actor_limit):
        lane_vector = actor_vector[i]  # [zeros|normal, other|tag]
        lane_vector_mask = actor_vector_mask[i]  # [zeros|ones] or [zeros|zeros]
        lane_cluseter_mask = actor_cluster_mask[i]  # one or zero

        # if actor_vector[-1, -1] == 0.0 and i > 0:
        if lane_cluseter_mask == 0 and i > 0:
            continue

        # if i > 0:
        #     st_idx = torch.where(actor_vector[:, -1] != 0.0)[0][0]
        # else:
        #     st_idx = 0  # agent is full?
        st_idx = torch.where(lane_vector_mask != 0)[0][0]
        lane_vector = lane_vector[st_idx:, :-1]

        if is_agent_cord:
            lane_vector_start = lane_vector[:, [0, 1]]
            lane_vector_end = lane_vector[:, [2, 3]]
        else:
            lane_vector_start = torch.matmul(lane_vector[:, [0, 1]] - agent_origin, agent_rotm)
            lane_vector_end = torch.matmul(lane_vector[:, [2, 3]] - agent_origin, agent_rotm)
        color = 'red' if i == 0 else 'blue'
        plot_vector(lane_vector_start, lane_vector_end, color, 0.1, 1.0, 1.5, 0.7)

    # plot future actor
    if hasattr(data, 'actor_future_vector'):
        actor_future_vector = data.actor_future_vector[0]
        actor_future_vector_mask = data.actor_future_vector_mask[0]
        actor_future_cluster_mask = data.actor_future_cluster_mask[0]
        # vector is [batch_size, agent|obstacle|zero, zero|normal, other|tag]
        assert actor_future_vector.size() == (data.actor_limit, data.actor_future_limit, data.actor_feature_len)
        # vector_mask is [batch_size, ones|zeros, (zeros|ones, zeros|zeros)]
        assert actor_future_vector_mask.size() == (data.actor_limit, data.actor_future_limit)
        # cluseter_mask is [batch_size, ones|zeros]
        assert actor_future_cluster_mask.size() == (data.actor_limit,)
        for i in range(0, data.actor_limit):
            lane_vector = actor_future_vector[i]  # [zeros|normal, other|tag]
            lane_vector_mask = actor_future_vector_mask[i]  # history:[zeros|ones] future:[ones|zeors] or [zeros|zeros]
            lane_cluseter_mask = actor_future_cluster_mask[i]  # one or zero

            # if actor_vector[-1, -1] == 0.0 and i > 0:
            if lane_cluseter_mask == 0 and i > 0:
                continue

            # if i > 0:
            #     st_idx = torch.where(actor_vector[:, -1] != 0.0)[0][0]
            # else:
            #     st_idx = 0  # agent is full?
            # st_idx = torch.where(lane_vector_mask != 0)[0][0]
            # lane_vector = lane_vector[st_idx:, :-1]
            lane_vector = lane_vector[lane_vector_mask == 1]

            if is_agent_cord:
                lane_vector_start = lane_vector[:, [0, 1]]
                lane_vector_end = lane_vector[:, [2, 3]]
            else:
                lane_vector_start = torch.matmul(lane_vector[:, [0, 1]] - agent_origin, agent_rotm)
                lane_vector_end = torch.matmul(lane_vector[:, [2, 3]] - agent_origin, agent_rotm)
            color = 'red' if i == 0 else 'blue'
            plot_vector(lane_vector_start, lane_vector_end, color, 0.2, 0.2, 1.5, 0.7)

    # plot pred traj [K, T, 2]
    if hasattr(data, 'pred_traj'):
        for k, pred_traj in enumerate(data.pred_traj):
            # [T, 2]
            # traj = torch.cat([torch.FloatTensor([[0, 0]]), pred_traj], dim=0)
            traj = pred_traj
            # traj_mask = torch.sum(pred_traj, dim=-1, keepdim=False) != 0
            # traj = traj[traj_mask]
            if len(traj) >= 2:
                alpha = 0.7
                if hasattr(data, 'pred_traj_score'):
                    alpha = data.pred_traj_score[k].item()
                plot_vector(traj[:-1], traj[1:], 'c', 0.1, alpha, 1.5, 0.7)
                plt.text(x=traj[-1][0], y=traj[-1][1], s=str(alpha))

            if hasattr(data, 'pred_traj_gt') and hasattr(data, 'pred_traj_gt_mask'):
                gt_traj = data.pred_traj_gt[data.pred_traj_gt_mask == 1]
                plot_vector(gt_traj[:-1], gt_traj[1:], 'green', 0.1, 1.0, 1.5, 0.7)

    # plot gt
    if hasattr(data, 'gt_traj'):
        # gt_traj = torch.cat([torch.FloatTensor([[0, 0]]), data.gt_traj[0]], dim=0)
        gt_traj = data.gt_traj[0]
        gt_traj = gt_traj[data.gt_traj_mask[0] == 1]
        # gt_traj = torch.cat([torch.FloatTensor([[0, 0]]), gt_traj], dim=0)
        plot_vector(gt_traj[:-1], gt_traj[1:], 'green', 0.1, 1.0, 1.5, 0.7)

    if show:
        figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        figManager.resize(*figManager.window.maxsize())
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    if save_path is not None:
        save_folder = os.path.dirname(save_path)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(save_path)

    return plt


if __name__ == '__main__':
    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print(f'{config_file} not exists!!')
    config_dir = os.path.dirname(config_file)

    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=config_dir)
    with open(config_file, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    sample_list_file = None
    if 'sample_list_file' in config and len(config['sample_list_file']) > 0:
        sample_list_file = config['sample_list_file']
    config = config['processer']

    dataset = SampleDataset(config['sample_database_folder'], 'lmdb', sample_list_file)
    print(f'Dataset sample number: {len(dataset)}')

    feature_processer = FeatureProcesser(config)
    for i in tqdm(range(len(dataset))):
        space_process_samples = feature_processer.process(dataset[i], True)
        # print(f'{i}-{dataset[i].scene_id}-{dataset[i].seq_num}')
