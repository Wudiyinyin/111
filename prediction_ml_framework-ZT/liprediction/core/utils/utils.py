# Copyright (c) 2021 Li Auto Company. All rights reserved.
import math

import numpy as np
import torch


def pose2rt(agent_pose):
    '''
        agent_pose: [B, 4]
        return: [B, 2], [B, 2, 2]
    '''
    agent_origin = agent_pose[:, [0, 1]]
    agent_rot = agent_pose[:, [2, 3]]
    agent_rotm = torch.stack([
        torch.stack((agent_rot[:, 0], -agent_rot[:, 1]), dim=-1),
        torch.stack((agent_rot[:, 1], agent_rot[:, 0]), dim=-1)
    ],
                             dim=1)
    # agent_origin.shape: [B, 2]
    # agent_rotm.shape: [B, 2, 2]
    return agent_origin, agent_rotm


def local2global(data, origin, rotm):
    '''
        transform to new coordinate
        data: [B, ..., 2], 2 is x, y
        origin: [B, 2]
        rotm: [B, 2, 2]
        return: [B, ..., 2]
    '''

    def multi_unsqueeze(data, dim, times):
        assert times >= 1  # times < 1 is no need to unsqueeze
        for _ in range(times):
            data = data.unsqueeze(dim)

        return data

    # Not support data dim <= 1
    assert data.dim() >= 2
    # if data dim <= 2, then no need to unsqueeze
    if data.dim() > 2:
        # [B, .., 2]
        origin = multi_unsqueeze(origin, dim=1, times=data.dim() - 2)
        # col_vector matmul to row_vector matmul need ^T
        # [B, 2, 2]=[B, R] -> [B, 2, 2]=[B, R^T] ->  [B, .., 2, 2]
        rotm = multi_unsqueeze(rotm, dim=1, times=data.dim() - 2)

    # data : [B, .., 2] -> [B, .., 1, 2]
    # rotm : R -> R.T
    # [B, .., 1, 2] * [B, .., 2, 2] = [B, .., 1, 2] -> [B, .., 2]
    # [B, .., 2] + [B, .., 2]
    data = torch.matmul(data.unsqueeze(-2), rotm.transpose(-1, -2)).squeeze(-2) + origin
    # [B, .., 2]
    return data


def global2local(vector, origin, rotm):
    '''
        vector: [B, C, L, 2]
        origin: [B, 2]
        rotm: [B, 2, 2]
        return: [B, C, L, 2]
    '''
    # [B, 1, 1, 2]
    origin = origin.unsqueeze(1).unsqueeze(1)
    # [B, 1, 2, 2]
    rotm = rotm.unsqueeze(1)
    # ([B, Cluter?, Vector?, 2] - [B, 1, 1, 2] -> [B, C, L, 2]
    # [B, C, L, 2] * [B, 1, 2, 2] -> [B, C, L, 2]
    # 1) global->local: rotm^T  2) col_vector->row_vector: rotm^T^T = rotm
    vector = torch.matmul(vector - origin, rotm)
    # 1) R_map_agent^T * R_map_other = R_agent_other         2) col_vector -> row_vector
    # [a -b] ^T     * [a' -b']  => col_vector = [a -b]^T * [a'] => row_vector [a' b'] * [a -b]
    # [b  a]          [b'  a']                  [b  a]     [b']                         [b  a]
    return vector


def global2local_array(vectors, pose):
    theta = np.pi / 2.0 - pose[-1]
    new_vectors = vectors - pose[:2]
    t = [math.cos(theta), math.sin(theta), -math.sin(theta), math.cos(theta)]
    t = np.reshape(t, (2, 2))
    new_vectors = np.matmul(new_vectors, t)
    return new_vectors


# ref: https://stackoverflow.com/questions/52804046/why-is-np-linalg-normx-2-slower-than-solving-it-directly
def np_vector_norm(x, axis=None):
    return math.sqrt(np.sum(x * x, axis=axis))


def rotm_from_vect(v0, v1):
    rotv = v1 - v0
    norm = np_vector_norm(rotv)

    if norm < 1e-6:
        cos_a, sin_a = 1.0, 0.0
    else:
        cos_a, sin_a = rotv / norm

    rotm = np.array([[cos_a, -sin_a], [sin_a, +cos_a]], dtype=np.float32)

    return rotm


# ref https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap/15927914
def normal_angle(input_angle):
    return np.fmod((input_angle + np.pi), 2 * np.pi) - np.pi
