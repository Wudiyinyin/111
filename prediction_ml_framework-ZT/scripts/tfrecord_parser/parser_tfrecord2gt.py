# Copyright (c) 2021 Li Auto Company. All rights reserved.

import argparse
import glob
import os
import pickle as pkl

import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from waymo_open_dataset.protos import scenario_pb2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--tfrecord_data_dir", type=str, required=True, help="TFRecord data dir for converter")
    parser.add_argument("--output_pkl_file", type=str, required=True, help="Dump the muliple streaming pb files")

    args = parser.parse_args()
    return args


def get_gt(track):
    assert len(track.states) == 91
    gt_info = np.zeros((91, 7))
    gt_mask = np.zeros((91))
    gt_type = np.zeros((91))
    for i, state in enumerate(track.states):
        gt_info[i] = np.array([
            state.center_x, state.center_y, state.length, state.width, state.heading, state.velocity_x, state.velocity_y
        ])
        gt_mask[i] = state.valid
        gt_type[i] = track.object_type

    return gt_info, gt_mask, gt_type


def process_one(file_name):
    print(file_name)
    info_list = []
    raw_dataset = tf.data.TFRecordDataset([file_name])
    for raw_record in raw_dataset:
        proto_string = raw_record.numpy()
        proto = scenario_pb2.Scenario()
        proto.ParseFromString(proto_string)
        scene_id = proto.scenario_id
        pred_obs_id = []
        gt_info = {}
        for index in proto.tracks_to_predict:
            obs_id = proto.tracks[index.track_index].id
            pred_obs_id.append(obs_id)
            track_index = index.track_index
            cur_track = proto.tracks[track_index]
            gt_state, gt_mask, gt_type = get_gt(cur_track)
            gt_info[obs_id] = [gt_state, gt_mask, gt_type]
        adc_id = proto.tracks[proto.sdc_track_index].id
        file_name = os.path.basename(file_name)
        info_dict = {'scene_id': scene_id, 'file_name': file_name, 'pred_obs': pred_obs_id, 'adc_id': adc_id}
        info_dict.update({'gt_info': gt_info})
        info_list.append(info_dict)
    return info_list


if __name__ == "__main__":
    args: argparse.ArgumentParser = parse_args()

    file_list = glob.glob(args.tfrecord_data_dir + '/*tfrecord*')
    file_info_list = Parallel(n_jobs=64)(delayed(process_one)(input_database_path) for input_database_path in file_list)

    scene_id_file_dict = {}
    for scene_list in file_info_list:
        for scene_info in scene_list:
            scene_id = scene_info['scene_id']
            scene_id_file_dict[scene_id] = scene_info
    with open(args.output_pkl_file, 'wb') as f:
        pkl.dump(scene_id_file_dict, f)
