import argparse
import glob
import os

import numpy as np
import torch
import tqdm
import yaml
from core.utils.utils import global2local_array, normal_angle, np_vector_norm
from datasets.database.sample_database import PickleDatabase
from datasets.fsd_processor.data_parser import Parser
from datasets.transform.agent_closure import AgentClosure
from joblib import Parallel, delayed


class SampleSaver():

    def __init__(self, output_database_folder):
        self.output_database = PickleDatabase(output_database_folder, write=True, map_size=80 * 1024 * 1024 * 1024)

    def __del__(self):
        del self.output_database

    def save_sample(self, sample_transformed):
        self.output_database.put(sample_transformed.raw_sample_name, sample_transformed)


class SampleGenerator():

    def __init__(self, config):
        self.config = config['fsd_processor']
        self.min_his_num = self.config['min_history_num']
        self.pred_time_length = self.config['pred_time_length']
        self.frame_time_reso = self.config['frame_time_reso']
        self.pred_step_num = int(self.pred_time_length / self.frame_time_reso)
        self.output_frame_num = self.config['output_frame_num']
        assert self.pred_step_num % self.output_frame_num == 0
        self.output_frame_reso = self.pred_step_num // self.output_frame_num
        self.adc_id = self.config['adc_id']

        self.forward_dist = self.config['forward_dist']
        self.back_dist = self.config['back_dist']
        self.left_dist = self.config['left_dist']
        self.right_dist = self.config['right_dist']
        self.max_lane_num = self.config['lane_polyline_num']
        self.max_bound_num = self.config['bound_polyline_num']
        self.max_obstacle_num = self.config['obstacle_polyline_num']

        self.lane_max_vector_num = self.config['default_lane_vector_num']
        self.lane_vector_dim = self.config['lane_vector_dim']
        self.obs_max_vector_num = self.config['default_history_num']
        self.obs_vector_dim = self.config['obs_vector_dim']

    def generate_sample(self, seq_num_obs_id_dict, obs_id_track_dict, seq_num_map_dict, curr_seq_num_dict, saver):
        seq_num_pred_id_dict = self.select_obstacle(seq_num_obs_id_dict, obs_id_track_dict)
        sample_list = self.feature_transform(seq_num_pred_id_dict, seq_num_obs_id_dict, obs_id_track_dict,
                                             seq_num_map_dict, curr_seq_num_dict)
        for sample in sample_list:
            saver.save_sample(sample)

    def select_obstacle(self, seq_num_obs_id_dict, obs_id_track_dict):
        all_seq_num = list(seq_num_obs_id_dict.keys())
        min_seq_nun = min(all_seq_num)
        max_seq_num = max(all_seq_num)

        seq_num_pred_id_dict = {}
        for seq_num in range(min_seq_nun, max_seq_num + 1):
            if seq_num not in seq_num_obs_id_dict:
                continue
            cur_frame_obs_id_list = seq_num_obs_id_dict[seq_num]
            assert self.adc_id in cur_frame_obs_id_list
            pred_id_list = []
            for obs_id in cur_frame_obs_id_list:
                obs_his_fea_array = obs_id_track_dict[obs_id][seq_num][0]
                valid_his_num = np.sum(obs_his_fea_array[:, -1])
                if valid_his_num < self.min_his_num:
                    continue
                is_contain_all_future = True
                for future_seq_num in range(seq_num, seq_num + self.pred_step_num + self.output_frame_reso,
                                            self.output_frame_reso):
                    if future_seq_num not in obs_id_track_dict[obs_id]:
                        is_contain_all_future = False
                        break
                if not is_contain_all_future:
                    continue
                pred_id_list.append(obs_id)
            seq_num_pred_id_dict[seq_num] = pred_id_list

        return seq_num_pred_id_dict

    def get_agent_array_list(self, seq_num_obs_id_dict, obs_id_track_dict, seq_num_map_dict, curr_seq_num_dict, obs_id,
                             seq_num):
        map_seq_num = curr_seq_num_dict[seq_num]
        obs_array_list = [obs_id_track_dict[obs_id][seq_num][0]]
        lane_array_list = []
        bound_array_list = []
        if obs_id_track_dict[obs_id][seq_num][1] is not None:
            polyline_info_list = obs_id_track_dict[obs_id][seq_num][1]
            for i, polylne_info in enumerate(polyline_info_list):
                polyline_type = polylne_info.type
                polyline_id = polylne_info.polyline_id
                if i == 0:
                    assert int(polyline_id) == self.adc_id
                if polyline_type == polylne_info.Obstacle:
                    obs_id = int(polyline_id)
                    obs_array_list.append(obs_id_track_dict[obs_id][seq_num][0])
                elif polyline_type == polylne_info.Lane:
                    polyline_array = seq_num_map_dict[map_seq_num][0][polyline_id]
                    lane_array_list.append(polyline_array)
                elif polyline_type == polylne_info.LaneBoundary:
                    polyline_array = seq_num_map_dict[map_seq_num][1][polyline_id]
                    bound_array_list.append(polyline_array)
                else:
                    assert False, polyline_type
        else:
            pass
        return obs_array_list, lane_array_list, bound_array_list
        # todo select polyline 高速polyline筛选不应该只看距离，应该看nearby关系
        '''
        pose = obs_id_track_dict[obs_id][seq_num][2]
        lane_polyline_dict = seq_num_map_dict[map_seq_num][0]
        bound_polyline_dict = seq_num_map_dict[map_seq_num][1]

        select_lane_polyline_list = []
        for polyline_id, lane_polyline in lane_polyline_dict.items():
            valid_polyline = lane_polyline[lane_polyline[:, -1] == 1]
            local_vectors = global2local_array(valid_polyline[:, :2], pose)
            for v in local_vectors:
                if self.left_dist <= v[0] <= self.right_dist and self.back_dist <= v[1] <= self.forward_dist:
                    min_dist = np.min(np_vector_norm(local_vectors), axis=0)
                    select_lane_polyline_list.append([lane_polyline, min_dist])
                    break
        select_bound_polyline_list = []
        for polyline_id, bound_polyline in bound_polyline_dict.items():
            valid_polyline = bound_polyline[bound_polyline[:, -1] == 1]
            local_vectors = global2local_array(valid_polyline[:, :2], pose)
            for v in local_vectors:
                if self.left_dist <= v[0] <= self.right_dist and self.back_dist <= v[1] <= self.forward_dist:
                    min_dist = np.min(np_vector_norm(local_vectors), axis=0)
                    select_bound_polyline_list.append(bound_polyline, min_dist)
                    break
        # select_obs_polyline_list = [
        #     obs_id_track_dict[obs_id][seq_num], obs_id_track_dict[self.adc_id][seq_num]
        # ]
        select_lane_polyline_list = []
        for other_obs_id in seq_num_obs_id_dict[seq_num]:
            if other_obs_id == obs_id or other_obs_id == self.adc_id:
                continue
            obs_polyline = obs_id_track_dict[other_obs_id][seq_num][0]
            valid_polyline = bound_polyline[obs_polyline[:, -1] == 1]
            local_vectors = global2local_array(valid_polyline[:, :2], pose)
            for v in local_vectors:
                if self.left_dist <= v[0] <= self.right_dist and self.back_dist <= v[1] <= self.forward_dist:
                    min_dist = np.min(np_vector_norm(local_vectors), axis=0)
                    select_bound_polyline_list.append(obs_polyline, min_dist)
                    break
        '''

    def feature_transform(self, seq_num_pred_id_dict, seq_num_obs_id_dict, obs_id_track_dict, seq_num_map_dict,
                          curr_seq_num_dict):
        sample_list = []
        # sample time not every frame TODO(zhangteng)
        for seq_num, pred_id_list in seq_num_pred_id_dict.items():
            for obs_id in pred_id_list:
                pose = obs_id_track_dict[obs_id][seq_num][-1]
                obs_array_list, lane_array_list, bound_array_list = self.get_agent_array_list(
                    seq_num_obs_id_dict, obs_id_track_dict, seq_num_map_dict, curr_seq_num_dict, obs_id, seq_num)

                for i, obs_array in enumerate(obs_array_list):
                    mask = obs_array[:, -1] == 1
                    # start pos
                    obs_array[mask, 0:2] = global2local_array(obs_array[mask][:, 0:2], pose)
                    # end pos
                    obs_array[mask, 2:4] = global2local_array(obs_array[mask][:, 2:4], pose)
                    # lf pos
                    obs_array[mask, 17:19] = global2local_array(obs_array[mask][:, 17:19], pose)
                    # rf pos
                    obs_array[mask, 19:21] = global2local_array(obs_array[mask][:, 19:21], pose)
                    # lr pos
                    obs_array[mask, 21:23] = global2local_array(obs_array[mask][:, 21:23], pose)
                    # rr pos
                    obs_array[mask, 23:25] = global2local_array(obs_array[mask][:, 23:25], pose)
                    velocity_pose = np.array([0, 0, pose[-1]])
                    # velocity
                    obs_array[mask, 7:9] = global2local_array(obs_array[mask][:, 7:9], velocity_pose)
                    # acc
                    obs_array[mask, 11:13] = global2local_array(obs_array[mask][:, 11:13], velocity_pose)
                    # heading
                    rotate_angle = np.pi / 2.0 - pose[-1]
                    obs_array[mask, 4] = np_vector_norm(obs_array[mask][:, 4] + rotate_angle)

                    obs_array_list[i] = obs_array

                for i, lane_array in enumerate(lane_array_list):
                    # start pos
                    lane_array[mask, 0:2] = global2local_array(lane_array[mask][:, 0:2], pose)
                    # end pos
                    lane_array[mask, 2:4] = global2local_array(lane_array[mask][:, 2:4], pose)
                    # heading
                    rotate_angle = np.pi / 2.0 - pose[-1]
                    lane_array[mask, 4] = np_vector_norm(lane_array[mask][:, 4] + rotate_angle)

                    lane_array_list[i] = lane_array

                for i, bound_array in enumerate(bound_array_list):
                    # start pos
                    bound_array[mask, 0:2] = global2local_array(bound_array[mask][:, 0:2], pose)
                    # end pos
                    bound_array[mask, 2:4] = global2local_array(bound_array[mask][:, 2:4], pose)
                    # heading
                    rotate_angle = np.pi / 2.0 - pose[-1]
                    bound_array[mask, 4] = np_vector_norm(bound_array[mask][:, 4] + rotate_angle)

                    bound_array_list[i] = bound_array

                assert len(obs_array_list) <= self.max_obstacle_num
                assert len(lane_array_list) <= self.max_lane_num
                assert len(bound_array_list) <= self.max_bound_num

                obs_mat = np.array(obs_array_list)
                obs_mask = np.ones((self.max_obstacle_num))
                if len(obs_array_list) < self.max_obstacle_num:
                    obs_mat = np.concatenate([
                        obs_mat,
                        np.zeros(self.max_obstacle_num - obs_mat.shape[0], self.obs_max_vector_num, self.obs_vector_dim)
                    ],
                                             dim=0)
                    obs_mask[len(obs_array_list):] = 0.0
                lane_mat = np.array(lane_array_list)
                lane_mask = np.ones((self.max_lane_num))
                if len(lane_array_list) < self.max_lane_num:
                    lane_mat = np.concatenate([
                        lane_mat,
                        np.zeros(self.max_lane_num - lane_mat.shape[0], self.lane_max_vector_num, self.lane_vector_dim)
                    ],
                                              dim=0)
                    lane_mask[len(lane_array_list):] = 0.0
                bound_mat = np.array(bound_array_list)
                bound_mask = np.ones((self.max_bound_num))
                if len(bound_array_list) < self.max_bound_num:
                    bound_mat = np.concatenate([
                        bound_mat,
                        np.zeros(self.max_bound_num - bound_mat.shape[0], self.lane_max_vector_num,
                                 self.lane_vector_dim)
                    ],
                                               dim=0)
                    bound_mask[len(bound_array_list):] = 0.0
                label_traj_array = self.sample_label(obs_id, obs_id_track_dict)

                sample = AgentClosure()
                sample.obs_tensor = torch.from_numpy(obs_mat).unsqueeze(0)
                sample.lane_tensor = torch.from_numpy(lane_mat).unsqueeze(0)
                sample.bound_tensor = torch.from_numpy(bound_mat).unsqueeze(0)
                sample.obs_mask = torch.from_numpy(obs_mask).unsqueeze(0)
                sample.lane_mask = torch.from_numpy(lane_mask).unsqueeze(0)
                sample.bound_mask = torch.from_numpy(bound_mask).unsqueeze(0)
                sample.traj_gt = torch.from_numpy(label_traj_array).unsqueeze(0)
                sample.raw_sample_name = str(seq_num) + '_' + str(obs_id)

                sample_list.append(sample)

        return sample_list

    def sample_label(self, obs_id, seq_num, obs_id_track_dict):
        seq_num_track_dict = obs_id_track_dict[obs_id]
        pose = seq_num_track_dict[seq_num][-1]
        future_pos_list = []
        for i in range(1, self.output_frame_num + 1):
            future_seq_num = seq_num + i * self.output_frame_reso
            assert future_seq_num in seq_num_track_dict
            future_pos_list.append(seq_num_track_dict[future_seq_num][-1][:2])
        assert len(future_pos_list) == self.output_frame_num
        future_pos_array = np.array(future_pos_list)
        agent_traj_label = global2local_array(future_pos_array, pose)
        return agent_traj_label


def generate_sample_per_file(parser: Parser, generator: SampleGenerator, input_file, output_dir):
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    saver = SampleSaver(output_file)
    seq_num_obs_id_dict, obs_id_track_dict, seq_num_map_dict, curr_seq_num_dict = parser.parser(input_file)
    generator.generate_sample(seq_num_obs_id_dict, obs_id_track_dict, seq_num_map_dict, curr_seq_num_dict, saver)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pb_dir", type=str, required=True, help="pb files dir to input")
    parser.add_argument("--output_dir", type=str, required=True, help="samples to output")
    parser.add_argument("--config", type=str, required=False, default='config.yaml', help="samples to output")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        data_config_dict = yaml.safe_load(f)
        print(data_config_dict)

    parser = Parser(data_config_dict)
    generator = SampleGenerator(data_config_dict)

    input_file_list = glob.glob(args.input_pb_dir + '/*.pb')
    if data_config_dict['fsd_processor']['db_process_num'] == 1:
        for input_file in tqdm.tqdm(input_file_list):
            generate_sample_per_file(parser, generator, input_file, args.output_dir)
    else:
        num_worker = data_config_dict['fsd_processor']['db_process_num']
        Parallel(n_jobs=num_worker)(delayed(generate_sample_per_file)(parser, generator, input_file, args.output_dir)
                                    for input_file in input_file_list)
