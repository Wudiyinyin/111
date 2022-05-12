# Copyright (c) 2021 Li Auto Company. All rights reserved.

import argparse
import copy
import glob
import os
import pickle
from typing import Dict, List

import numpy as np
from joblib import Parallel, delayed
from waymo_open_dataset.protos.motion_submission_pb2 import ChallengeScenarioPredictions, MotionChallengeSubmission


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_pkl_dir", type=str, required=True, help="dir contain inference pkl file")
    parser.add_argument("--output_pb_file", type=str, required=True, help="file output to submission")
    parser.add_argument("--gt_file", type=str, required=True, help="dataset ground truth of pkl file")
    parser.add_argument("--task_type", type=str, required=True, help="single or multi")
    parser.add_argument("--email", type=str, required=True, help="email register in waymo")
    args = parser.parse_args()
    return args


class SaveWaymoPb():

    def __init__(self, input_data_dir, output_pb_file, e_mail, desc, info_dict_file, task_type):
        self.input_data_dir = input_data_dir
        self.output_pb_file = output_pb_file
        self.email = e_mail
        self.desc = desc

        # scene_info_dict: scene_id(str): {'scene_id'(str), 'file_name'(str), 'pred_obs'(list), 'adc_id'(int)}
        self.scene_info_dict = None
        with open(info_dict_file, 'rb') as f:
            self.scene_info_dict = pickle.load(f)

        self.task_type = task_type

    def convert_pkl_to_pb(self):
        file_list = glob.glob(self.input_data_dir + '/*.pkl')
        scene_file_dict = {}
        for input_file in file_list:
            scene_id, _, obs_id = os.path.basename(input_file[:-4]).split('_')
            scene_file_dict.setdefault(scene_id, []).append(input_file)
        # check all pred_obs is predicted
        for scene_id, scene_file_list in scene_file_dict.items():
            pred_list = copy.deepcopy(self.scene_info_dict[scene_id]['pred_obs'])
            for file_name in scene_file_list:
                scene_id, _, obs_id = os.path.basename(file_name[:-4]).split('_')
                obs_id = int(obs_id) if obs_id != '-1' else self.scene_info_dict[scene_id]['adc_id']
                pred_list.remove(obs_id)
            assert len(pred_list) == 0

        # Not use Parallel for now
        scenes_pb = Parallel(n_jobs=1)(delayed(self.save_submission_per_scene)(scene_id, file_list)
                                       for scene_id, file_list in scene_file_dict.items())

        self.save_submission_file(self.output_pb_file, self.task_type, self.email, scenes_pb, desc=self.desc)

    def save_submission_per_scene(self, scene_id: str, pred_files):
        """Save submission result

        Args:
            scene_id: str
            pred_files: list of pkl_files
        """

        instance = ChallengeScenarioPredictions()
        instance.scenario_id = scene_id
        adc_id = self.scene_info_dict[scene_id]['adc_id']

        if self.task_type == 1:
            pred_info = {}
            for input_file in pred_files:
                # example: input_file
                #     scene_id, seq_num, obs_id
                #     8231bec88aa207c8_10_542.pkl
                _, _, obs_id = os.path.basename(input_file[:-4]).split('_')
                with open(input_file, 'rb') as f:
                    obs_info = pickle.load(f)

                # Convert adc_id to obs_id
                obs_id = int(obs_id) if obs_id != '-1' else adc_id

                pred_info[obs_id] = [obs_info['traj'], obs_info['traj_prob']]

            self.save_single_pred(pred_info, instance)

        elif self.task_type == 2:
            # Joint prediction task only save one files for scenario
            assert len(pred_files) == 1

            pred_info = {}
            with open(pred_files[0], 'rb') as f:
                obs_info = pickle.load(f)

            # Convert adc_id to obs_id
            obs_ids = []
            for obs_id in obs_info['obs_id']:
                obs_id = int(obs_id) if obs_id != '-1' else adc_id
                obs_ids.append(obs_id)

            pred_info['obs_id'] = obs_ids
            pred_info['pred_scene'] = [obs_info['traj'], obs_info['traj_prob']]

            self.save_interaction_pred(obs_info, instance)

        else:
            raise ValueError('task_type should be 1 or 2')

        return instance

    def save_single_pred(self, pred_info: Dict, instance):
        """
        Args:
            pred_info:
                scene_id(str): str
                obs_id(int): [traj(ndarray, (N, T, 2)), prob(nd.array, (N))]
            instance: pb object
        """
        for obs_id in pred_info.keys():
            pred_trajs = instance.single_predictions.predictions.add()
            pred_trajs.object_id = obs_id

            for i in range(pred_info[obs_id][0].shape[0]):
                pred_traj: np.ndarray = pred_info[obs_id][0][i]
                pred_score = pred_info[obs_id][1][i]
                dump_traj = pred_trajs.trajectories.add()
                dump_traj.confidence = pred_score
                traj_x = pred_traj[:, 0].tolist()
                traj_y = pred_traj[:, 1].tolist()
                dump_traj.trajectory.center_x.extend(traj_x)
                dump_traj.trajectory.center_y.extend(traj_y)

    def save_interaction_pred(self, pred_info: Dict, instance):
        """
        Args:
            pred_info:
                obs_id(str): list/ndarray (2)
                pred_scene(str): [traj(ndarray, (N, T, 4)), prob(nd.array, (N))]
                Note: obs0-traj[:, :2],  obs1-traj[:, 2:4]
            instance: pb object
        """
        obs_id = pred_info['obs_id']
        for i in range(pred_info['pred_scene'][0].shape[0]):
            dump_scene = instance.joint_prediction.joint_trajectories.add()
            pred_traj = pred_info['pred_scene'][0][i]
            pred_score = pred_info['pred_scene'][1][i]
            dump_scene.confidence = pred_score
            obs_traj0 = dump_scene.trajectories.add()
            obs_traj0.object_id = obs_id[0]
            obs_traj0.trajectory.center_x.extend(pred_traj[:, 0].tolist())
            obs_traj0.trajectory.center_y.extend(pred_traj[:, 1].tolist())
            obs_traj1 = dump_scene.trajectories.add()
            obs_traj1.object_id = obs_id[1]
            obs_traj1.trajectory.center_x.extend(pred_traj[:, 2].tolist())
            obs_traj1.trajectory.center_y.extend(pred_traj[:, 3].tolist())

    def save_submission_file(self,
                             output_pb_file: str,
                             submission_type: int,
                             account: str,
                             pred_scenes: List,
                             method_name: str = None,
                             author_list: List[str] = None,
                             desc: str = ''):
        with open(output_pb_file, 'wb') as f:
            submission_instance = MotionChallengeSubmission()
            if method_name is None:
                method_name = 'DenseTNT'
            if author_list is None:
                author_list = ['Kun Zhan', 'Leimeng Xu', 'Teng Zhang']
            assert submission_type in [
                MotionChallengeSubmission.SubmissionType.MOTION_PREDICTION,
                MotionChallengeSubmission.SubmissionType.INTERACTION_PREDICTION
            ]
            submission_instance.submission_type = submission_type
            submission_instance.account_name = account
            submission_instance.description = desc
            submission_instance.unique_method_name = method_name
            submission_instance.authors.extend(author_list)

            print(f"Total scenario number: {len(pred_scenes)}")
            for pred_scene in pred_scenes:
                scene_pred_pb = submission_instance.scenario_predictions.add()
                scene_pred_pb.CopyFrom(pred_scene)

            f.write(submission_instance.SerializeToString())


if __name__ == "__main__":
    args: argparse.ArgumentParser = parse_args()

    if args.task_type == 'single':
        task_type = 1
    elif args.task_type == 'multi':
        task_type = 2
    else:
        assert False

    save_pb = SaveWaymoPb(args.input_pkl_dir, args.output_pb_file, args.email, args.desc, args.gt_file, task_type)
    save_pb.convert_pkl_to_pb()
