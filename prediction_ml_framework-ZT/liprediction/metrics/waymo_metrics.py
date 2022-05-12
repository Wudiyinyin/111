# Copyright (c) 2021 Li Auto Company. All rights reserved.

import csv
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2
from waymo_open_dataset.protos.motion_submission_pb2 import MotionChallengeSubmission


class MotionMetrics(tf.keras.metrics.Metric):
    """Wrapper for motion metrics computation."""

    def __init__(self, config):
        super().__init__()
        self._prediction_trajectory = []
        self._prediction_score = []
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_ground_truth_indices = []
        self._prediction_ground_truth_indices_mask = []
        self._object_type = []
        self._metrics_config = config

    def num_of_predictions(self, metric_names: List[str]) -> Dict[str, int]:
        """Returns the number of predictions for each metric."""
        num_of_predictions = {}

        object_type: np.ndarray = np.array(self._object_type)
        object_type = np.squeeze(object_type)
        gt_truth_valid: np.ndarray = np.array(self._ground_truth_is_valid)
        gt_truth_valid = np.squeeze(gt_truth_valid)

        vehicle_gt_truth_valid: np.ndarray = gt_truth_valid[object_type == 1]
        pedestrian_gt_truth_valid: np.ndarray = gt_truth_valid[object_type == 2]
        cyclist_gt_truth_valid: np.ndarray = gt_truth_valid[object_type == 3]

        prediction_steps_per_second: int = self._metrics_config.prediction_steps_per_second
        track_history_samples: int = self._metrics_config.track_history_samples

        for metric_name in metric_names:
            type_name: str = metric_name.split('_')[1]
            step_measurement: int = int(metric_name.split('_')[-1])
            # 1 means the current frame
            measurement_idx: int = track_history_samples + 1 + prediction_steps_per_second * step_measurement

            if type_name == "VEHICLE":
                num_of_predictions[metric_name] = np.count_nonzero(vehicle_gt_truth_valid[:, measurement_idx])
            elif type_name == "PEDESTRIAN":
                num_of_predictions[metric_name] = np.count_nonzero(pedestrian_gt_truth_valid[:, measurement_idx])
            elif type_name == "CYCLIST":
                num_of_predictions[metric_name] = np.count_nonzero(cyclist_gt_truth_valid[:, measurement_idx])
            else:
                raise ValueError("Unknown type name: {}".format(type_name))

        return num_of_predictions

    def reset_states(self):
        self._prediction_trajectory = []
        self._prediction_score = []
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_ground_truth_indices = []
        self._prediction_ground_truth_indices_mask = []
        self._object_type = []

    def update_state(self, prediction_trajectory, prediction_score, ground_truth_trajectory, ground_truth_is_valid,
                     prediction_ground_truth_indices, prediction_ground_truth_indices_mask, object_type):
        self._prediction_trajectory.append(prediction_trajectory)
        self._prediction_score.append(prediction_score)
        self._ground_truth_trajectory.append(ground_truth_trajectory)
        self._ground_truth_is_valid.append(ground_truth_is_valid)
        self._prediction_ground_truth_indices.append(prediction_ground_truth_indices)
        self._prediction_ground_truth_indices_mask.append(prediction_ground_truth_indices_mask)
        self._object_type.append(object_type)

    def result(self):
        # [batch_size, num_preds, 1, 1, steps, 2].
        # The ones indicate top_k = 1, num_agents_per_joint_prediction = 1.
        prediction_trajectory = tf.concat(self._prediction_trajectory, 0)
        # [batch_size, num_preds, 1].
        prediction_score = tf.concat(self._prediction_score, 0)
        # [batch_size, num_agents, gt_steps, 7].
        ground_truth_trajectory = tf.concat(self._ground_truth_trajectory, 0)
        # [batch_size, num_agents, gt_steps].
        ground_truth_is_valid = tf.concat(self._ground_truth_is_valid, 0)
        # [batch_size, num_preds, 1].
        prediction_ground_truth_indices = tf.concat(self._prediction_ground_truth_indices, 0)
        # [batch_size, num_preds, 1].
        prediction_ground_truth_indices_mask = tf.concat(self._prediction_ground_truth_indices_mask, 0)
        # [batch_size, num_agents].
        object_type = tf.cast(tf.concat(self._object_type, 0), tf.int64)

        return py_metrics_ops.motion_metrics(config=self._metrics_config.SerializeToString(),
                                             prediction_trajectory=prediction_trajectory,
                                             prediction_score=prediction_score,
                                             ground_truth_trajectory=ground_truth_trajectory,
                                             ground_truth_is_valid=ground_truth_is_valid,
                                             prediction_ground_truth_indices=prediction_ground_truth_indices,
                                             prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
                                             object_type=object_type)


def dump_prediction_results(file_path: Path, output_file: Path, gt_file: Path) -> None:
    """Dump prediction results from a file.

    Args:
        file_path: Path to the file containing prediction results pb.
        output_file: Path to the output file.
        gt_file: ground truth file.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file {gt_file} does not exist")

    with open(file_path, "rb") as f:
        prediction_results = MotionChallengeSubmission()
        prediction_results.ParseFromString(f.read())

        metrics_config = default_metrics_config()
        motion_metrics = MotionMetrics(metrics_config)
        metric_names = config_util.get_breakdown_names_from_motion_config(metrics_config)

        print("Total scenario num : ", len(prediction_results.scenario_predictions))
        if prediction_results.submission_type == MotionChallengeSubmission.MOTION_PREDICTION:
            prediction_metrics, prediction_num = dump_single_prediction_results(prediction_results.scenario_predictions,
                                                                                gt_file, motion_metrics, metric_names)
        elif prediction_results.submission_type == MotionChallengeSubmission.INTERACTION_PREDICTION:
            dump_joint_prediction_results(prediction_results.scenario_predictions, gt_file, motion_metrics)
        else:
            raise ValueError(f"Unknown submission type {prediction_results.submission_type}")

        avg_metrics = calculate_avg_result(prediction_metrics, prediction_num)
        dump_metric_csv(avg_metrics, metric_names, output_file)


def calculate_avg_result(prediction_metrics: List[tf.Tensor], num_of_predictions: List[Dict[str, int]]) -> np.ndarray:
    """Calculate average result.

    Args:
        prediction_metrics: Metrics.
        num_of_predictions: Number of predictions.

    Returns:
        Average metrics.
    """
    metric_shape = prediction_metrics[0].numpy().shape
    sum_metrics = np.zeros(metric_shape)
    total_num = np.zeros(metric_shape[1])

    for metric, num_of_predict in zip(prediction_metrics, num_of_predictions):
        sum_metrics += metric.numpy() * np.array(list(num_of_predict.values()))
        total_num += np.array(list(num_of_predict.values()))

    avg_metrics = sum_metrics / total_num

    avg_all_type = np.sum(avg_metrics * total_num / np.sum(total_num), axis=-1)
    avg_all_type = np.expand_dims(avg_all_type, axis=-1)

    avg_metrics = np.concatenate((avg_metrics, avg_all_type), axis=-1)

    return avg_metrics


def dump_metric_csv(prediction_metric_overall, metric_names: List[str], output_file: Path) -> None:
    """Dump metric csv.

    Args:
        prediction_metric_overall: shape [5, 10], last column is average of all metrics
        metric_names: metric names, which is 9 metrics for standard competition
        output_file: output file
    """

    with open(str(output_file), 'w', newline='') as csvfile:
        fieldnames = ['metric_name', 'min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'mAP']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, metric_name in enumerate(metric_names):
            writer.writerow({
                'metric_name': metric_name,
                'min_ade': float(prediction_metric_overall[0][i]),
                'min_fde': float(prediction_metric_overall[1][i]),
                'miss_rate': float(prediction_metric_overall[2][i]),
                'overlap_rate': float(prediction_metric_overall[3][i]),
                'mAP': float(prediction_metric_overall[4][i])
            })

        writer.writerow({
            'metric_name': 'Avg',
            'min_ade': float(prediction_metric_overall[0][-1]),
            'min_fde': float(prediction_metric_overall[1][-1]),
            'miss_rate': float(prediction_metric_overall[2][-1]),
            'overlap_rate': float(prediction_metric_overall[3][-1]),
            'mAP': float(prediction_metric_overall[4][-1])
        })


def load_ground_truth(gt_file: Path) -> Dict:
    """Load ground truth from a pickle.

    Args:
        gt_file: Path to the ground truth file.

    Returns:
        A dictionary containing the ground truth.
    """
    gt_scene_dict = {}
    with open(gt_file, "rb") as f:
        gt_scene_dict = pickle.load(f)
    return gt_scene_dict


def dump_single_prediction_results(scenario_predictions: List[Any],
                                   gt_file: Path,
                                   motion_metrics: MotionMetrics,
                                   metric_names: List[str],
                                   dump_result_num: int = 1000) -> Tuple[List]:

    gt_scene_dict: Dict = load_ground_truth(gt_file)

    prediction_metrics: List[Dict] = []
    prediction_num: List[Dict] = []
    scene_num = 0
    for scenario_prediction in scenario_predictions:
        scene_num += 1
        assert scenario_prediction.HasField('single_predictions')

        scenario_id = scenario_prediction.scenario_id
        for single_prediction in scenario_prediction.single_predictions.predictions:
            obj_id = single_prediction.object_id

            trajectories: List[np.ndarray] = []
            confidence_scores: List[float] = []
            for multi_modal_trajectory in single_prediction.trajectories:
                trajectory_x: np.ndarray = np.array(multi_modal_trajectory.trajectory.center_x)
                trajectory_y: np.ndarray = np.array(multi_modal_trajectory.trajectory.center_y)
                trajectory: np.ndarray = np.stack([trajectory_x, trajectory_y], axis=1)
                trajectories.append(trajectory)

                confidence_scores.append(multi_modal_trajectory.confidence)

            # Modality = 6
            # pred_trajectory.shape : [1, 1, 6, 1, 16, 2]
            pred_trajectory: np.ndarray = np.stack(trajectories, axis=0)
            pred_trajectory = np.expand_dims(pred_trajectory, axis=1)
            pred_trajectory = np.expand_dims(pred_trajectory, axis=0)
            pred_trajectory = np.expand_dims(pred_trajectory, axis=0)
            pred_trajectory = tf.convert_to_tensor(pred_trajectory, dtype=tf.float32)

            # pred_score.shape : (1, 1, 6)
            pred_score: np.ndarray = np.array(confidence_scores)
            pred_score = np.expand_dims(pred_score, axis=0)
            pred_score = np.expand_dims(pred_score, axis=0)
            pred_score = tf.convert_to_tensor(pred_score, dtype=tf.float32)

            gt_info: List = gt_scene_dict[scenario_id]['gt_info'][obj_id]
            gt_state, gt_mask, gt_type = gt_info

            # gt_trajectory.shape : (1, 1, 91, 7)
            gt_trajectory: np.ndarray = gt_state
            gt_trajectory = np.expand_dims(gt_trajectory, axis=0)
            gt_trajectory = np.expand_dims(gt_trajectory, axis=0)
            gt_trajectory = tf.convert_to_tensor(gt_trajectory, dtype=tf.float32)

            # gt_is_valid.shape : (1, 1, 91)
            gt_is_valid: np.ndarray = gt_mask
            gt_is_valid = np.expand_dims(gt_is_valid, axis=0)
            gt_is_valid = np.expand_dims(gt_is_valid, axis=0)
            gt_is_valid = tf.convert_to_tensor(gt_is_valid, dtype=tf.bool)

            # pred_gt_indices.shape : (1, 1, 1)
            pred_gt_indices: np.ndarray = np.zeros((1, 1, 1))
            pred_gt_indices = tf.convert_to_tensor(pred_gt_indices, dtype=tf.int64)

            # pred_gt_indices_mask.shape : (1, 1, 1)
            pred_gt_indices_mask: np.ndarray = np.ones((1, 1, 1))
            pred_gt_indices_mask = tf.convert_to_tensor(pred_gt_indices_mask, dtype=tf.bool)

            # object_type.shape : (1, 1)
            object_type: np.ndarray = gt_type[11]  # current timestamp
            object_type = np.expand_dims(object_type, axis=0)
            object_type = np.expand_dims(object_type, axis=0)
            object_type = tf.convert_to_tensor(object_type, dtype=tf.int64)

            motion_metrics.update_state(pred_trajectory, pred_score, gt_trajectory, gt_is_valid, pred_gt_indices,
                                        pred_gt_indices_mask, object_type)

        if scene_num % dump_result_num == 0:
            print("Dump result idx : ", scene_num)
            prediction_metric_values = motion_metrics.result()
            prediction_metrics.append(prediction_metric_values)
            prediction_num.append(motion_metrics.num_of_predictions(metric_names))

            motion_metrics.reset_states()

    return prediction_metrics, prediction_num


def dump_joint_prediction_results(scenario_predictions: List[Any], gt_file: Path, motion_metrics):
    pass


def default_metrics_config():
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
        track_steps_per_second: 10
        prediction_steps_per_second: 2
        track_history_samples: 10
        track_future_samples: 80
        speed_lower_bound: 1.4
        speed_upper_bound: 11.0
        speed_scale_lower: 0.5
        speed_scale_upper: 1.0
        step_configurations {
            measurement_step: 5
            lateral_miss_threshold: 1.0
            longitudinal_miss_threshold: 2.0
        }
        step_configurations {
            measurement_step: 9
            lateral_miss_threshold: 1.8
            longitudinal_miss_threshold: 3.6
        }
        step_configurations {
            measurement_step: 15
            lateral_miss_threshold: 3.0
            longitudinal_miss_threshold: 6.0
        }
        max_predictions: 6
    """
    text_format.Parse(config_text, config)
    return config


if __name__ == '__main__':

    metrics_config = default_metrics_config()
    motion_metrics = MotionMetrics(metrics_config)
    metric_names = config_util.get_breakdown_names_from_motion_config(metrics_config)
    print(metric_names)

    # - Notations:
    # - B: batch size. Each batch should contain exactly 1 scenario.
    # - M: Number of joint prediction groups to predict per scenario.
    # - K: top_K predictions per joint prediction.
    # - N: number of agents in a joint prediction. 1 if mutual independence is
    #     assumed between agents.
    # - A: number of agents in the groundtruth.
    # - TP: number of steps to evaluate on. Matches len(config.step_measurement).
    # - TG: number of steps in the groundtruth track. Matches
    #     config.track_history_samples + 1 + config.future_history_samples.
    # - BR: number of breakdowns.

    Modality = 6
    # Example data : Single Trajectory
    # [B, M, K, N, TP, 2] Predicted trajectories, 2 = [x, y]
    pred_trajectory = np.random.rand(1, 1, Modality, 1, 16, 2)
    pred_trajectory[0, 0, :, 0, :, :] = 0.0
    pred_trajectory[0, 0, 2, 0, :, :] = 3.0
    pred_trajectory = tf.convert_to_tensor(pred_trajectory, dtype=tf.float32)
    # [B, M, K] Scores per joint prediction
    pred_score = np.random.rand(1, 1, Modality)
    # [0.6, 0.5, 0.4]
    # [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    pred_score[0, 0, :] = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    pred_score = tf.convert_to_tensor(pred_score, dtype=tf.float32)
    # [B, A, TG, 7] Groundtruth trajectories, 7 = [x, y, length, width, heading, velocity_x, velocity_y]
    gt_trajectory = np.random.rand(1, 1, 91, 7)
    gt_trajectory[0, 0, :, :] = 3.0
    gt_trajectory = tf.convert_to_tensor(gt_trajectory, dtype=tf.float32)
    # [B, A, TG]
    gt_is_valid = np.ones((1, 1, 91))
    gt_is_valid = tf.convert_to_tensor(gt_is_valid, dtype=tf.bool)
    # [B, M, N]
    pred_gt_indices = np.zeros((1, 1, 1))
    pred_gt_indices = tf.convert_to_tensor(pred_gt_indices, dtype=tf.int64)
    # [B, M, N]
    pred_gt_indices_mask = np.ones((1, 1, 1))
    pred_gt_indices_mask = tf.convert_to_tensor(pred_gt_indices_mask, dtype=tf.bool)
    # [B, A]
    object_type = np.ones((1, 1))
    object_type = tf.convert_to_tensor(object_type, dtype=tf.int64)

    motion_metrics.update_state(pred_trajectory, pred_score, gt_trajectory, gt_is_valid, pred_gt_indices,
                                pred_gt_indices_mask, object_type)
    motion_metrics.update_state(pred_trajectory, pred_score, gt_trajectory, gt_is_valid, pred_gt_indices,
                                pred_gt_indices_mask, object_type)
    motion_metrics.update_state(pred_trajectory, pred_score, gt_trajectory, gt_is_valid, pred_gt_indices,
                                pred_gt_indices_mask, object_type)

    # Display metrics at the end of each epoch.
    prediction_metric_values = motion_metrics.result()
    for i, m in enumerate(['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'mAP']):
        for j, n in enumerate(metric_names):
            print('{}/{}: {}'.format(m, n, prediction_metric_values[i, j]))

    motion_metrics.reset_states()

    Modality = 6
    # Example data : Joint Trajectory
    # [B, M, K, N, TP, 2] Predicted trajectories, 2 = [x, y]
    pred_trajectory = np.random.rand(1, 1, Modality, 2, 16, 2)
    pred_trajectory[0, 0, :, :, :, :] = 0.0
    pred_trajectory[0, 0, 2, :, :, :] = 3.0
    pred_trajectory = tf.convert_to_tensor(pred_trajectory, dtype=tf.float32)
    # [B, M, K] Scores per joint prediction
    pred_score = np.random.rand(1, 1, Modality)
    # [0.6, 0.5, 0.4]
    # [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    pred_score[0, 0, :] = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    pred_score = tf.convert_to_tensor(pred_score, dtype=tf.float32)
    # [B, A, TG, 7] Groundtruth trajectories, 7 = [x, y, length, width, heading, velocity_x, velocity_y]
    gt_trajectory = np.random.rand(1, 2, 91, 7)
    gt_trajectory[0, 0, :, :] = 3.0
    gt_trajectory = tf.convert_to_tensor(gt_trajectory, dtype=tf.float32)
    # [B, A, TG]
    gt_is_valid = np.ones((1, 2, 91))
    gt_is_valid = tf.convert_to_tensor(gt_is_valid, dtype=tf.bool)
    # [B, M, N]
    pred_gt_indices = np.zeros((1, 1, 2))
    pred_gt_indices = tf.convert_to_tensor(pred_gt_indices, dtype=tf.int64)
    # [B, M, N]
    pred_gt_indices_mask = np.ones((1, 1, 2))
    pred_gt_indices_mask = tf.convert_to_tensor(pred_gt_indices_mask, dtype=tf.bool)
    # [B, A]
    object_type = np.ones((1, 2))
    object_type = tf.convert_to_tensor(object_type, dtype=tf.int64)

    motion_metrics.update_state(pred_trajectory, pred_score, gt_trajectory, gt_is_valid, pred_gt_indices,
                                pred_gt_indices_mask, object_type)
    motion_metrics.update_state(pred_trajectory, pred_score, gt_trajectory, gt_is_valid, pred_gt_indices,
                                pred_gt_indices_mask, object_type)
    motion_metrics.update_state(pred_trajectory, pred_score, gt_trajectory, gt_is_valid, pred_gt_indices,
                                pred_gt_indices_mask, object_type)

    # Display metrics at the end of each epoch.
    prediction_metric_values = motion_metrics.result()
    for i, m in enumerate(['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'mAP']):
        for j, n in enumerate(metric_names):
            print('{}/{}: {}'.format(m, n, prediction_metric_values[i, j]))
