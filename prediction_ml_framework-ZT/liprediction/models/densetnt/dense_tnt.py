# Copyright (c) 2021 Li Auto Company. All rights reserved.

import os
import pickle as pkl
import sys
from pathlib import Path
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from core.utils import metrics_calc as mc
from core.utils.utils import local2global, pose2rt
from datasets.processer.feature_processer import plot_transformed_data
from datasets.transform.agent_closure import AgentClosureBatch
from datasets.transform.dataset import PickleDataset
from datasets.transform.sample_transform.dense_tnt_transform import DenseTNTTransform
from module.context_encoder.context_encoder import ContextEncoder
from module.intention_decoder.intention_decoder import IntentionDecoder
from module.intention_sampler.intention_sampler import IntentionSampler
from module.trajectory_decoder.trajectory_decoder import TrajectoryDecoder
from torch.utils.data import DataLoader


class DenseTNT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.context_encoder = ContextEncoder.Create(config['context_encoder'])
        self.intention_decoder = IntentionDecoder.Create(config['intention_decoder'])
        self.traj_decoder = TrajectoryDecoder.Create(config['trajectory_decoder'])

    def forward(self, vectors, vector_masks, cluster_masks, target_info):
        '''
            B: batch_size, C: max_polyline_len(context), L: max_vector_len, F: feature_size

            vectors: List Tensor [[B, C, L, F], [B, C, L, F]],
                respectively for Obstacle Vector and Lane Vector (or Map Vector)
            vector_masks: List Tensor [[B, C, L], [B, C, L]]
            cluster_masks: List Tensor [[B, C], [B, C]]
            target_info: list
                [0] cand_target_points: Candidate target point tensor [B, S, 2],
                    S is the number of goals (H_dim * W_dim)
                [1] gt_target_grid_center [B, I, 2], I is the number of goals [3s 5s 8s]
        '''
        # list[[B, C, L, F]], list[[B, C, L]], list[[B, C]]
        #   -> [ [B, C', F_c'], [B, C', F_c'] ], [ [B, C'], [B, C'] ], cluster along L
        polyline_features, polyline_masks = self.context_encoder(vectors, vector_masks, cluster_masks)

        # [B, S, 2], [ [B, C', F_c'], [B, C', F_c'] ], [ [B, C'], [B, C'] ]
        #   -> [ [B, S, 1] x 3 ], predict feature for each goals 3s, 5s, 8s
        target_score_raws = self.intention_decoder(target_info[0], polyline_features, polyline_masks)

        # [B, 1, 2], [ [B, C', F_c'], [B, C', F_c'] ], [ [B, C'], [B, C'] ]
        #   -> [B, 1, T*2], predict the target goal trajectory and target_info[1][:, -1, :] is the 8s target
        traj = self.traj_decoder(target_info[1][:, [-1], :], polyline_features, polyline_masks)

        # [ [B, S, 1] x 3 ], [B, 1, T*2], T is the number of trajectory points (such as 30 or 50 ...)
        return target_score_raws, traj


class DenseTNTInfer(nn.Module):

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

        self.intention_sampler = IntentionSampler.Create(config['intention_sampler'])

    def forward(self, vectors, vector_masks, cluster_masks, target_info):
        '''
            B: batch_size, C: max_polyline_len(context), L: max_vector_len, F: feature_size

            vectors: List Tensor [[B, C, L, F], [B, C, L, F]],
                respectively for Obstacle Vector and Lane Vector (or Map Vector)
            vector_masks: List Tensor [[B, C, L], [B, C, L]]
            cluster_masks: List Tensor [[B, C], [B, C]]
            target_info: list
                [0] cand_target_points: Candidate target point tensor [B, S, 2],
                    S is the number of goals (H_dim * W_dim)
        '''
        # list[[B, C, L, F]], list[[B, C, L]], list[[B, C]]
        #   -> [ [B, C', F_c'], [B, C', F_c'] ], [ [B, C'], [B, C'] ], cluster along L
        polyline_features, polyline_masks = self.model.context_encoder(vectors, vector_masks, cluster_masks)

        # [B, S, 2], [ [B, C', F_c'], [B, C', F_c'] ], [ [B, C'], [B, C'] ]
        #   -> [ [B, S, 1] x 3 ], predict feature for each goals 3s, 5s, 8s
        target_score_raws = self.model.intention_decoder(target_info[0], polyline_features, polyline_masks)

        # [B, S, 1], [B, S, 2] -> [B, M, 2], [B, M, 1], M is modality number and target_score_raws[-1] is the 8s target
        teacher_points, target_prob = self.intention_sampler(target_score_raws[-1], target_info[0])

        # [B, M, 2], [ [B, C', F_c'], [B, C', F_c'] ], [ [B, C'], [B, C'] ]
        #   -> [B, M, T*2], predict the target goal trajectory
        traj = self.model.traj_decoder(teacher_points, polyline_features, polyline_masks)

        # [B, M, 2], [B, M, 1], [B, M, T*2], [ [B, S, 1] x 3 ]
        return teacher_points, target_prob, traj, target_score_raws


class DenseTNTPredictor(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = DenseTNT(config['model'])
        self.infer_model = DenseTNTInfer(self.model, config['model'])

    def forward(self, vectors, vector_masks, cluster_masks, target_info):
        return self.model(vectors, vector_masks, cluster_masks, target_info)

    def training_step(self, batch, batch_idx):
        vectors = [batch.actor_vector, batch.map_vector]
        vector_masks = [batch.actor_vector_mask, batch.map_vector_mask]
        cluster_masks = [batch.actor_cluster_mask, batch.map_cluster_mask]
        target_info = [batch.candi_target_points, batch.gt_target_grid_center]

        # [ [B, S, 1] x 3 ], [B, 1, T*2], S is the number of candicate goals
        target_score_raws, traj = self.model(vectors, vector_masks, cluster_masks, target_info)

        intention_3s_loss = self.intention_loss(target_score_raws[0], batch.gt_target_grid_idx[:, 0],
                                                batch.gt_target_point_mask[:, 0])
        intention_5s_loss = self.intention_loss(target_score_raws[1], batch.gt_target_grid_idx[:, 1],
                                                batch.gt_target_point_mask[:, 1])
        intention_8s_loss = self.intention_loss(target_score_raws[2], batch.gt_target_grid_idx[:, 2],
                                                batch.gt_target_point_mask[:, 2])
        traj_loss = self.traj_loss(traj, batch.gt_traj, batch.gt_traj_mask)

        loss = self.config['loss']['intention_3s_loss_wgt'] * intention_3s_loss + \
            self.config['loss']['intention_5s_loss_wgt'] * intention_5s_loss + \
            self.config['loss']['intention_8s_loss_wgt'] * intention_8s_loss + \
            self.config['loss']['traj_loss_wgt'] * traj_loss

        self.log('train_intention_8s_loss', intention_8s_loss)
        self.log('train_traj_loss', traj_loss)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        vectors = [batch.actor_vector, batch.map_vector]
        vector_masks = [batch.actor_vector_mask, batch.map_vector_mask]
        cluster_masks = [batch.actor_cluster_mask, batch.map_cluster_mask]
        target_info = [batch.candi_target_points, batch.gt_target_grid_center]

        # [ [B, S, 1] x 3 ], [B, 1, T*2], S is the number of candicate goals
        target_score_raws, traj = self.model(vectors, vector_masks, cluster_masks, target_info)

        intention_3s_loss = self.intention_loss(target_score_raws[0], batch.gt_target_grid_idx[:, 0],
                                                batch.gt_target_point_mask[:, 0])
        intention_5s_loss = self.intention_loss(target_score_raws[1], batch.gt_target_grid_idx[:, 1],
                                                batch.gt_target_point_mask[:, 1])
        intention_8s_loss = self.intention_loss(target_score_raws[2], batch.gt_target_grid_idx[:, 2],
                                                batch.gt_target_point_mask[:, 2])
        traj_loss = self.traj_loss(traj, batch.gt_traj, batch.gt_traj_mask)

        loss = self.config['loss']['intention_3s_loss_wgt'] * intention_3s_loss + \
            self.config['loss']['intention_5s_loss_wgt'] * intention_5s_loss + \
            self.config['loss']['intention_8s_loss_wgt'] * intention_8s_loss + \
            self.config['loss']['traj_loss_wgt'] * traj_loss

        self.log('val_intention_8s_loss', intention_8s_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_traj_loss', traj_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        vectors = [batch.actor_vector, batch.map_vector]
        vector_masks = [batch.actor_vector_mask, batch.map_vector_mask]
        cluster_masks = [batch.actor_cluster_mask, batch.map_cluster_mask]
        target_info = [batch.candi_target_points]
        # B is batch size. M is the num of trajectories. T is the num of timesteps.
        # [B, M, 2], [B, M, 1], [B, M, T*2]
        target_points, target_prob, pred_traj, target_score_raw = self.infer_model(vectors, vector_masks, cluster_masks,
                                                                                   target_info)

        if self.config['show_badcase']['plot_badcase']:
            traj_pred = pred_traj.view(pred_traj.shape[0], pred_traj.shape[1], -1, 2)
            plot_variables = [target_score_raw[2], traj_pred]
            self.plot_bad_case(batch, plot_variables, self.config['show _badcase']['final_timestep'])
        else:
            # Transform agent coord to global coord
            # rotation matrix
            agent_origin, agent_rotm = pose2rt(batch.agent_pose)
            # convert to global frame
            # T =  T*2 / 2, 8s = 80
            time_steps = pred_traj.shape[-1] // 2
            # [B, M, T*2] -> [B, M, T, 2]
            pred_traj = local2global(pred_traj.reshape(pred_traj.shape[0], pred_traj.shape[1], time_steps, 2),
                                     agent_origin, agent_rotm)

            # Downsample the trajectory with Waymo submission config - prediction_steps_per_second: 2
            # timesteps : 1, 2, ..., 80 (80 frames) -> 5, 10, ..., 80 (16 frames)
            pred_traj = pred_traj[:, :, 4::5, :]

            self.save_test_result(target_points, target_prob, pred_traj, batch)

    def plot_bad_case(self, batch, plot_variables, time_string):
        with torch.no_grad():
            gt_vector = batch.actor_future_vector
            gt_vector_mask = batch.actor_future_vector_mask
            gt_cluster_mask = batch.actor_cluster_mask
            # Check if agent exists in every cluster
            assert torch.all(gt_cluster_mask[:, 0] == 1)
            gt_vector_masked = gt_vector.masked_fill((gt_vector_mask == 0).unsqueeze(-1), 0)
            gt_traj = mc.gt_vector_to_gt_traj(gt_vector_masked)
            target_score_raw = plot_variables[0]
            traj_pred = plot_variables[1]

            if time_string == '3s':
                end_timestep_index = 29
            elif time_string == '5s':
                end_timestep_index = 49
            elif time_string == '8s':
                end_timestep_index = 79
            miss, MR = mc.miss_judge(traj_pred, gt_traj[:, 0, :, :], end_timestep_index)
            minADE = mc.minADE_calc(traj_pred, gt_traj[:, 0, :, :], end_timestep_index)
            minFDE = mc.minFDE_calc(traj_pred, gt_traj[:, 0, :, :], end_timestep_index)

            B = target_score_raw.shape[0]
            D = batch.high_reso_dim[0]
            target_score_raw = F.softmax(target_score_raw.view(B, -1), dim=-1).view(B, D, D)
            data_list = batch.to_data_list()

            max_metric_with_index = []
            for i, data in enumerate(data_list):
                if not (torch.any(gt_vector_mask[i] == 1) and torch.any(gt_cluster_mask[i] == 1)):
                    continue
                if self.config['show _badcase']['only_plot_top_K_bad'] > 0:
                    if (self.config['show _badcase']['plot_with_metric'] == "minADE") \
                            and (minADE[i] >= self.config['show _badcase']['minADE_threshold']):
                        max_metric_with_index.append((minADE[i], i))
                    elif (self.config['show _badcase']['plot_with_metric'] == "minFDE") \
                            and (minFDE[i] >= self.config['show _badcase']['minFDE_threshold']):
                        max_metric_with_index.append((minFDE[i], i))
                    elif (self.config['show _badcase']['plot_with_metric'] == "miss") and miss[i]:
                        max_metric_with_index.append((minFDE[i], i))
                else:
                    data.pred_heatmap = target_score_raw[i]
                    data.pred_traj = traj_pred[i]
                    plt = plot_transformed_data(data.to('cpu'), show=True)
                    if self.config['show _badcase']['save_fig']:
                        plt.savefig(f'{self.config["test"]["plot_out_folder"]}/{data.raw_sample_name}.svg')
            if self.config['show _badcase']['only_plot_top_K_bad'] > 0:
                max_metric_with_index = sorted(max_metric_with_index, key=lambda x: (x[0], x[1]))
                for i in range(min(len(max_metric_with_index), self.config['show _badcase']['only_plot_top_K_bad'])):
                    data.pred_heatmap = target_score_raw[max_metric_with_index[i][1]]
                    data.pred_traj = traj_pred[max_metric_with_index[i][1]]
                    plt = plot_transformed_data(data.to('cpu'), show=True)
                    if self.config['show _badcase']['save_fig']:
                        plt.savefig(f'{self.config["test"]["plot_out_folder"]}/{data.raw_sample_name}.svg')

    def save_test_result(self, target_points, target_prob, traj, batch):
        sample_name: List[str] = batch.raw_sample_name

        target_points = target_points.cpu().numpy()
        target_prob = target_prob.cpu().numpy()
        traj = traj.cpu().numpy()

        for i in range(len(sample_name)):
            _, _, obs_id = sample_name[i].split('_')

            data: Dict = {
                'obs_id': obs_id,  # str
                'traj': traj[i, :, :, :],  # [M, T, 2]
                'traj_prob': target_prob[i, :, :],  # [M, 1]
            }
            save_pkl_file: str = self.config['test']['test_result_pkl_dir'] + '/' + sample_name[i] + '.pkl'
            with open(save_pkl_file, 'wb') as f:
                pkl.dump(data, f)

    def intention_loss(self, target_score_raw, gt_target_idx, gt_target_mask=None):
        # [B, S, 1] -> [B, S, 1]
        goals_score = F.log_softmax(target_score_raw, dim=1)

        # goals_score : [B, S, 1] -> [B, S]
        # gt_target_idx : [B, 1] -> [B]
        # negtive likelihood loss: [B, I], [B] -> [B]
        loss = F.nll_loss(goals_score.squeeze(-1), gt_target_idx.squeeze(-1), reduction='none')

        # mask out the goals with invalid idx
        if gt_target_mask is not None:
            # [B] -> [B] with masked
            loss = loss.masked_select(gt_target_mask == 1)

        # [B] -> [1]
        loss = loss.mean()
        return loss

    def traj_loss(self, traj, gt_traj, gt_traj_mask=None):
        # traj: [B, 1, T*2] -> [B, T*2]
        # gt_traj: [B, T, 2] -> [B, T*2]
        # smooth l1 loss: [B, T*2] [B, T*2] -> [B, T*2]
        loss = F.smooth_l1_loss(traj.squeeze(1),
                                gt_traj.view(traj.shape[0], traj.shape[1] * traj.shape[2]),
                                beta=1.0,
                                reduction='none')

        # mask out the goals with invalid idx
        if gt_traj_mask is not None:
            # [B, T] -> [B, T*2]
            B, T = gt_traj_mask.shape
            gt_traj_mask = torch.stack([gt_traj_mask, gt_traj_mask], dim=-1).reshape((B, T * 2))
            # [B, T*2] -> [B * T*2] with masked
            loss = loss.masked_select(gt_traj_mask == 1)

        # [B] -> [1]
        loss = loss.mean()
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['optim']['init_lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=self.config['optim']['step_size'],
                                              gamma=self.config['optim']['step_factor'])

        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_set = PickleDataset(Path(self.config['training']['sample_database_folder']),
                                  DenseTNTTransform(self.config['transform']['train']))
        train_loader = DataLoader(train_set,
                                  batch_size=self.config['training']['batch_size'],
                                  shuffle=True,
                                  collate_fn=AgentClosureBatch.from_data_list,
                                  num_workers=self.config['training']['loader_worker_num'],
                                  drop_last=True,
                                  pin_memory=True)

        return train_loader

    def val_dataloader(self):
        val_set = PickleDataset(Path(self.config['validation']['sample_database_folder']))
        val_loader = DataLoader(val_set,
                                batch_size=self.config['validation']['batch_size'],
                                shuffle=False,
                                collate_fn=AgentClosureBatch.from_data_list,
                                num_workers=self.config['validation']['loader_worker_num'],
                                drop_last=True,
                                pin_memory=True)

        return val_loader

    def test_dataloader(self):
        test_set = PickleDataset(Path(self.config['test']['sample_database_folder']))
        test_loader = DataLoader(test_set,
                                 batch_size=self.config['test']['batch_size'],
                                 shuffle=False,
                                 collate_fn=AgentClosureBatch.from_data_list,
                                 num_workers=self.config['test']['loader_worker_num'],
                                 drop_last=False,
                                 pin_memory=True)

        return test_loader

    def export_onnx(self, path):
        self.eval()

        for _, batch in enumerate(self.val_dataloader()):
            vectors = [batch.actor_vector, batch.map_vector]
            vector_masks = [batch.actor_vector_mask, batch.map_vector_mask]
            cluster_masks = [batch.actor_cluster_mask, batch.map_cluster_mask]
            target_info = [batch.candi_target_points]
            dummy_input = (vectors, vector_masks, cluster_masks, target_info)
            break

        input_names = [
            'actor_vector', 'map_vector', 'actor_vector_mask', 'map_vector_mask', 'actor_cluster_mask',
            'map_cluster_mask', 'candi_target_points'
        ]
        output_names = ['target_points', 'target_prob', 'pred_traj']

        torch.onnx.export(self.infer_model,
                          dummy_input,
                          path,
                          input_names=input_names,
                          output_names=output_names,
                          verbose=True,
                          opset_version=11)


if __name__ == '__main__':
    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print(f'{config_file} not exists!!')
    with open(config_file, 'rb') as f:
        config = yaml.safe_load(f)

    torch.manual_seed(42)

    predictor = DenseTNTPredictor(config)

    dataloader = predictor.train_dataloader()

    for batch in dataloader:

        vectors = [batch.actor_vector, batch.map_vector]
        vector_masks = [batch.actor_vector_mask, batch.map_vector_mask]
        cluster_masks = [batch.actor_cluster_mask, batch.map_cluster_mask]
        target_info = [batch.candi_target_points, batch.gt_target_grid_center]

        target_score_raws, traj = predictor.model(vectors, vector_masks, cluster_masks, target_info)

        intention_loss = predictor.intention_loss(target_score_raws[2], batch.gt_target_grid_idx[:, 2],
                                                  batch.gt_target_point_mask[:, 2])
        traj_loss = predictor.traj_loss(traj, batch.gt_traj, batch.gt_traj_mask)

        print(target_score_raws[2].shape)
        print(traj.shape)
        print(intention_loss)
        print(traj_loss)
        break
