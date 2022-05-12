# Copyright (c) 2021 Li Auto Company. All rights reserved.

import pickle as pkl
from pathlib import Path
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from core.utils.utils import local2global, pose2rt
from datasets.transform.agent_closure import AgentClosureBatch
from datasets.transform.dataset import PickleDataset
from datasets.transform.sample_transform.multi_path_pro_transform import MultiPathProTransform
from module.context_encoder.context_encoder import ContextEncoder
from module.intention_decoder.intention_decoder import IntentionDecoder
from module.intention_sampler.intention_sampler import IntentionSampler
from module.trajectory_decoder.trajectory_decoder import TrajectoryDecoder
from torch.utils.data import DataLoader


class MultiPathPro(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder_layer = ContextEncoder.Create(config['context_encoder'])

        if config['decoder_type'] == 'dense_tnt_decoder':
            self.interaction_layer = IntentionDecoder.Create(config['interaction_decoder'])
            self.decoder_layer = TrajectoryDecoder.Create(config['trajectory_decoder'])
        elif config['decoder_type'] == 'multi_path_decoder':
            pass
        else:
            raise NotImplementedError

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
        # list[[B, C, L, F]], list[[B, C, L]], list[[B, C]] -> [B, 1, F' + F' + F_l]
        global_mcg_ctx = self.encoder_layer(vectors, vector_masks, cluster_masks)

        # [B, S, 2], [B, 1, F' + F' + F_l] -> [ [B, S, 1] x 3 ], predict feature for each goals 3s, 5s, 8s
        target_score_raws = self.interaction_layer(target_info[0], global_mcg_ctx)

        # [B, 1, 2], [B, 1, F' + F' + F_l] -> [B, 1, T*2]
        # predict the target goal trajectory and target_info[1][:, -1, :] is the 8s target
        traj = self.decoder_layer(target_info[1][:, [-1], :], global_mcg_ctx)

        # [ [B, S, 1] x 3 ], [B, 1, T*2], T is the number of trajectory points (such as 30 or 50 ...)
        return target_score_raws, traj


class MultiPathProInfer(nn.Module):

    def __init__(self, model, config):
        super().__init__()
        self.model: MultiPathPro = model
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
        # list[[B, C, L, F]], list[[B, C, L]], list[[B, C]] -> [B, 1, F' + F' + F_l]
        global_mcg_ctx = self.model.encoder_layer(vectors, vector_masks, cluster_masks)

        # [B, S, 2], [B, 1, F' + F' + F_l] -> [ [B, S, 1] x 3 ], predict feature for each goals 3s, 5s, 8s
        target_score_raws = self.model.interaction_layer(target_info[0], global_mcg_ctx)

        # [B, S, 1], [B, S, 2] -> [B, M, 2], [B, M, 1], M is modality number and target_score_raws[-1] is the 8s target
        teacher_points, target_prob = self.intention_sampler(target_score_raws[-1], target_info[0])

        # [B, M, 2], [B, 1, F' + F' + F_l] -> [B, M, T*2], predict the target goal trajectory
        traj = self.model.decoder_layer(teacher_points, global_mcg_ctx)

        # [B, M, 2], [B, M, 1], [B, M, T*2]
        return teacher_points, target_prob, traj


class MultiPathProPredictor(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = MultiPathPro(config['model'])
        self.infer_model = MultiPathProInfer(self.model, config['model'])

    def forward(self, vectors, vector_masks, cluster_masks, target_info):
        return self.model(vectors, vector_masks, cluster_masks, target_info)

    def training_step(self, batch, batch_idx):
        vectors = [batch.actor_vector, batch.actor_diff_vector, batch.map_vector]
        vector_masks = [batch.actor_vector_mask, batch.actor_diff_vector_mask, batch.map_vector_mask]
        cluster_masks = [batch.actor_cluster_mask, batch.actor_diff_cluster_mask, batch.map_cluster_mask]
        target_info = [batch.candi_target_points, batch.gt_target_grid_center]

        # [ [B, S, 1] x 3 ], [B, 1, T*2], S is the number of candicate goals
        target_score_raws, traj = self.model(vectors, vector_masks, cluster_masks, target_info)

        intention_3s_loss = self.intention_loss(target_score_raws[0], batch.gt_target_grid_idx[:, 0],
                                                batch.gt_target_point_mask[:, 0])
        intention_5s_loss = self.intention_loss(target_score_raws[1], batch.gt_target_grid_idx[:, 1],
                                                batch.gt_target_point_mask[:, 1])
        intention_8s_loss = self.intention_loss(target_score_raws[2], batch.gt_target_grid_idx[:, 2],
                                                batch.gt_target_point_mask[:, 2])
        traj_loss = self.traj_loss(traj, batch.gt_traj, batch.gt_traj_mask, self.config['model']['traj_downsample'])

        loss = self.config['loss']['intention_3s_loss_wgt'] * intention_3s_loss + \
            self.config['loss']['intention_5s_loss_wgt'] * intention_5s_loss + \
            self.config['loss']['intention_8s_loss_wgt'] * intention_8s_loss + \
            self.config['loss']['traj_loss_wgt'] * traj_loss

        self.log('train_intention_8s_loss', intention_8s_loss)
        self.log('train_traj_loss', traj_loss)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        vectors = [batch.actor_vector, batch.actor_diff_vector, batch.map_vector]
        vector_masks = [batch.actor_vector_mask, batch.actor_diff_vector_mask, batch.map_vector_mask]
        cluster_masks = [batch.actor_cluster_mask, batch.actor_diff_cluster_mask, batch.map_cluster_mask]
        target_info = [batch.candi_target_points, batch.gt_target_grid_center]

        # [ [B, S, 1] x 3 ], [B, 1, T*2], S is the number of candicate goals
        target_score_raws, traj = self.model(vectors, vector_masks, cluster_masks, target_info)

        intention_3s_loss = self.intention_loss(target_score_raws[0], batch.gt_target_grid_idx[:, 0],
                                                batch.gt_target_point_mask[:, 0])
        intention_5s_loss = self.intention_loss(target_score_raws[1], batch.gt_target_grid_idx[:, 1],
                                                batch.gt_target_point_mask[:, 1])
        intention_8s_loss = self.intention_loss(target_score_raws[2], batch.gt_target_grid_idx[:, 2],
                                                batch.gt_target_point_mask[:, 2])
        traj_loss = self.traj_loss(traj, batch.gt_traj, batch.gt_traj_mask, self.config['model']['traj_downsample'])

        loss = self.config['loss']['intention_3s_loss_wgt'] * intention_3s_loss + \
            self.config['loss']['intention_5s_loss_wgt'] * intention_5s_loss + \
            self.config['loss']['intention_8s_loss_wgt'] * intention_8s_loss + \
            self.config['loss']['traj_loss_wgt'] * traj_loss

        self.log('val_intention_8s_loss', intention_8s_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_traj_loss', traj_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        vectors = [batch.actor_vector, batch.actor_diff_vector, batch.map_vector]
        vector_masks = [batch.actor_vector_mask, batch.actor_diff_vector_mask, batch.map_vector_mask]
        cluster_masks = [batch.actor_cluster_mask, batch.actor_diff_cluster_mask, batch.map_cluster_mask]
        target_info = [batch.candi_target_points]

        # [B, M, 2], [B, M, 1], [B, M, T*2]
        target_points, target_prob, pred_traj = self.infer_model(vectors, vector_masks, cluster_masks, target_info)

        # Transform agent coord to global coord
        # rotation matrix
        agent_origin, agent_rotm = pose2rt(batch.agent_pose)
        # convert to global frame
        # T =  T*2 / 2
        time_steps = pred_traj.shape[-1] // 2
        # [B, M, T*2] -> [B, M, T, 2]
        pred_traj = local2global(pred_traj.reshape(pred_traj.shape[0], pred_traj.shape[1], time_steps, 2), agent_origin,
                                 agent_rotm)

        # Dont use downsample, because the traj is already downsampled in model forward
        # Downsample the trajectory with Waymo submission config - prediction_steps_per_second: 2
        # timesteps : 1, 2, ..., 80 (80 frames) -> 5, 10, ..., 80 (16 frames)
        # pred_traj = pred_traj[:, :, 4::5, :]

        self.save_test_result(target_points, target_prob, pred_traj, batch)

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

    def traj_loss(self, traj, gt_traj, gt_traj_mask=None, downsample_rate=1):
        assert downsample_rate >= 1
        # downsample_rate = 1, its not downsample
        if downsample_rate > 1:
            start_idx = downsample_rate - 1
            # If downsample rate == 5, the predicted trajectory is 0.5s/step, 80 -> 16
            gt_traj = gt_traj[:, start_idx::downsample_rate, :]
            gt_traj_mask = gt_traj_mask[:, start_idx::downsample_rate]

        # traj: [B, 1, T*2] -> [B, T*2]
        # gt_traj: [B, T, 2] -> [B, T*2]
        # smooth l1 loss: [B, T*2] [B, T*2] -> [B, T*2]
        loss = F.smooth_l1_loss(traj.squeeze(1),
                                gt_traj.reshape(traj.shape[0], traj.shape[1] * traj.shape[2]),
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
                                  MultiPathProTransform(self.config['transform']['train']))
        train_loader = DataLoader(train_set,
                                  batch_size=self.config['training']['batch_size'],
                                  shuffle=True,
                                  collate_fn=AgentClosureBatch.from_data_list,
                                  num_workers=self.config['training']['loader_worker_num'],
                                  drop_last=True,
                                  pin_memory=True)

        return train_loader

    def val_dataloader(self):
        val_set = PickleDataset(Path(self.config['validation']['sample_database_folder']),
                                MultiPathProTransform(self.config['transform']['val']))
        val_loader = DataLoader(val_set,
                                batch_size=self.config['validation']['batch_size'],
                                shuffle=False,
                                collate_fn=AgentClosureBatch.from_data_list,
                                num_workers=self.config['validation']['loader_worker_num'],
                                drop_last=True,
                                pin_memory=True)

        return val_loader

    def test_dataloader(self):
        test_set = PickleDataset(Path(self.config['test']['sample_database_folder']),
                                 MultiPathProTransform(self.config['transform']['test']))
        test_loader = DataLoader(test_set,
                                 batch_size=self.config['test']['batch_size'],
                                 shuffle=False,
                                 collate_fn=AgentClosureBatch.from_data_list,
                                 num_workers=self.config['test']['loader_worker_num'],
                                 drop_last=False,
                                 pin_memory=True)

        return test_loader
