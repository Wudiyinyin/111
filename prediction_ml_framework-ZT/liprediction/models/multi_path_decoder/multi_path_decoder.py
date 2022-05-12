# Copyright (c) 2021 Li Auto Company. All rights reserved.

import pickle as pkl
from pathlib import Path
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from core.utils.loss import two_dimension_gaussian_nll_loss
from core.utils.utils import local2global, pose2rt
from datasets.transform.agent_closure import AgentClosureBatch
from datasets.transform.dataset import PickleDataset
from datasets.transform.sample_transform.multi_path_pro_transform import MultiPathProTransform
from module.context_encoder.context_encoder import ContextEncoder
from module.intention_decoder.intention_decoder import IntentionDecoder
from module.intention_sampler.intention_sampler import IntentionSampler
from module.trajectory_decoder.trajectory_decoder import TrajectoryDecoder
from torch.utils.data import DataLoader


class MultiPathDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder_layer = ContextEncoder.Create(config['context_encoder'])
        self.decoder_layer = TrajectoryDecoder.Create(config['trajectory_decoder'])

    def forward(self, vectors, vector_masks, cluster_masks):
        '''
            B: batch_size, C: max_polyline_len(context), L: max_vector_len, F: feature_size

            vectors: List Tensor [[B, C, L, F], [B, C, L, F]],
                respectively for Obstacle Vector and Lane Vector (or Map Vector)
            vector_masks: List Tensor [[B, C, L], [B, C, L]]
            cluster_masks: List Tensor [[B, C], [B, C]]
        '''
        # list[[B, C, L, F]], list[[B, C, L]], list[[B, C]] -> [B, 1, F' + F' + F_l]
        global_mcg_ctx = self.encoder_layer(vectors, vector_masks, cluster_masks)

        # [B, 1, F' + F' + F_l] -> [B, M, 1], [B, M, T, 2], [B, M, T, 2], [B, M, T, 1], predict the modality trajectory
        traj_prob, traj_pt, traj_sigma, traj_rho = self.decoder_layer(global_mcg_ctx)

        # [B, M, 1], [B, M, T, 2], [B, M, T, 2], [B, M, T, 1]
        # T is the number of trajectory points (such as 30 or 50 ...)
        return traj_prob, traj_pt, traj_sigma, traj_rho


class MultiPathDecoderInfer(nn.Module):

    def __init__(self, model, config):
        super().__init__()
        self.model: MultiPathDecoder = model
        self.config = config

    def forward(self, vectors, vector_masks, cluster_masks):
        '''
            B: batch_size, C: max_polyline_len(context), L: max_vector_len, F: feature_size

            vectors: List Tensor [[B, C, L, F], [B, C, L, F]],
                respectively for Obstacle Vector and Lane Vector (or Map Vector)
            vector_masks: List Tensor [[B, C, L], [B, C, L]]
            cluster_masks: List Tensor [[B, C], [B, C]]
        '''
        # list[[B, C, L, F]], list[[B, C, L]], list[[B, C]] -> [B, 1, F' + F' + F_l]
        global_mcg_ctx = self.model.encoder_layer(vectors, vector_masks, cluster_masks)

        # [B, 1, F' + F' + F_l] -> [B, M, 1], [B, M, T, 2], [B, M, T, 2], [B, M, T, 1], predict the modality trajectory
        traj_prob, traj_pt, traj_sigma, traj_rho = self.decoder_layer(global_mcg_ctx)

        # [B, M, 1], [B, M, T, 2], [B, M, T, 2], [B, M, T, 1]
        # T is the number of trajectory points (such as 30 or 50 ...)
        return traj_prob, traj_pt, traj_sigma, traj_rho


class MultiPathDecoderPredictor(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = MultiPathDecoder(config['model'])
        self.infer_model = MultiPathDecoderInfer(self.model, config['model'])

    def forward(self, vectors, vector_masks, cluster_masks, target_info):
        return self.model(vectors, vector_masks, cluster_masks, target_info)

    def training_step(self, batch, batch_idx):
        vectors = [batch.actor_vector, batch.actor_diff_vector, batch.map_vector]
        vector_masks = [batch.actor_vector_mask, batch.actor_diff_vector_mask, batch.map_vector_mask]
        cluster_masks = [batch.actor_cluster_mask, batch.actor_diff_cluster_mask, batch.map_cluster_mask]

        # [B, M, 1], [B, M, T, 2], [B, M, T, 2], [B, M, T, 1]
        # T is the number of trajectory points (such as 30 or 50 ...)
        traj_prob, traj_pt, traj_sigma, traj_rho = self.model(vectors, vector_masks, cluster_masks)

        assert self.config['model']['traj_downsample'] >= 1
        # downsample_rate = 1, its not downsample
        downsample_rate = self.config['model']['traj_downsample']
        if downsample_rate > 1:
            start_idx = downsample_rate - 1
            # If downsample rate == 5, the predicted trajectory is 0.5s/step, 80 -> 16
            gt_traj = batch.gt_traj[:, start_idx::downsample_rate, :]
            gt_traj_mask = batch.gt_traj_mask[:, start_idx::downsample_rate]

        # [B, M, 1], [B, M, T, 2], [B, T, 2] -> [1], [B]
        prob_loss, gt_traj_idx = self.prob_loss(traj_prob, traj_pt, gt_traj)
        # # [B], [B, M, T, 2], [B, M, T, 2], [B, M, T, 1], [B, T] -> [1]
        traj_gmm_loss = self.traj_gmm_loss(gt_traj_idx, traj_pt, traj_sigma, traj_rho, gt_traj, gt_traj_mask)
        # [B], [B, M, T, 2], [B, M, T, 2], [B, M, T, 1], [B, T] -> [1]
        traj_pt_loss = self.traj_pt_loss(gt_traj_idx, traj_pt, gt_traj, gt_traj_mask)

        loss = self.config['loss']['prob_loss_wgt'] * prob_loss \
            + self.config['loss']['traj_gmm_loss_wgt'] * traj_gmm_loss \
            + self.config['loss']['traj_pt_loss_wgt'] * traj_pt_loss

        self.log('train_prob_loss', prob_loss)
        self.log('train_traj_gmm_loss', traj_gmm_loss)
        self.log('train_traj_pt_loss', traj_pt_loss)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        vectors = [batch.actor_vector, batch.actor_diff_vector, batch.map_vector]
        vector_masks = [batch.actor_vector_mask, batch.actor_diff_vector_mask, batch.map_vector_mask]
        cluster_masks = [batch.actor_cluster_mask, batch.actor_diff_cluster_mask, batch.map_cluster_mask]

        # [B, M, 1], [B, M, T, 2], [B, M, T, 2], [B, M, T, 1]
        # T is the number of trajectory points (such as 30 or 50 ...)
        traj_prob, traj_pt, traj_sigma, traj_rho = self.model(vectors, vector_masks, cluster_masks)

        assert self.config['model']['traj_downsample'] >= 1
        # downsample_rate = 1, its not downsample
        downsample_rate = self.config['model']['traj_downsample']
        if downsample_rate > 1:
            start_idx = downsample_rate - 1
            # If downsample rate == 5, the predicted trajectory is 0.5s/step, 80 -> 16
            gt_traj = batch.gt_traj[:, start_idx::downsample_rate, :]
            gt_traj_mask = batch.gt_traj_mask[:, start_idx::downsample_rate]

        # [B, M, 1], [B, M, T, 2], [B, T, 2] -> [1], [B]
        prob_loss, gt_traj_idx = self.prob_loss(traj_prob, traj_pt, gt_traj)
        # # [B], [B, M, T, 2], [B, M, T, 2], [B, M, T, 1], [B, T] -> [1]
        traj_gmm_loss = self.traj_gmm_loss(gt_traj_idx, traj_pt, traj_sigma, traj_rho, gt_traj, gt_traj_mask)
        # [B], [B, M, T, 2], [B, M, T, 2], [B, M, T, 1], [B, T] -> [1]
        traj_pt_loss = self.traj_pt_loss(gt_traj_idx, traj_pt, gt_traj, gt_traj_mask)

        loss = self.config['loss']['prob_loss_wgt'] * prob_loss \
            + self.config['loss']['traj_gmm_loss_wgt'] * traj_gmm_loss \
            + self.config['loss']['traj_pt_loss_wgt'] * traj_pt_loss

        self.log('val_prob_loss', prob_loss)
        self.log('val_traj_gmm_loss', traj_gmm_loss)
        self.log('val_traj_pt_loss', traj_pt_loss)
        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        vectors = [batch.actor_vector, batch.actor_diff_vector, batch.map_vector]
        vector_masks = [batch.actor_vector_mask, batch.actor_diff_vector_mask, batch.map_vector_mask]
        cluster_masks = [batch.actor_cluster_mask, batch.actor_diff_cluster_mask, batch.map_cluster_mask]

        # [B, M, 1], [B, M, T*2], [B, M, T*3], T is the number of trajectory points (such as 30 or 50 ...)
        traj_prob, traj_pt, traj_var = self.model(vectors, vector_masks, cluster_masks)

        # Transform agent coord to global coord
        # rotation matrix
        agent_origin, agent_rotm = pose2rt(batch.agent_pose)
        # convert to global frame
        # T =  T*2 / 2
        time_steps = traj_pt.shape[-1] // 2
        # [B, M, T*2] -> [B, M, T, 2]
        pred_traj = local2global(traj_pt.reshape(traj_pt.shape[0], traj_pt.shape[1], time_steps, 2), agent_origin,
                                 agent_rotm)

        # Downsample the trajectory with Waymo submission config - prediction_steps_per_second: 2
        # timesteps : 1, 2, ..., 80 (80 frames) -> 5, 10, ..., 80 (16 frames)
        # pred_traj = pred_traj[:, :, 4::5, :]

        self.save_test_result(traj_prob, pred_traj, batch)

    def save_test_result(self, target_prob, traj, batch):
        sample_name: List[str] = batch.raw_sample_name

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

    def prob_loss(self, traj_prob, traj, gt_traj):
        """Trajectory probability loss

        Args:
            traj_prob: [B, M, 1]
            traj: [B, M, T, 2]
            gt_traj: [B, T, 2]
        """

        # [B, T, 2] -> [B, T*2] -> [B, 1, T*2]
        gt_traj = gt_traj.reshape(gt_traj.shape[0], -1).unsqueeze(1)
        # [B, M, T*2] - [B, 1, T*2] -> [B, M, T*2]
        traj_diff = traj.reshape(traj.shape[0], traj.shape[1], -1) - gt_traj
        # [B, M, T*2] -> [B, M]
        traj_mse = torch.mean(traj_diff**2, dim=-1)
        # [B, M] -> [B]
        _, gt_traj_idx = torch.min(traj_mse, dim=-1)

        # [B, M, 1] -> [B, M, 1]
        traj_score = F.log_softmax(traj_prob, dim=1)

        # traj_score : [B, M, 1] -> [B, M]
        # gt_traj_idx : [B]
        # negtive likelihood loss: [B, M], [B] -> [B] -> [1]
        loss = F.nll_loss(traj_score.squeeze(-1), gt_traj_idx, reduction='mean')

        # [1], [B]
        return loss, gt_traj_idx

    def traj_pt_loss(self, gt_traj_idx, traj_pt, gt_traj, gt_traj_mask=None):
        """Trajectory point loss

        Args:
            gt_traj_idx: [B]
            traj_pt: [B, M, T, 2]
            gt_traj: [B, T, 2]
            gt_traj_mask: [B, T]
        """
        B, _, _, _ = traj_pt.shape
        # [B, M, T, 2] -> [B, T, 2]
        traj_pt = traj_pt[range(B), gt_traj_idx, :, :]

        # smooth l1 loss: [B, T, 2] [B, T, 2] -> [B, T, 2]
        loss = F.smooth_l1_loss(traj_pt, gt_traj, beta=1.0, reduction='none')

        # mask out the goals with invalid idx
        if gt_traj_mask is not None:
            # [B, T] -> [B, T, 2]
            gt_traj_mask = torch.stack([gt_traj_mask, gt_traj_mask], dim=-1)
            # [B, T, 2] -> [B*T*2] with masked
            loss = loss.masked_select(gt_traj_mask == 1)

        # [B] -> [1]
        loss = loss.mean()
        return loss

    def traj_gmm_loss(self, gt_traj_idx, traj_pt, traj_sigma, traj_rho, gt_traj, gt_traj_mask=None):
        """Trajectory GMM loss

        Args:
            gt_traj_idx: [B]
            traj_pt: [B, M, T, 2]
            traj_sigma: [B, M, T, 2]
            traj_rho: [B, M, T, 1]
            gt_traj: [B, T, 2]
            gt_traj_mask: [B, T]
        """

        B, _, _, _ = traj_pt.shape
        # gaussian nll loss: [B, T, 2], [B, T, 2], [B, T, 2], [B, T, 1] -> [B, T, 1]
        loss = two_dimension_gaussian_nll_loss(traj_pt[range(B), gt_traj_idx, :, :],
                                               gt_traj,
                                               traj_sigma[range(B), gt_traj_idx, :, :],
                                               traj_rho[range(B), gt_traj_idx, :, :],
                                               reduction='none')

        # mask out the goals with invalid idx
        if gt_traj_mask is not None:
            # [B, T, 1] -> [B, T] -> [B * T] with masked
            loss = loss.squeeze(-1).masked_select(gt_traj_mask == 1)

        loss.nan_to_num_()

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
