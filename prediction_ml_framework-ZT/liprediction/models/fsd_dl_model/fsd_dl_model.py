# Copyright (c) 2022 Li Auto Company. All rights reserved.

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
from module.context_encoder.context_encoder import ContextEncoder
from module.intention_decoder.intention_decoder import IntentionDecoder
from module.intention_sampler.intention_sampler import IntentionSampler
from module.trajectory_decoder.trajectory_decoder import TrajectoryDecoder
from torch.utils.data import DataLoader


class FSDModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = ContextEncoder.Create(config['context_encoder'])
        self.intention_decoder = IntentionDecoder.Create(config['intention_decoder'])
        self.traj_decoder = TrajectoryDecoder.Create(config['trajectory_decoder'])

    def forward(self, vectors, vector_masks, cluster_masks):
        '''
            B: batch_size, C: max_polyline_len(context), L: max_vector_len, F: feature_size

            vectors: List Tensor [[B, C, L, F], [B, C, L, F]],
                respectively for Obstacle Vector and Lane Vector (or Map Vector)
            vector_masks: List Tensor [[B, C, L], [B, C, L]]
            cluster_masks: List Tensor [[B, C], [B, C]]
        '''
        # list[[B, C, L, F]], list[[B, C, L]], list[[B, C]] -> [B, 1, F']
        agent_all_feature = self.encoder(vectors, vector_masks, cluster_masks)

        # [B, 1, F'] -> [ [B, S, 4] x 3 ], [ [B, S, 1] x 3 ] predict cutin_cls and is_cutin  1s, 2s, 3s
        cutin_cls_prob_list, is_cutin_prob_list = self.intention_decoder(agent_all_feature)

        # [B, 1, F'] -> [B, T, 2]
        traj = self.traj_decoder(agent_all_feature)

        return cutin_cls_prob_list, is_cutin_prob_list, traj


class MultiPathProPredictor(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = FSDModel(config['model'])

    def forward(self, vectors, vector_masks, cluster_masks):
        return self.model(vectors, vector_masks, cluster_masks)

    def training_step(self, batch, batch_idx):
        vectors = [batch.actor_vector, batch.map_vector]
        vector_masks = [batch.actor_vector_mask, batch.map_vector_mask]
        cluster_masks = [batch.actor_cluster_mask, batch.map_cluster_mask]

        # [ [B, S, 4] x 3,  [B, S, 1] x 3], [B, 1, T, 2], S is the number of candicate goals
        cutin_cls_prob_list, is_cutin_prob_list, traj = self.model(vectors, vector_masks, cluster_masks)

        cutin_cls_1s_loss, is_cutin_1s_loss = self.intention_loss(cutin_cls_prob_list[0], is_cutin_prob_list[0],
                                                                  batch.cutin_cls_label[:, 0], batch.is_cutin_label[:,
                                                                                                                    0])
        cutin_cls_2s_loss, is_cutin_2s_loss = self.intention_loss(cutin_cls_prob_list[1], is_cutin_prob_list[1],
                                                                  batch.cutin_cls_label[:, 1], batch.is_cutin_label[:,
                                                                                                                    1])
        cutin_cls_3s_loss, is_cutin_3s_loss = self.intention_loss(cutin_cls_prob_list[2], is_cutin_prob_list[2],
                                                                  batch.cutin_cls_label[:, 2], batch.is_cutin_label[:,
                                                                                                                    2])
        traj_loss = self.traj_loss(traj, batch.traj_label)

        cutin_cls_loss = cutin_cls_1s_loss + cutin_cls_2s_loss + cutin_cls_3s_loss
        is_cutin_loss = is_cutin_1s_loss + is_cutin_2s_loss + is_cutin_3s_loss

        loss = self.config['loss']['cutin_cls_loss_wgt'] * cutin_cls_loss + \
            self.config['loss']['is_cutin_loss_wgt'] * is_cutin_loss + \
            self.config['loss']['traj_reg_loss_wgt'] * traj_loss

        self.log('train_cutin_cls_loss', cutin_cls_loss)
        self.log('train_is_cutin_loss', is_cutin_loss)
        self.log('train_traj_loss', traj_loss)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        vectors = [batch.actor_vector, batch.map_vector]
        vector_masks = [batch.actor_vector_mask, batch.map_vector_mask]
        cluster_masks = [batch.actor_cluster_mask, batch.map_cluster_mask]

        # [ [B, S, 4] x 3,  [B, S, 1] x 3], [B, 1, T, 2], S is the number of candicate goals
        cutin_cls_prob_list, is_cutin_prob_list, traj = self.model(vectors, vector_masks, cluster_masks)

        cutin_cls_1s_loss, is_cutin_1s_loss = self.intention_loss(cutin_cls_prob_list[0], is_cutin_prob_list[0],
                                                                  batch.cutin_cls_label[:, 0], batch.is_cutin_label[:,
                                                                                                                    0])
        cutin_cls_2s_loss, is_cutin_2s_loss = self.intention_loss(cutin_cls_prob_list[1], is_cutin_prob_list[1],
                                                                  batch.cutin_cls_label[:, 1], batch.is_cutin_label[:,
                                                                                                                    1])
        cutin_cls_3s_loss, is_cutin_3s_loss = self.intention_loss(cutin_cls_prob_list[2], is_cutin_prob_list[2],
                                                                  batch.cutin_cls_label[:, 2], batch.is_cutin_label[:,
                                                                                                                    2])
        traj_loss = self.traj_loss(traj, batch.traj_label)

        cutin_cls_loss = cutin_cls_1s_loss + cutin_cls_2s_loss + cutin_cls_3s_loss
        is_cutin_loss = is_cutin_1s_loss + is_cutin_2s_loss + is_cutin_3s_loss

        loss = self.config['loss']['cutin_cls_loss_wgt'] * cutin_cls_loss + \
            self.config['loss']['is_cutin_loss_wgt'] * is_cutin_loss + \
            self.config['loss']['traj_reg_loss_wgt'] * traj_loss

        self.log('val_cutin_cls_loss', cutin_cls_loss)
        self.log('val_is_cutin_loss', is_cutin_loss)
        self.log('val_traj_loss', traj_loss)
        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        vectors = [batch.actor_vector, batch.map_vector]
        vector_masks = [batch.actor_vector_mask, batch.map_vector_mask]
        cluster_masks = [batch.actor_cluster_mask, batch.map_cluster_mask]

        # [B, M, 2], [B, M, 1], [B, M, T*2]
        cutin_cls_prob_list, is_cutin_prob_list, traj = self.model(vectors, vector_masks, cluster_masks)

        # self.save_test_result(target_points, target_prob, pred_traj, batch)

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

    def intention_loss(self, cutin_cls_prob, is_cutin_cls_prob, cutin_cls_label, is_cutin_label):
        # [B, S, 1] -> [B, S, 1]
        cutin_cls_log_prob = torch.log(cutin_cls_prob)
        cutin_cls_loss = F.nll_loss(cutin_cls_log_prob, cutin_cls_label)

        is_cutin_loss = F.binary_cross_entropy(is_cutin_cls_prob, is_cutin_label)

        return cutin_cls_loss, is_cutin_loss

    def traj_loss(self, traj, gt_traj):
        loss = F.smooth_l1_loss(traj, gt_traj)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['optim']['init_lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=self.config['optim']['step_size'],
                                              gamma=self.config['optim']['step_factor'])

        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_set = PickleDataset(Path(self.config['training']['sample_database_folder']))
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
