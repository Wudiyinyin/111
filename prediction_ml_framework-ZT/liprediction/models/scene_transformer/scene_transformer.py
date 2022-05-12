# Copyright (c) 2021 Li Auto Company. All rights reserved.

import os
import pickle as pkl
import sys
from pathlib import Path
from typing import Dict, List

import onnx
import pytorch_lightning as pl
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import yaml
from core.utils.utils import local2global, pose2rt
from datasets.processer.feature_processer import (ACTOR_FEATURE_END_X, ACTOR_FEATURE_END_Y, ACTOR_FEATURE_HIDDEN_MASK,
                                                  ACTOR_FEATURE_TIME_EMBED, plot_transformed_data)
from datasets.transform.agent_closure import AgentClosureBatch
from datasets.transform.dataset import PickleDataset
from module.context_encoder.context_encoder import ContextEncoder
from module.intention_decoder.intention_decoder import IntentionDecoder
from module.intention_sampler.intention_sampler import IntentionSampler
from module.trajectory_decoder.trajectory_decoder import TrajectoryDecoder
from onnxsim import simplify
from torch.utils.data import DataLoader
from yamlinclude import YamlIncludeConstructor


def handle_hidden_mask(actor_vector, actor_vector_hidden_mask):
    B, A, T, Da = actor_vector.shape
    # handle hidden mask
    # [B, A, T] 0/1 -> [B, A, T, 1] 0/1 -> [B, A, T, Da-2] 0/1  1:valid 0:mask
    mask1 = actor_vector_hidden_mask.unsqueeze(-1).expand(-1, -1, -1, Da - 2)
    # [B, A, T, 2] 1, the last two feature 'TIME_EMBED, HIDDEN_MASK' should not be hidden
    mask2 = torch.ones((B, A, T, 2), dtype=torch.int32).type_as(mask1)
    # [B, A, T, Da-2] + [B, A, T, 2] -> [B, A, T, Da]
    hidden_mask = torch.cat((mask1, mask2), dim=-1)
    assert hidden_mask.shape == (B, A, T, Da)
    # [B, A, T, Da], [B, A, T, Da] -> [B, A, T, Da], set hidden element to 0
    return actor_vector.masked_fill((hidden_mask == 0), 0.0)


class SceneTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.context_encoder = ContextEncoder.Create(config['context_encoder'])
        self.traj_decoder = TrajectoryDecoder.Create(config['trajectory_decoder'])

    def forward(self, vectors, vectors_padding_mask, vectors_hidden_mask, target_info):
        '''
            vectors: List Tensor [[B, A, T, Da], [B, G, L, Dg]],
                respectively for Obstacle Vector and Lane Vector (or Map Vector)
            vector_padding_masks: List Tensor [[B, A, T], [B, G, L]]
            vector_hidden_masks: List Tensor [[B, A, T], [B, G, L]]
            target_info: list
                [0] cand_target_points: Candidate target point tensor [B, S, 2],
                    S is the number of goals (H_dim * W_dim)
                [1] gt_target_grid_center [B, I, 2], I is the number of goals [3s 5s 8s]
        '''

        # list[[B, A, T, D], [B, G, L, D]], list[[B, A, T], [B, G, L]], list[[B, A, T], [B, G, L]]
        # -> [B, A+1, T+1, D], [B, A+1, T+1]
        actor_feature, actor_mask = self.context_encoder(vectors, vectors_padding_mask, vectors_hidden_mask)

        # TODO(xlm) handle hidden mask only in decoder
        # but there is no valid future traj to encode when interence

        # [B, A+1, T+1, D], [B, A+1, T+1] -> traj:[B, F, A, T, 2], score:[B, F, A, 1]
        traj, score = self.traj_decoder(actor_feature, actor_mask)

        # [B, F, A, 1], [B, F, A, T, 2]
        return score, traj


class SceneTransformerInfer(nn.Module):

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, vectors, vectors_padding_mask, vectors_hidden_mask, target_info):
        '''
            vectors: List Tensor [[B, A, T, Da], [B, G, L, Dg]],
                respectively for Obstacle Vector and Lane Vector (or Map Vector)
            vector_padding_masks: List Tensor [[B, A, T], [B, G, L]]
            vector_hidden_masks: List Tensor [[B, A, T], [B, G, L]]
            target_info: list
                [0] cand_target_points: Candidate target point tensor [B, S, 2],
                    S is the number of goals (H_dim * W_dim)
                [1] gt_target_grid_center [B, I, 2], I is the number of goals [3s 5s 8s]
        '''

        return self.model(vectors, vectors_padding_mask, vectors_hidden_mask, target_info)


class SceneTransformerPredictor(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = SceneTransformer(config['model'])
        self.infer_model = SceneTransformerInfer(self.model, config['model'])
        self.loss_type = config['loss']['type']

    def forward(self, vectors, vectors_padding_mask, vectors_hidden_mask, target_info):
        return self.model(vectors, vectors_padding_mask, vectors_hidden_mask, target_info)

    def training_step(self, batch, batch_idx):
        vectors, vectors_padding_mask, vectors_hidden_mask, gt_traj, gt_traj_padding_mask = self.make_vectors(batch)
        target_info = []

        # score: [B, F, A, 1], traj:[B, F, A, T, 2] F is future_num
        score, traj = self.model(vectors, vectors_padding_mask, vectors_hidden_mask, target_info)

        if self.loss_type == "marginal":
            traj_loss, intention_loss = self.marginal_loss(score, traj, gt_traj, gt_traj_padding_mask)
        elif self.loss_type == "marginal_agent_only":
            traj_loss, intention_loss = self.marginal_agent_only_loss(score, traj, gt_traj, gt_traj_padding_mask)
        elif self.loss_type == "joint":
            traj_loss, intention_loss = self.joint_loss(score, traj, gt_traj, gt_traj_padding_mask)
        else:
            raise Exception(f'Not implemented {self.loss_type}')

        loss = self.config['loss']['intention_loss_wgt'] * intention_loss + \
            self.config['loss']['traj_loss_wgt'] * traj_loss

        self.log('train_intention_loss', intention_loss)
        self.log('train_traj_loss', traj_loss)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        vectors, vectors_padding_mask, vectors_hidden_mask, gt_traj, gt_traj_padding_mask = self.make_vectors(batch)
        target_info = []

        # score: [B, F, A, 1], traj:[B, F, A, T, 2] F is future_num
        score, traj = self.model(vectors, vectors_padding_mask, vectors_hidden_mask, target_info)

        if self.loss_type == "marginal":
            traj_loss, intention_loss = self.marginal_loss(score, traj, gt_traj, gt_traj_padding_mask)
        elif self.loss_type == "marginal_agent_only":
            traj_loss, intention_loss = self.marginal_agent_only_loss(score, traj, gt_traj, gt_traj_padding_mask)
        elif self.loss_type == "joint":
            traj_loss, intention_loss = self.joint_loss(score, traj, gt_traj, gt_traj_padding_mask)
        else:
            raise Exception(f'Not implemented {self.loss_type}')

        loss = self.config['loss']['intention_loss_wgt'] * intention_loss + \
            self.config['loss']['traj_loss_wgt'] * traj_loss

        self.log('val_intention_loss', intention_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_traj_loss', traj_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        vectors, vectors_padding_mask, vectors_hidden_mask, gt_traj, gt_traj_padding_mask = self.make_vectors(batch)
        target_info = []

        B, A, T_history, Da = batch.actor_vector.shape
        # actor_vector = vectors[0]
        # actor_vector_padding_mask = vectors_padding_mask[0]

        # score: [B, F, A, 1], traj:[B, F, A, T, 2] F is future_num
        score, traj = self.infer_model(vectors, vectors_padding_mask, vectors_hidden_mask, target_info)
        # [B, F, A, T, 2] -> [B, A, F, T, 2]
        traj = traj.transpose(1, 2)
        # [B, F, A, 1] -> [B, F, A, 1] along F axis
        score = torch.nn.functional.softmax(score, dim=1)
        # [B, F, A, 1] -> [B, A, F, 1]
        score = score.transpose(1, 2)
        # B, D = pix_score.shape[:2]
        # pix_score = torch.nn.functional.softmax(pix_score.view(B, -1), dim=-1).view(B, D, D)
        data_list = batch.to_data_list()

        if self.config['test']['type'] == 'svg':
            for i, data in enumerate(data_list):
                data.actor_cluster_mask  # shape=[1, A], euqual to batch.actor_cluster_mask[[i],...]
                for j in range(data.actor_cluster_mask.shape[-1]):
                    if data.actor_cluster_mask[0, j] == 0:
                        # print("skip ", i, j, data.actor_cluster_mask[0, j])
                        continue

                    # [B, A, F, T, 2] -> [F, T_future, 2]
                    data.pred_traj = traj[i][j][:, T_history:, :]
                    # [B, A, F, 1] -> [F, 1] -> [F]
                    data.pred_traj_score = score[i][j].squeeze(-1)

                    # [B, A, T_history, Da] + [B, A, T_future, Da] -> [B, A, T, Da]
                    # actor_vector = torch.cat([batch.actor_vector, batch.actor_future_vector], dim=-2)
                    # [B, A, T, Da] -> [T_future, 2]
                    data.pred_traj_gt = gt_traj[i][j][T_history:, :]
                    # [B, A, T_history] + [B, A, T_future] -> [B, A, T] 1:not_mask 0:mask
                    # actor_vector_padding_mask = torch.cat([batch.actor_vector_mask,
                    # batch.actor_future_vector_mask], dim=-1)
                    # [B, A, T] -> [T_future]
                    data.pred_traj_gt_mask = gt_traj_padding_mask[i][j][T_history:]

                    # delete following key to disable plot
                    if hasattr(data, 'pred_heatmap'):
                        del data.pred_heatmap
                    if hasattr(data, 'candi_target_points_x'):
                        del data.candi_target_points_x
                    if hasattr(data, 'actor_future_vector'):
                        del data.actor_future_vector
                    if hasattr(data, 'gt_traj'):
                        del data.gt_traj

                    plot_transformed_data(
                        data.to('cpu'),
                        show=True,
                        save_path=f'{self.config["test"]["test_result_pkl_dir"]}/{data.raw_sample_name}_{j}.svg')

        elif self.config['test']['type'] == 'pkl':
            # Transform agent coord to global coord
            # rotation matrix
            agent_origin, agent_rotm = pose2rt(batch.agent_pose)
            # convert to global frame
            # T =  T*2 / 2, 8s = 80
            # [B, A, F, T, 2] -> [B, F, T, 2]
            pred_traj = traj[:, 0, :, T_history:, :]
            pred_traj = local2global(pred_traj, agent_origin, agent_rotm)

            # Downsample the trajectory with Waymo submission config - prediction_steps_per_second: 2
            # timesteps : 1, 2, ..., 80 (80 frames) -> 5, 10, ..., 80 (16 frames)
            # [B, F, T, 2]
            pred_traj = pred_traj[:, :, 4::5, :]
            # [B, F, 2]
            target_points = pred_traj[:, :, -1, :]
            # [B, A, F, 1] -> [B, F, 1]
            target_prob = score[:, 0, :, :]
            # B, F, P = target_prob.shape
            # for i in range(B):
            #     for j in range(F):
            #         print(f"{i}-{j} ", target_prob[i, j, :], target_points[i, j, :], pred_traj[i, j, :, :])

            # [2, 2, 2, 2]
            # pred_traj = torch.FloatTensor([[[[1111, 1112], [1121, 1122]], [[1211, 1212], [1221, 1222]]],
            #                               [[[2111, 2112], [2121, 2122]], [[2211, 2212], [2221, 2222]]]])
            # [2, 2, 1]
            # target_prob = torch.FloatTensor([[[0.9], [0.1]], [[0.2], [0.3]]])

            # [B, F, 1] -> [B, F, 1], descending order
            sort_prob, sort_idx = torch.sort(target_prob, dim=1, descending=True)
            # sort_prob = target_prob.gather(1, sort_idx)
            # print("sort_val", sort_val.shape, sort_val)
            # print("sort_idx", sort_idx.shape, sort_idx)
            # print("target_prob", target_prob.shape, target_prob)
            # print("sort_prob", sort_prob.shape, sort_prob)

            # [B, F, 2]
            B, F, D = target_points.shape
            points_idx = sort_idx.view(B, F, 1).expand(-1, -1, D)
            # [B, F, 2]
            sort_points = target_points.gather(1, points_idx)
            # print("points_idx", points_idx.shape, points_idx)
            # print("target_points", target_points.shape, target_points)
            # print("sort_points", sort_points.shape, sort_points)

            # [B, F, T, 2]
            B, F, T, D = pred_traj.shape
            # indx = sort_idx.view(B, F, 1).expand(-1, -1, T * D).view(B, F, T, D)
            traj_idx = sort_idx.view(B, F, 1, 1).expand(-1, -1, T, D)
            # [B, F, T, 2]
            sort_traj = pred_traj.gather(1, traj_idx)
            # print("traj_idx", traj_idx.shape, traj_idx)
            # print("pred_traj", pred_traj.shape, pred_traj)
            # print("sort_traj", sort_traj.shape, sort_traj)
            # for i in range(B):
            #     for j in range(F):
            #         print(f">> {i}-{j} ", sort_prob[i, j, :], sort_points[i, j, :], sort_traj[i, j, :, :])

            self.save_test_result(sort_points, sort_prob, sort_traj, batch)
        else:
            raise Exception(f"unknow type {self.config['test']['type']}")

    def save_test_result(self, target_points, target_prob, traj, batch):
        '''
        Args:
            target_points: [B, M, 2]
            target_prob: [B, M, 1]
            traj: [B, M, T, 2]
        '''
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

    def make_vectors(self, batch):
        # actor_vector: [B, A, T_history, Da]  actor_vector_mask: [B, A, T_hisstory]
        # actor_future_vector:[B, A, T_future, Da] actor_future_vector_mask: [B, A, T_future]
        # [B, A, T_history, Da] + [B, A, T_future, Da] -> [B, A, T, Da]
        actor_vector = torch.cat([batch.actor_vector, batch.actor_future_vector], dim=-2)
        B, A, T, Da = actor_vector.shape
        # [B, A, T_history] + [B, A, T_future] -> [B, A, T] 1:not_mask 0:mask
        # if replace `batch.actor_future_vector_mask` with `torch.ones_like(batch.actor_future_vector_mask)`
        # will enable all the future vector to involved in attention process
        # actor_vector_padding_mask = torch.cat([batch.actor_vector_mask, batch.actor_future_vector_mask], dim=-1)
        actor_vector_padding_mask = torch.cat([
            batch.actor_vector_mask,
            torch.ones_like(batch.actor_future_vector_mask, dtype=torch.int32).type_as(batch.actor_future_vector_mask)
        ],
                                              dim=-1)
        # [B, A, T_history] + [B, A, T_future] -> [B, A, T] 1:not_mask 0:mask
        actor_vector_hidden_mask = torch.cat([
            torch.ones_like(batch.actor_vector_mask, dtype=torch.int32).type_as(batch.actor_vector_mask),
            torch.zeros_like(batch.actor_future_vector_mask, dtype=torch.int32).type_as(batch.actor_future_vector_mask)
        ],
                                             dim=-1)

        # [B, G, L, Dg]
        map_vector = batch.map_vector
        # [B, G, L] 1:not_mask 0:mask
        map_vector_padding_mask = batch.map_vector_mask
        # [B, G, L] 1:not_mask 0:mask
        map_vector_hidden_mask = torch.ones_like(batch.map_vector_mask, dtype=torch.int32)

        vectors = [actor_vector, map_vector]
        vectors_padding_mask = [actor_vector_padding_mask, map_vector_padding_mask]
        vectors_hidden_mask = [actor_vector_hidden_mask, map_vector_hidden_mask]

        # [B, A, T, Da] (where Da = [start_x, start_y, end_x, end_y, vector_type]) -> [B, A, T, 2]
        gt_traj = actor_vector[:, :, :, [ACTOR_FEATURE_END_X, ACTOR_FEATURE_END_Y]]
        # [B, A, T_history] + [B, A, T_future] -> [B, A, T] 1:not_mask 0:mask
        gt_traj_padding_mask = torch.cat([batch.actor_vector_mask, batch.actor_future_vector_mask], dim=-1)

        # handle hidden mask
        # actor_vector = vectors[0]
        # actor_vector_hidden_mask = vectors_hidden_mask[0]
        # torch.set_printoptions(profile="full")
        vectors[0] = handle_hidden_mask(vectors[0], vectors_hidden_mask[0])
        # mask: [B, A, T] -> [B, A, T, 1].float().type_as()
        vectors[0][:, :, :, [ACTOR_FEATURE_HIDDEN_MASK]] = vectors_hidden_mask[0].unsqueeze(-1).type_as(vectors[0])

        # hidden TIME_EMBED / HIDDEN_MASK
        # [T] -> [1, 1, T, 1] -> [B, A, T, 1]
        time_embed = torch.arange(T).type_as(vectors[0]).view(1, 1, T, 1).expand(B, A, -1, -1)
        vectors[0][:, :, :, [ACTOR_FEATURE_TIME_EMBED]] = time_embed

        return vectors, vectors_padding_mask, vectors_hidden_mask, gt_traj, gt_traj_padding_mask

    def marginal_loss(self, score, traj, gt_traj, gt_traj_padding_mask):
        """
        Args:
            score: [B, F, A, 1]
            traj [B, F, A, T, 2]
            gt_traj: [B, A, T, 2]
            gt_traj_padding_mask: [B, A, T]
        """
        # traj [B, F, A, T, 2], gt_traj: [B, A, T, 2] -> [B, F, A, T, 2] -> loss: [B, F, A, T, 2]
        loss = torch.nn.functional.smooth_l1_loss(traj,
                                                  gt_traj.unsqueeze(1).expand(*traj.shape),
                                                  beta=1.0,
                                                  reduction='none')

        # set the padding element in gt_traj to 0
        # mask: [B, A, T] 0/1 -> [B, A, T] True/False -> [B, 1, A, T] True/False -> [B, 1, A, T, 1] True/False
        # loss: [B, F, A, T, 2], [B, 1, A, T, 1] -> [B, F, A, T, 2]
        # loss = loss.masked_fill((gt_traj_padding_mask == 0).unsqueeze(1).unsqueeze(-1), 0.0)

        # mask: [B, A, T] -> [B, 1, A, T, 1]
        mask = gt_traj_padding_mask.unsqueeze(1).unsqueeze(-1)
        # loss: [B, F, A, T, 2], [B, 1, A, T, 1] -> [B, F, A, T, 2]
        loss = loss * mask

        # Now reduce across all timesteps and values to produce a tensor of shape [F, A]
        # [B, F, A, T, 2] -> [B, F, A]
        loss = torch.sum(loss, dim=(-2, -1), keepdim=False)

        # The marginal loss, we only apply the loss to the best trajectory
        # per agent (so min across the future dimension).
        # [B, F, A] -> val:[B, A], idx:[B, A]
        traj_marginal_loss, indices = torch.min(loss, dim=-2, keepdim=False)

        # Then sum over the agent dimension
        # [B, A] -> [B]
        traj_marginal_loss = torch.sum(traj_marginal_loss, dim=-1, keepdim=False)

        # Then mean over the batch dimension
        # [B] -> 1
        traj_marginal_loss = torch.mean(traj_marginal_loss, dim=-1, keepdim=False)

        # score loss
        # [B, F, A, 1] -> [B, F, A, 1] along F axis
        score = torch.nn.functional.log_softmax(score, dim=1)
        _B, _F, _A, _ = score.shape
        # score: [B, F, A, 1] -> [B, F, A] -> [B, A, F] -> [B*A, F]
        score = score.squeeze(-1).transpose(-2, -1).contiguous().view(-1, _F)
        # gt_idx: [B, A] -> [B*A]
        gt_idx = indices.view(-1)
        assert score.shape == (_B * _A, _F)
        assert gt_idx.shape == (_B * _A,)
        # negtive likelihood loss: [B*A, F], [B*A] -> [1]
        score_marginal_loss = torch.nn.functional.nll_loss(score, gt_idx, reduction='mean')

        return traj_marginal_loss, score_marginal_loss

    def marginal_agent_only_loss(self, score, traj, gt_traj, gt_traj_padding_mask):
        """
        Args:
            score: [B, F, A, 1]
            traj [B, F, A, T, 2]
            gt_traj: [B, A, T, 2]
            gt_traj_padding_mask: [B, A, T]
        """
        # select agent only
        score = score[:, :, [0], :]
        traj = traj[:, :, [0], :, :]
        gt_traj = gt_traj[:, [0], :, :]
        gt_traj_padding_mask = gt_traj_padding_mask[:, [0], :]
        B, F, A, T, D = traj.shape

        # traj [B, F, A, T, 2], gt_traj: [B, A, T, 2] -> [B, F, A, T, 2] -> loss: [B, F, A, T, 2]
        loss = torch.nn.functional.smooth_l1_loss(traj,
                                                  gt_traj.unsqueeze(1).expand(*traj.shape),
                                                  beta=1.0,
                                                  reduction='none')
        assert loss.shape == (B, F, A, T, D)

        # set the padding element in gt_traj to 0
        # mask: [B, A, T] 0/1 -> [B, A, T] True/False -> [B, 1, A, T] True/False -> [B, 1, A, T, 1] True/False
        # loss: [B, F, A, T, 2], [B, 1, A, T, 1] -> [B, F, A, T, 2]
        # loss = loss.masked_fill((gt_traj_padding_mask == 0).unsqueeze(1).unsqueeze(-1), 0.0)

        # mask: [B, A, T] -> [B, 1, A, T, 1]
        mask = gt_traj_padding_mask.unsqueeze(1).unsqueeze(-1)
        # loss: [B, F, A, T, 2], [B, 1, A, T, 1] -> [B, F, A, T, 2]
        loss = loss * mask
        assert loss.shape == (B, F, A, T, D)

        # Now reduce across all timesteps and values to produce a tensor of shape [F, A]
        # [B, F, A, T, 2] -> [B, F, A]
        loss = torch.sum(loss, dim=(-2, -1), keepdim=False)
        assert loss.shape == (B, F, A)

        # The marginal loss, we only apply the loss to the best trajectory
        # per agent (so min across the future dimension).
        # [B, F, A] -> val:[B, A], idx:[B, A]
        traj_marginal_loss, indices = torch.min(loss, dim=-2, keepdim=False)
        assert traj_marginal_loss.shape == (B, A)
        assert indices.shape == (B, A)

        # Then sum over the agent dimension
        # [B, A] -> [B]
        traj_marginal_loss = torch.sum(traj_marginal_loss, dim=-1, keepdim=False)
        assert traj_marginal_loss.shape == (B,)

        # Then mean over the batch dimension
        # [B] -> 1
        traj_marginal_loss = torch.mean(traj_marginal_loss, dim=-1, keepdim=False)
        # assert traj_marginal_loss.shape == (1,)

        # score loss
        # [B, F, A, 1] -> [B, F, A, 1] along F axis
        score = torch.nn.functional.log_softmax(score, dim=1)
        _B, _F, _A, _ = score.shape
        # score: [B, F, A, 1] -> [B, F, A] -> [B, A, F] -> [B*A, F]
        score = score.squeeze(-1).transpose(-2, -1).contiguous().view(-1, _F)
        # gt_idx: [B, A] -> [B*A]
        gt_idx = indices.view(-1)
        assert score.shape == (_B * _A, _F)
        assert gt_idx.shape == (_B * _A,)
        # negtive likelihood loss: [B*A, F], [B*A] -> [1]
        score_marginal_loss = torch.nn.functional.nll_loss(score, gt_idx, reduction='mean')

        return traj_marginal_loss, score_marginal_loss

    def joint_loss(self, score, traj, gt_traj, gt_traj_padding_mask):
        """
        Args:
            score: [B, F, A, 1]
            traj [B, F, A, T, 2]
            gt_traj: [B, A, T, 2]
            gt_traj_padding_mask: [B, A, T]
        """
        # traj [B, F, A, T, 2], gt_traj: [B, A, T, 2] -> [B, F, A, T, 2] -> loss: [B, F, A, T, 2]
        loss = torch.nn.functional.smooth_l1_loss(traj,
                                                  gt_traj.unsqueeze(1).expand(*traj.shape),
                                                  beta=1.0,
                                                  reduction='none')

        # set the padding element in gt_traj to 0
        # mask: [B, A, T] 0/1 -> [B, A, T] True/False -> [B, 1, A, T] True/False -> [B, 1, A, T, 1] True/False
        # loss: [B, F, A, T, 2], [B, 1, A, T, 1] -> [B, F, A, T, 2]
        # loss = loss.masked_fill((gt_traj_padding_mask == 0).unsqueeze(1).unsqueeze(-1), 0.0)

        # mask: [B, A, T] -> [B, 1, A, T, 1]
        mask = gt_traj_padding_mask.unsqueeze(1).unsqueeze(-1)
        # loss: [B, F, A, T, 2], [B, 1, A, T, 1] -> [B, F, A, T, 2]
        loss = loss * mask

        # Now reduce across all timesteps and values to produce a tensor of shape [F, A]
        # [B, F, A, T, 2] -> [B, F, A]
        loss = torch.sum(loss, dim=(-2, -1), keepdim=False)

        # The joint loss, we sum over all agents to get a loss value per future.
        # [B, F, A] -> [B, F]
        traj_joint_loss = torch.sum(loss, dim=-1, keepdim=False)

        # Then only apply the loss to the best future prediction
        # [B, F] -> [B], [B]
        traj_joint_loss, indices = torch.min(traj_joint_loss, dim=-1, keepdim=False)

        # Then mean over the batch dimension
        # [B] -> 1
        traj_joint_loss = torch.mean(traj_joint_loss, dim=-1, keepdim=False)

        # score loss
        # [B, F, A, 1] -> [B, F]
        score = torch.sum(score, dim=(-2, -1), keepdim=False)
        # [B, F] -> [B, F] along F axis
        score = torch.nn.functional.log_softmax(score, dim=1)
        # [B]
        gt_idx = indices
        assert score.shape[0] == gt_idx.shape[0]
        # negtive likelihood loss: [B, F], [B] -> [1]
        score_joint_loss = torch.nn.functional.nll_loss(score, gt_idx, reduction='mean')

        return traj_joint_loss, score_joint_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['optim']['init_lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=self.config['optim']['step_size'],
                                              gamma=self.config['optim']['step_factor'])

        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_set = PickleDataset(Path(self.config['training']['sample_database_folder']))
        print(f"...............................Total Samples {len(train_set)} .......................................")
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
        # TODO(xlm) finish the following code
        '''
        self.eval()
        for _, batch in enumerate(self.test_dataloader()):
            vectors = [batch.actor_vector, batch.lane_center_vector]
            vector_masks = [batch.actor_vector_mask, batch.lane_center_vector_mask]
            cluster_masks = [batch.actor_cluster_mask, batch.lane_center_cluster_mask]
            target_info = [batch.candi_target_points]
            example_input = (vectors, vector_masks, cluster_masks, target_info)
            break

        kwargs = {}
        kwargs['example_outputs'] = self.infer_model(*example_input)
        kwargs['input_names'] = ['vectors', 'vector_masks', 'cluster_masks', 'target_info']
        kwargs['output_names'] = ['target_points', 'target_prob', 'traj']

        torch.onnx.export(self.infer_model, example_input, path, opset_version=11, **kwargs)
        onnx_model = onnx.load(path)
        simple_model, check = simplify(onnx_model)
        assert check, 'Simplifed ONNX model could not be validated'
        onnx.save(simple_model, f'{path}.simple')
        '''


if __name__ == '__main__':
    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print(f'{config_file} not exists!!')
    config_dir = os.path.dirname(config_file)

    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=config_dir)
    with open(config_file, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    torch.manual_seed(42)

    predictor = SceneTransformerPredictor(config)

    dataloader = predictor.train_dataloader()

    for batch in dataloader:
        predictor.test_step(batch, 0)
        break
