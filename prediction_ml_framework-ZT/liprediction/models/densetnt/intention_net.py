# Copyright (c) 2021 Li Auto Company. All rights reserved.
import os
import sys

import onnx
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from datasets.transform.dataset import PredDatasetWithCache
from datasets.transform.sample_transform.transform_upsample import (IntentionTransform8, IntentionTransform8Batch,
                                                                    IntentionTransform8RandomMask,
                                                                    PlotIntentionTransform8)
from module.context_encoder.context_encoder import ContextEncoder
from module.intention_decoder.intention_decoder import IntentionDecoder
from module.intention_sampler.intention_sampler import IntentionSampler
from module.postprocessor.postprocessor import Postprocessor
from module.preprocessor.preprocessor import Preprocessor
from module.trajectory_decoder.trajectory_decoder import TrajectoryDecoder
from onnxsim import simplify
from torch.utils.data import DataLoader
from yamlinclude import YamlIncludeConstructor


class IntentionNet8(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.preprocessor = Preprocessor.Create(config['preprocessor'])
        self.context_encoder = ContextEncoder.Create(config['context_encoder'])
        self.intention_decoder = IntentionDecoder.Create(config['intention_decoder'])
        self.traj_decoder = TrajectoryDecoder.Create(config['trajectory_decoder'])
        self.intention_sampler = IntentionSampler.Create(config['intention_sampler'])
        self.postprocessor = Postprocessor.Create(config['postprocessor'])

    def forward(self, vector, vector_mask, cluster_mask, agent_pose, teacher_intention):
        '''
            vector: [B, C, L, F]
            vector_mask: [B, C, L]
            cluster_mask: [B, C]
            agent_pose: [B, 4]
            teacher_intertion: [B, K, 2]  topK intention point
        '''
        # [B, C, L, F], [B, 4] -> [B, C, L, F]
        normlized_vector = self.preprocessor(vector, agent_pose)

        # [B, C, L, F], [B, C, L], [B, C] -> [B, C, F_c], cluster along L
        cluster_feature = self.context_encoder(normlized_vector, vector_mask, cluster_mask)

        # [B, C, F_c], [B, C] -> [B, 1, high_dim, high_dim], [B, 2, high_dim, high_dim]
        pix_score, pix_offset = self.intention_decoder(cluster_feature, cluster_mask)

        # [B, C, F_c], [B, C], [B, K, 2] -> [B, K, predict_traj_num, 2], topK traj
        traj = self.traj_decoder(cluster_feature, cluster_mask, teacher_intention)

        # [B, 1, high_dim, high_dim], [B, 2, high_dim, high_dim], [B, K, L, 2]
        return pix_score, pix_offset, traj


class IntentionNet8Infer(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, vector, vector_mask, cluster_mask, agent_pose):
        # normlize vector
        normlized_vector = self.model.preprocessor(vector, agent_pose)

        # context encoding
        cluster_feature = self.model.context_encoder(normlized_vector, vector_mask, cluster_mask)

        # intention decoding
        pix_score, pix_offset = self.model.intention_decoder(cluster_feature, cluster_mask)

        # intention sampling
        # [B, 2, candi_num], [B, candi_num]
        intention, score = self.model.intention_sampler(pix_score, pix_offset)

        # trajectory decoding
        # [B, candi_num, predict_traj_num, 2]
        pred_traj = self.model.traj_decoder(cluster_feature, cluster_mask, intention)

        # convert trajectory to global frame
        normlized_pred_traj = self.model.postprocessor(pred_traj, agent_pose)

        return normlized_pred_traj, score


class Intention8Predictor(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = IntentionNet8(config['model'])
        self.infer_model = IntentionNet8Infer(self.model)
        self.using_cache = False

    def forward(self, vector, vector_mask, cluster_mask, agent_pose, teacher_intention):
        return self.model(vector, vector_mask, cluster_mask, agent_pose, teacher_intention)

    def training_step(self, batch, batch_idx):
        # gt_pos: [B, L, 2] -> [B, 2] -> [B, 1, 2]
        # [B, 1, high_dim, high_dim], [B, 2, high_dim, high_dim], [B, K, L, 2]
        pix_score, pix_offset, traj = self(batch.vector, batch.vector_mask, batch.cluster_mask, batch.agent_pose,
                                           batch.gt_pos[:, -1].unsqueeze(1))

        # [B, 1, high_dim, high_dim] -> [1]
        intention_loss = self.score_loss(pix_score, batch)
        # [B, 2, high_dim, high_dim] -> [1]
        offset_loss = self.offset_loss(pix_offset, batch)
        # traj: [B, 1, L, 2] -> [B, L, 2]
        # mse_loss: [B, L, 2] [B, L, 2] -> [1]
        traj_loss = F.mse_loss(traj.squeeze(1), batch.gt_pos)

        loss = self.config['loss']['intention_loss_wgt'] * intention_loss + \
            self.config['loss']['offset_loss_wgt'] * offset_loss + \
            self.config['loss']['traj_loss_wgt'] * traj_loss

        self.log('train_intention_loss', intention_loss)
        self.log('train_offset_loss', offset_loss)
        self.log('train_traj_loss', traj_loss)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        pix_score, pix_offset, traj = self(batch.vector, batch.vector_mask, batch.cluster_mask, batch.agent_pose,
                                           batch.gt_pos[:, -1].unsqueeze(1))

        intention_loss = self.score_loss(pix_score, batch)
        offset_loss = self.offset_loss(pix_offset, batch)
        traj_loss = F.mse_loss(traj.squeeze(1), batch.gt_pos)

        loss = self.config['loss']['intention_loss_wgt'] * intention_loss + \
            self.config['loss']['offset_loss_wgt'] * offset_loss + \
            self.config['loss']['traj_loss_wgt'] * traj_loss

        self.log('val_intention_loss', intention_loss)
        self.log('val_offset_loss', offset_loss)
        self.log('val_traj_loss', traj_loss)
        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        normlized_vector = self.model.preprocessor(batch.vector, batch.agent_pose)

        cluster_feature = self.model.context_encoder(normlized_vector, batch.vector_mask, batch.cluster_mask)

        pix_score, pix_offset = self.model.intention_decoder(cluster_feature, batch.cluster_mask)

        intention, score = self.model.intention_sampler(pix_score, pix_offset)

        pred_traj = self.model.traj_decoder(cluster_feature, batch.cluster_mask, intention)

        B, D = pix_score.shape[:2]
        pix_score = F.softmax(pix_score.view(B, -1), dim=-1).view(B, D, D)
        data_list = batch.to_data_list()
        for i, data in enumerate(data_list):
            data.pred_heatmap = pix_score[i]
            data.pred_traj = pred_traj[i]
            plt = PlotIntentionTransform8(data.to('cpu'), show=False)
            plt.savefig(f'{self.config["test"]["plot_out_folder"]}/{data.raw_sample_name}.svg')

    def test_epoch_end(self, res_list):
        pass

    def train_dataloader(self):
        train_set = PredDatasetWithCache(self.config['training']['sample_database_folder'],
                                         self.config['training']['sample_list_file'],
                                         transform=IntentionTransform8(self.config['transform']),
                                         cache_transform=IntentionTransform8RandomMask(self.config['transform']),
                                         cache_root=self.config['training']['cache_database_folder'],
                                         cache_worker_num=self.config['training']['cache_worker_num'],
                                         compress_lvl=self.config['training']['cache_compress_lvl'],
                                         using_cache=self.using_cache)

        train_loader = DataLoader(train_set,
                                  batch_size=self.config['training']['batch_size'],
                                  shuffle=True,
                                  collate_fn=IntentionTransform8Batch.from_data_list,
                                  num_workers=self.config['training']['loader_worker_num'],
                                  drop_last=True,
                                  pin_memory=False)

        return train_loader

    def val_dataloader(self):
        val_set = PredDatasetWithCache(self.config['validation']['sample_database_folder'],
                                       self.config['validation']['sample_list_file'],
                                       transform=IntentionTransform8(self.config['transform']),
                                       cache_root=self.config['validation']['cache_database_folder'],
                                       cache_worker_num=self.config['validation']['cache_worker_num'],
                                       compress_lvl=self.config['validation']['cache_compress_lvl'],
                                       using_cache=self.using_cache)

        val_loader = DataLoader(val_set,
                                batch_size=self.config['validation']['batch_size'],
                                shuffle=False,
                                collate_fn=IntentionTransform8Batch.from_data_list,
                                num_workers=self.config['validation']['loader_worker_num'],
                                drop_last=True,
                                pin_memory=False)

        return val_loader

    def test_dataloader(self):
        test_set = PredDatasetWithCache(self.config['test']['sample_database_folder'],
                                        self.config['test']['sample_list_file'],
                                        transform=IntentionTransform8(self.config['transform']),
                                        cache_root=self.config['test']['cache_database_folder'],
                                        cache_worker_num=self.config['test']['cache_worker_num'],
                                        compress_lvl=self.config['test']['cache_compress_lvl'],
                                        using_cache=self.using_cache)

        test_loader = DataLoader(test_set,
                                 batch_size=self.config['test']['batch_size'],
                                 shuffle=False,
                                 collate_fn=IntentionTransform8Batch.from_data_list,
                                 num_workers=self.config['test']['loader_worker_num'],
                                 drop_last=False,
                                 pin_memory=False)

        return test_loader

    #  def on_after_backward(self):
    #      if self.trainer.global_step < 10:  # don't make the tf file huge
    #          for k, v in self.named_parameters():
    #              self.logger.experiment.add_histogram(
    #                  tag=k, values=v.grad, global_step=self.trainer.global_step
    #              )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['optim']['init_lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=self.config['optim']['step_size'],
                                              gamma=self.config['optim']['step_factor'])

        return [optimizer], [scheduler]

    def score_loss(self, pix_score_raw, batch):
        '''
            pix_score_raw: [B, 1, high_dim, high_dim]
        '''
        B, dev = pix_score_raw.shape[0], pix_score_raw.device
        # [B, 1, high_dim, high_dim] -> [B, high_dim*high_dim]
        pix_score = F.softmax(pix_score_raw.view(B, -1), dim=-1)
        # [B, 1, high_dim, high_dim] -> [B, high_dim*high_dim]
        pix_log_score = F.log_softmax(pix_score_raw.view(B, -1), dim=-1)
        index = (torch.arange(B, device=dev), batch.gt_pix)
        # Focal loss: -(1-p_t)^r * log(p_t)
        # [B, high_dim*high_dim] -> [B] -> [1]
        pix_loss = torch.mean(-torch.pow(1 - pix_score[index], self.config['loss']['alpha']) * pix_log_score[index])
        # [1]
        return pix_loss

    def offset_loss(self, pix_offset, batch):
        '''
            pix_offset: [B, 2, high_dim, high_dim]
        '''
        B, dev = pix_offset.shape[0], pix_offset.device
        # [B, 2, high_dim*high_dim]
        pix_offset = pix_offset.view(B, 2, -1)
        index = (torch.arange(B, device=dev), batch.gt_pix)
        # [B, 2] vs [B, 2] -> [1]
        offset_loss = F.mse_loss(pix_offset[index[0], :, index[1]], batch.gt_pix_offset)
        # [1]
        return offset_loss

    def fde(self, pred_target, gt_target):
        '''
        pred_target in shape [B, K, 2]
        gt_target in shape [B,2]
        '''

        dist = torch.linalg.norm(pred_target - gt_target.unsqueeze(1), dim=-1)
        min_fde, min_idx = torch.min(dist, dim=-1)

        return min_fde, min_idx

    def cache_dataset(self, split):
        if split == 'train':
            print('Caching training set')
            dataloader = self.train_dataloader()
        elif split == 'val':
            print('Caching validation set')
            dataloader = self.val_dataloader()
        elif split == 'test':
            print('Caching test set')
            dataloader = self.test_dataloader()
        else:
            raise Exception('Not implementation')

        dataloader.dataset.cache()

    def export_jit(self, device='cuda'):
        self.eval()

        dl = self.test_dataloader()

        for i, batch in enumerate(dl):
            batch = batch.to(device)
            example_input = (batch.vector, batch.vector_mask, batch.cluster_mask, batch.agent_pose,
                             batch.gt_pos[:, -1].unsqueeze(1))
            break

        model = torch.jit.trace(self.model.to(device), example_input)
        model.to(device)

        return model

    def export_onnx(self, path):
        self.eval()
        dl = self.test_dataloader()

        for i, batch in enumerate(dl):
            example_input = (batch.vector, batch.vector_mask, batch.cluster_mask, batch.agent_pose)
            break

        kwargs = {}
        kwargs['example_outputs'] = self.infer_model(*example_input)
        kwargs['input_names'] = ['vector', 'vector_mask', 'cluster_mask', 'agent_pose']
        kwargs['output_names'] = ['traj', 'score']

        torch.onnx.export(self.infer_model, example_input, path, opset_version=11, **kwargs)
        onnx_model = onnx.load(path)
        simple_model, check = simplify(onnx_model)
        assert check, 'Simplifed ONNX model could not be validated'
        onnx.save(simple_model, f'{path}.simple')


if __name__ == '__main__':
    config_file = sys.argv[1]
    if not os.path.exists(config_file):
        print(f'{config_file} not exists!!')
    config_dir = os.path.dirname(config_file)

    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=config_dir)
    with open(config_file, 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    torch.manual_seed(1988)

    predictor = Intention8Predictor(config)

    dataloader = predictor.val_dataloader()

    for batch in dataloader:
        pix_score, pix_offset, traj = predictor.model(batch.vector, batch.vector_mask, batch.cluster_mask,
                                                      batch.agent_pose, batch.gt_pos[:, -1].unsqueeze(1))
        print(pix_score.shape)
        print(pix_offset.shape)
        print(traj.shape)
        break
