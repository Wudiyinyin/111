# Copyright (c) 2021 Li Auto Company. All rights reserved.

import argparse
import os
import time
import warnings

import pytorch_lightning as pl
import yaml
from models.densetnt.dense_tnt import DenseTNTPredictor
from models.multi_path_decoder.multi_path_decoder import MultiPathDecoderPredictor
from models.multi_path_pro.multi_path_pro import MultiPathProPredictor
from models.scene_transformer.scene_transformer import SceneTransformerPredictor
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

warnings.filterwarnings("ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*")


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='train intention network.')

    parser.add_argument('-cfg', '--config', type=str, default='', required=True, help='config file')
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None, help='checkpoint file')
    parser.add_argument('-save', '--saved_model', type=str, default='./result', help='path to save model')
    parser.add_argument('-log', '--log_dir', type=str, default='./log', help='log directory')
    parser.add_argument('-g', '--gpu', type=int, default=1, help='0: cpu, n: n num of gpus, -1: all gpus')
    parser.add_argument('-n', '--node', type=int, default=1, help='num of nodes across multi machines')
    parser.add_argument('--resume_weight_only',
                        dest='resume_weight_only',
                        action='store_true',
                        default=False,
                        help='resume only weights from chekpoint file')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Must set seed everything in multi node multi gpus training
    seed: int = 42
    seed_everything(seed)

    args: argparse.ArgumentParser = parse_args()
    print("\nargs:", args)

    if os.environ.get('MASTER_ADDR') is not None:
        # os.environ['MASTER_ADDR'] = os.environ['MASTER_ADDR']
        print(f"environment variable:  MASTER_ADDR={os.environ['MASTER_ADDR']} ")
    if os.environ.get('MASTER_PORT') is not None:
        # os.environ['MASTER_PORT'] = os.environ['MASTER_PORT']
        print(f"environment variable:  MASTER_PORT={os.environ['MASTER_PORT']} ")
    if os.environ.get('WORLD_SIZE') is not None:
        # os.environ['WORLD_SIZE'] = os.environ['WORLD_SIZE']
        print(f"environment variable:  WORLD_SIZE={os.environ['WORLD_SIZE']} ")
    if os.environ.get('RANK') is not None:
        os.environ['NODE_RANK'] = os.environ['RANK']
        print(f"environment variable:  RANK={os.environ['RANK']} ")
    if os.environ.get('PYTHONUNBUFFERED') is not None:
        print(f"environment variable:  PYTHONUNBUFFERED={os.environ['PYTHONUNBUFFERED']} ")

    # load config
    config_file = args.config
    print(f'\nUsing config: {config_file}')
    with open(config_file, 'rb') as f:
        config = yaml.safe_load(f)
    print("\nconfig:", config)

    # set GPU device
    gpu_num = args.gpu
    if gpu_num == 0:
        print('Gpu not specified, exit normally')
        exit(0)

    node_num = args.node
    if node_num < 1:
        print('Node num must be greater than 0')
        exit(0)

    # create logger
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb_logger = TensorBoardLogger(log_dir, name=config['model_name'])

    predictor_name = f"{config['model_name']}"

    # set checkpoint callback to save best val_error_rate and last epoch
    saved_path = args.saved_model
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=saved_path,
                                          filename="checkpoint-{epoch:04d}-{val_loss:.5f}",
                                          save_weights_only=False,
                                          mode='min',
                                          save_top_k=10)
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)

    # set checkpoint path
    checkpoint_file = args.checkpoint
    if checkpoint_file is not None:
        print(f'Using checkpoint: {checkpoint_file}')

    resume_weight_only = args.resume_weight_only
    if resume_weight_only:
        predictor = globals()[predictor_name].load_from_checkpoint(config=config,
                                                                   checkpoint_path=checkpoint_file,
                                                                   strict=True)
    else:
        predictor = globals()[predictor_name](config=config)

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_monitor],
        gpus=gpu_num,
        auto_select_gpus=True,
        num_nodes=node_num,
        max_epochs=config['training']['epoch'],
        val_check_interval=config['validation']['check_interval'],
        limit_val_batches=config['validation']['limit_batches'],
        logger=tb_logger,
        log_every_n_steps=config['log_every_n_steps'],
        gradient_clip_val=config['optim']['gradient_clip_val'],
        gradient_clip_algorithm=config['optim']['gradient_clip_algorithm'],
        sync_batchnorm=True,
        accelerator="gpu",
        strategy=DDPPlugin(find_unused_parameters=False),
        replace_sampler_ddp=True,
        deterministic=True,
    )

    if resume_weight_only:
        trainer.fit(predictor)
    else:
        trainer.fit(predictor, ckpt_path=checkpoint_file)
