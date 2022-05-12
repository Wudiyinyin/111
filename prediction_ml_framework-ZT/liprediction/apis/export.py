# Copyright (c) 2021 Li Auto Company. All rights reserved.

import argparse
import os

import yaml
from models.densetnt.dense_tnt import DenseTNTPredictor


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='eval intention network.')

    parser.add_argument('-cfg', '--config', type=str, default='', required=True, help='config file')
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None, required=True, help='checkpoint file')
    parser.add_argument('-t', '--type', type=str, default='onnx', help='only support onnx for now')
    parser.add_argument('-p', '--path', type=str, default='./output', help='dst folder')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args: argparse.ArgumentParser = parse_args()

    config_file = args.config
    if config_file is None:
        raise ValueError('config file is required')
    print(f'Using config: {config_file}')
    with open(config_file, 'rb') as f:
        config = yaml.safe_load(f)

    checkpoint_file = args.checkpoint
    if checkpoint_file is not None:
        print(f'Using checkpoint: {checkpoint_file}')

    output_path = args.path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    predictor_name = f"{config['model_name']}"
    predictor = globals()[predictor_name].load_from_checkpoint(checkpoint_path=checkpoint_file, config=config)

    if args.type == 'onnx':
        onnx_model_file_path = f'{output_path}/{predictor_name}_model.onnx'
        predictor.export_onnx(onnx_model_file_path)
        print(f'save onnx model to {onnx_model_file_path}')
