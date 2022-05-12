# Copyright (c) 2021 Li Auto Company. All rights reserved.

import argparse
import os
from pathlib import Path

from metrics import waymo_metrics


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='dump prediction metrics results.')

    parser.add_argument('-i', '--input', type=str, required=True, help='input prediction result pb')
    parser.add_argument('-o', '--output', type=str, default='./benchmark_result', help='output folder')
    parser.add_argument('-g', '--gt_file', type=str, required=True, help='ground truth file ')
    parser.add_argument('-t', '--type', type=str, default='waymo', help='only support waymo for now')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args: argparse.ArgumentParser = parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print('input file not exists')
        raise FileNotFoundError

    gt_file = Path(args.gt_file)
    if not gt_file.exists():
        print('ground truth file not exists')
        raise FileNotFoundError

    output_path = Path(args.output)
    if not output_path.exists():
        os.makedirs(output_path)
    output_file: Path = output_path / (input_file.stem + '.csv')

    if args.type not in ['waymo']:
        print(f'Dont support this type : {args.type}')
        raise ValueError

    metrics_name = f"{args.type}_metrics"
    metrics = globals()[metrics_name]

    metrics.dump_prediction_results(input_file, output_file, gt_file)
    print(f'Save prediction result to {str(output_file)}')
