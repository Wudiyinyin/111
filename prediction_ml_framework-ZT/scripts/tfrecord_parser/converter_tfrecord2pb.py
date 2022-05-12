# Copyright (c) 2021 Li Auto Company. All rights reserved.

import argparse
import os
from pathlib import Path

import tensorflow as tf
from joblib import Parallel, delayed
from waymo_open_dataset.protos import scenario_pb2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--tfrecord_data_dir", type=str, required=True, help="TFRecord data dir for converter")
    parser.add_argument("--pb_data_dir", type=str, required=True, help="Dump the muliple streaming pb files")

    args = parser.parse_args()
    return args


def converter(file_path: Path, target_path: Path) -> None:
    target_file = file_path.stem + '-' + file_path.suffix[1:] + '.bin'
    dump_file_path = target_path / target_file
    print(f"Convert file path : {file_path}")

    raw_dataset = tf.data.TFRecordDataset([file_path])

    pb_frames = bytearray()
    for raw_record in raw_dataset:
        proto_string = raw_record.numpy()
        proto = scenario_pb2.Scenario()
        proto.ParseFromString(proto_string)
        pb_bin = proto.SerializeToString()

        pb_len = len(pb_bin)
        byte_len = pb_len.to_bytes(4, byteorder='big')

        pb_frames.extend(byte_len)
        pb_frames.extend(pb_bin)

    with open(dump_file_path, 'wb') as f:
        f.write(pb_frames)

    print(f"Dump pb file : {dump_file_path}")


if __name__ == "__main__":
    args: argparse.ArgumentParser = parse_args()

    tfrecord_data_path: Path = Path(args.tfrecord_data_dir)
    target_path: Path = Path(args.pb_data_dir)
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    Parallel(n_jobs=-1)(delayed(converter)(file_path, target_path) for file_path in tfrecord_data_path.iterdir())
