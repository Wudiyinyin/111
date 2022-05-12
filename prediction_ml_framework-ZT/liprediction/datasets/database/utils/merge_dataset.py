# Copyright (c) 2021 Li Auto Company. All rights reserved.
import argparse
import os
import shutil

from tqdm import tqdm

from ..sample_database import SampleDatabase, SampleDataset

parser = argparse.ArgumentParser(description='Meger two dataset.')

parser.add_argument('-i0',
                    '--input0',
                    type=str,
                    nargs='+',
                    default='',
                    required=True,
                    help='First input dataset, in format: /path/to/database /path/to/list.txt')

parser.add_argument('-i1',
                    '--input1',
                    type=str,
                    nargs='+',
                    default='',
                    required=True,
                    help='Second input dataset, in format /path/to/database /path/to/list.txt')

parser.add_argument('-o',
                    '--output',
                    type=str,
                    nargs='+',
                    default='',
                    help='Specify the output database, if not specified, use input0')

if __name__ == '__main__':
    args = parser.parse_args()
    input0_arg_list = [None, None]
    for i, arg in enumerate(args.input0):
        input0_arg_list[i] = arg

    input1_arg_list = [None, None]
    for i, arg in enumerate(args.input1):
        input1_arg_list[i] = arg

    output_arg_list = [None, None]
    for i, arg in enumerate(args.output):
        output_arg_list[i] = arg

    input0_as_output = False
    if not output_arg_list[0]:
        input0_as_output = True

    dataset_list = []

    # open dataset0
    if not input0_as_output:
        dataset_list.append(SampleDataset(input0_arg_list[0], 'lmdb', input0_arg_list[1]))

    # open dataset1
    dataset_list.append(SampleDataset(input1_arg_list[0], 'lmdb', input1_arg_list[1]))

    # open output database
    if input0_as_output:
        output_database_folder = input0_arg_list[0]
        output_list_path = input0_arg_list[1]  # this can be None
    else:
        shutil.rmtree(output_arg_list[0], ignore_errors=True)
        os.makedirs(output_arg_list[0])
        output_database_folder = output_arg_list[0]
        output_list_path = output_arg_list[1]  # this can be None

    output_database = SampleDatabase.create(output_database_folder, 'lmdb', True)
    sample_name_set = set()
    if output_list_path is not None and os.path.exists(output_list_path):
        with open(output_list_path, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                sample_name_set.add(line.strip())

    # do merge work
    for did, dataset in enumerate(dataset_list):
        print(f'Processing dataset {did}\n')
        for i in tqdm(range(len(dataset))):
            sample_name, sample_data = dataset.get_raw_sample(i)
            output_database.put_raw(sample_name.encode('utf-8'), sample_data)
            sample_name_set.add(sample_name)

    # write back list file
    if output_list_path is not None:
        with open(output_list_path, 'w') as fout:
            for sample_name in sample_name_set:
                fout.write(f'{sample_name}\n')

    print(f'Merge done! output to \'{output_database_folder}\' and \'{output_list_path}\'')
