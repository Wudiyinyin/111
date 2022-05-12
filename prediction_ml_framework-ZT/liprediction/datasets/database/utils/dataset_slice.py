# Copyright (c) 2021 Li Auto Company. All rights reserved.
import argparse
import random


def create_file(file_path, msg):
    f = open(file_path, "a")
    for i in range(len(msg)):
        s = str(msg[i])
        f.write(s)
    f.close


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("attention_labeled_file", type=str)
    parser.add_argument("slice_config", type=int, nargs="*")
    args = parser.parse_args()
    attention_labeled_list = []
    with open(args.attention_labeled_file, "r") as f:
        for line in f.readlines():
            attention_labeled_list.append(line)

    random.shuffle(attention_labeled_list)
    used_num = 0
    total_num = sum(args.slice_config)
    attention_dataset_path = args.attention_labeled_file[:args.attention_labeled_file.rindex("/")]
    for i in range(len(args.slice_config)):
        split_num = int(args.slice_config[i] / total_num * len(attention_labeled_list))
        split_list = attention_labeled_list[used_num:used_num + split_num]
        used_num += split_num
        create_file(attention_dataset_path + "/slice_" + str(i) + ".txt", split_list)
