#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)
from collections import OrderedDict

from common.detection3d import Detection3D
from easydict import EasyDict as edict

import numpy as np
import argparse
import ast


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefixs",
        default="[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_configs",
        default="[]",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--max_voxel_num",
        default="[16000]",
        help="model max voxel number",
    )
    parser.add_argument(
        "--voxel_size",
        default="[0.16, 0.16, 4]",
        help="voxel size",
    )
    parser.add_argument(
        "--coors_range",
        default="[0, -39.68, -3, 69.12, 39.68, 1]",
        help="coors range",
    )
    parser.add_argument(
        "--elf_file",
        default="/opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op",
        help="elf file",
    )
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--max_points_num",
        default=120000,
        type=int,
        help="max points number per input",
    )
    parser.add_argument(
        "--shuffle_enabled",
        default=0,
        type=int,
        help="shuffle enabled",
    )
    parser.add_argument(
        "--normalize_enabled",
        default=0,
        type=int,
        help="normalize enabled",
    )
    parser.add_argument(
        "--feat_size",
        default="[864,496,480,480]",
        help="set model feature sizes,[max_feature_width,max_feature_height,actual_feature_width,actual_feature_height]",
    )
    parser.add_argument(
        "--input_file",
        default="/opt/vastai/vaststreamx/data/datasets/fov_pointcloud_float16/000001.bin",
        help="input file",
    )
    parser.add_argument(
        "--dataset_filelist",
        default="",
        help="dataset filename list",
    )
    parser.add_argument(
        "--dataset_root",
        default="",
        help="dataset root",
    )
    parser.add_argument(
        "--dataset_output_folder",
        default="",
        help="dataset output folder path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1
    model_prefixs = args.model_prefixs.strip("[").strip("]").split(",")
    hw_configs = args.hw_configs.strip("[").strip("]").split(",")
    max_voxel_nums = ast.literal_eval(args.max_voxel_num)
    voxel_size = ast.literal_eval(args.voxel_size)
    coors_range = ast.literal_eval(args.coors_range)
    max_points_num = args.max_points_num
    shuffle_enabled = args.shuffle_enabled
    normalize_enabled = args.normalize_enabled
    feat_size = ast.literal_eval(args.feat_size)

    assert len(model_prefixs) == len(max_voxel_nums)
    assert len(feat_size) == 4

    modle_configs = [
        edict(
            {
                "max_voxel_num": max_voxel_nums[i],
                "model_prefix": model_prefixs[i],
                "hw_config": hw_configs[i] if len(hw_configs) > i else "",
            }
        )
        for i in range(len(model_prefixs))
    ]

    model = Detection3D(
        model_configs=modle_configs,
        elf_file=args.elf_file,
        voxel_sizes=voxel_size,
        coors_range=coors_range,
        shuffle_enabled=shuffle_enabled,
        normalize_enabled=normalize_enabled,
        device_id=args.device_id,
        max_points_num=max_points_num,
        max_feature_width=feat_size[0],
        max_feature_height=feat_size[1],
        actual_feature_width=feat_size[2],
        actual_feature_height=feat_size[3],
    )

    if args.dataset_filelist == "":
        input = None
        input = np.fromfile(args.input_file, dtype=np.float16)
        outputs = model.process(input)
        outputs = [out.astype(np.float32) for out in outputs]
        scores, labels, boxes = outputs
        np.set_printoptions(precision=4)
        for i in range(500):
            if scores[i] < 0:
                break
            print(f"label: {int(labels[i])}, score: {scores[i]:.6f}, box:{boxes[i]}")
    else:
        filelist = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = [line.strip() for line in f.readlines()]
        for file in filelist:
            fullname = os.path.join(args.dataset_root, file)
            print(fullname)
            input = None
            input = np.fromfile(fullname, dtype=np.float16)
            scores, labels, boxes = model.process(input)
            basename = os.path.basename(fullname)
            prefix = os.path.join(args.dataset_output_folder, basename)
            print(f"prefix:{prefix}")
            scores.tofile(prefix + ".score")
            labels.tofile(prefix + ".label")
            boxes.tofile(prefix + ".box")
