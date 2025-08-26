#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
from tqdm import tqdm
import glob
from collections import OrderedDict


current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../../")
sys.path.append(common_path)

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
        default="[32000]",
        help="model max voxel number",
    )
    parser.add_argument(
        "--voxel_size",
        default="[0.32,0.32,4.2]",
        help="voxel size",
    )
    parser.add_argument(
        "--coors_range",
        default="[-50,-103.6,-0.1,103.6,50,4.1]",
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
        default=2000000,
        type=int,
        help="max points number per input",
    )
    parser.add_argument(
        "--shuffle_enabled",
        default=1,
        type=int,
        help="shuffle enabled",
    )
    parser.add_argument(
        "--normalize_enabled",
        default=1,
        type=int,
        help="normalize enabled",
    )
    parser.add_argument(
        "--input_file",
        default="",
        help="input file",
    )
    parser.add_argument(
        "--save_npz",
        default=0,
        type=int,
        help="save npz file",
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
    if args.dataset_root != "" and os.path.exists(args.dataset_output_folder) is False:
        os.makedirs(args.dataset_output_folder)
    batch_size = 1
    model_prefixs = args.model_prefixs.strip("[").strip("]").split(",")
    hw_configs = args.hw_configs.strip("[").strip("]").split(",")
    max_voxel_nums = ast.literal_eval(args.max_voxel_num)
    voxel_size = ast.literal_eval(args.voxel_size)
    coors_range = ast.literal_eval(args.coors_range)
    max_points_num = args.max_points_num
    shuffle_enabled = args.shuffle_enabled
    normalize_enabled = args.normalize_enabled
    assert len(model_prefixs) == len(max_voxel_nums)

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
        max_feature_width=864,
        max_feature_height=496,
        actual_feature_width=432,
        actual_feature_height=496,
    )

    if args.input_file != "":
        input = None
        input_fp32 = np.fromfile(args.input_file, dtype=np.float32)
        input = input_fp32.astype(np.float16)
        outputs = model.process(input)
        outputs = [out.astype(np.float32) for out in outputs]
        scores, labels, boxes = outputs
        np.set_printoptions(precision=4)
        for i in range(500):
            if scores[i] < 0:
                break
            print(f"label: {int(labels[i])}, score: {scores[i]:.6f}, box:{boxes[i]}")
    else:
        filelist = [
            f for f in sorted(glob.glob(args.dataset_root + "/*.bin", recursive=True))
        ]
        for file in tqdm(filelist):
            print(file)
            input_fp32 = np.fromfile(file, dtype=np.float32)
            input = input_fp32.astype(np.float16)
            scores, labels, boxes = model.process(input)
            np.savez(
                os.path.join(
                    args.dataset_output_folder,
                    file.split("/")[-1].replace("bin", "npz"),
                ),
                **{
                    "score": np.array(scores),
                    "label": np.array(labels),
                    "boxes": np.array(boxes),
                },
            )
