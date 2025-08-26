#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import vaststreamx as vsx

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

from pathlib import Path

from common.jencoder import Jencoder
from common.media_profiler import MediaProfiler
from easydict import EasyDict as edict
import argparse
import ast


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--device_ids",
        default="[0]",
        help="device ids to run",
    )
    parser.add_argument(
        "--width",
        default=0,
        type=int,
        help="frame width",
    )
    parser.add_argument(
        "--height",
        default=0,
        type=int,
        help="frame height",
    )
    parser.add_argument(
        "--input_file",
        default="../data/images/plate_1920_1080.yuv",
        help="input file path",
    )
    parser.add_argument(
        "-i",
        "--instance",
        default=1,
        type=int,
        help="instance number for each device",
    )
    parser.add_argument(
        "--iterations",
        default=10240,
        type=int,
        help="iterations count for one profiling",
    )
    parser.add_argument(
        "--percentiles",
        default="[50, 90, 95, 99]",
        help="percentiles of latency",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    device_ids = ast.literal_eval(args.device_ids)
    instance = args.instance
    iterations = args.iterations
    percentiles = ast.literal_eval(args.percentiles)

    input_file = args.input_file

    encoders = []
    for i in range(instance):
        device_id = device_ids[i % len(device_ids)]
        encoder = Jencoder(
            device_id,
            input_file,
            args.width,
            args.height,
            vsx.ImageFormat.YUV_NV12,
        )
        encoders.append(encoder)

    # print(f"len encoder: {len(encoders)}")
    config = edict(
        {
            "instance": instance,
            "iterations": iterations,
            "device_ids": device_ids,
            "percentiles": percentiles,
        }
    )
    profiler = MediaProfiler(config, encoders)
    print(profiler.profiling())
