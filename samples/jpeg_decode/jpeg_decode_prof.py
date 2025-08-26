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
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

import numpy as np
import cv2
import argparse

import vaststreamx as vsx
import common.utils as utils
from common.media_profiler import MediaProfiler
from common.jdecoder import Jdecoder
import ast
from pathlib import Path
from easydict import EasyDict as edict


def argument_parser():
    parser = argparse.ArgumentParser("JPEG_DECODER_PROF")

    parser.add_argument(
        "-d",
        "--device_ids",
        default="[0]",
        help="device ids to run",
    )
    parser.add_argument(
        "--input_file",
        default="../data/images/plate_1920_1080.jpg",
        help="input file",
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

    input_file = ""
    if args.input_file != "":
        path = Path(args.input_file)
        if path.is_file():
            input_file = args.input_file
        else:
            print(f"{args.input_file} not exist")
            exit(-1)
    else:
        print("input_file is empty")
        exit(-1)

    decoders = []

    for i in range(instance):
        device_id = device_ids[i % len(device_ids)]
        decoder = Jdecoder(device_id, input_file)
        decoders.append(decoder)

    config = edict(
        {
            "instance": instance,
            "device_ids": device_ids,
            "iterations": iterations,
            "percentiles": percentiles,
        }
    )
    profiler = MediaProfiler(config, decoders)
    print(profiler.profiling())
