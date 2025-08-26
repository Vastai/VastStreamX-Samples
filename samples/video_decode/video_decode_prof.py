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

from common.vdecoder import Vdecoder
from common.media_profiler import MediaProfiler
from easydict import EasyDict as edict
import argparse
import ast


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--codec_type",
        default="H264",
        help="codec type eg. H264/H265",
    )
    parser.add_argument(
        "-d",
        "--device_ids",
        default="[0]",
        help="device ids to run",
    )
    parser.add_argument(
        "--input_file",
        default="",
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
    # codec_type = args.codec_type
    device_ids = ast.literal_eval(args.device_ids)
    # input_file = args.input_file
    instance = args.instance
    iterations = args.iterations
    percentiles = ast.literal_eval(args.percentiles)

    codec_type = vsx.CodecType.CODEC_TYPE_H264
    if args.codec_type == "H264" or args.codec_type == "h264":
        codec_type = vsx.CodecType.CODEC_TYPE_H264
    elif args.codec_type == "H265" or args.codec_type == "h265":
        codec_type = vsx.CodecType.CODEC_TYPE_H265
    elif args.codec_type == "AV1" or args.codec_type == "av1":
        codec_type = vsx.CodecType.CODEC_TYPE_AV1
    else:
        print(f"undefined codec_type:{args.codec_type}")
        exit(-1)

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

    # models = []
    decoders = []

    for i in range(instance):
        device_id = device_ids[i % len(device_ids)]
        decoder = Vdecoder(codec_type, device_id, input_file)
        decoders.append(decoder)

    config = edict(
        {
            "instance": instance,
            "iterations": iterations,
            "device_ids": device_ids,
            "percentiles": percentiles,
        }
    )
    profiler = MediaProfiler(config, decoders)
    print(profiler.profiling())
