
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

from common.super_resolution import SuperResolution
from common.model_profiler import ModelProfiler
from easydict import EasyDict as edict
import argparse
import ast


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        help="hw-config file of the model suite",
        default="",
    )
    parser.add_argument(
        "--vdsp_params",
        default="./data/configs/rcan_bgr888.json",
        help="vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--postproc_elf", default="/opt/vastai/vaststreamx/data/elf/postprocessimage"
    )
    parser.add_argument(
        "--denorm",
        default="[0,1,1]",
        help="denormalization params [mean, std, scale]",
    )
    parser.add_argument("-d", "--device_ids", default="[0]", help="device ids to run")
    parser.add_argument(
        "-b",
        "--batch_size",
        default=1,
        type=int,
        help="profiling batch size of the model",
    )
    parser.add_argument(
        "-i",
        "--instance",
        default=1,
        type=int,
        help="instance number for each device",
    )
    parser.add_argument(
        "-s",
        "--shape",
        help="model input shape",
    )
    parser.add_argument(
        "--iterations",
        default=20,
        type=int,
        help="iterations count for one profiling",
    )
    parser.add_argument(
        "--queue_size",
        default=1,
        type=int,
        help="aync wait queue size",
    )
    parser.add_argument(
        "--percentiles",
        default="[50, 90, 95, 99]",
        help="percentiles of latency",
    )
    parser.add_argument(
        "--input_host",
        default=0,
        type=int,
        help="cache input data into host memory",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    model_prefix = args.model_prefix
    vdsp_params = args.vdsp_params
    hw_config = args.hw_config
    device_ids = ast.literal_eval(args.device_ids)
    batch_size = args.batch_size
    instance = args.instance
    iterations = args.iterations
    queue_size = args.queue_size
    input_host = args.input_host
    percentiles = ast.literal_eval(args.percentiles)
    postproc_elf = args.postproc_elf
    mean, std, scale = ast.literal_eval(args.denorm)

    models = []
    contexts = []
    for i in range(instance):
        device_id = device_ids[i % len(device_ids)]
        model = SuperResolution(
            model_prefix,
            vdsp_params,
            postproc_elf,
            device_id,
            std,
            scale,
            mean,
            batch_size,
            hw_config,
        )
        models.append(model)
        if input_host:
            contexts.append("CPU")
        else:
            contexts.append("VACC")

    if args.shape:
        shape = ast.literal_eval(args.shape)
    else:
        shape = models[0].input_shape[0]

    config = edict(
        {
            "instance": instance,
            "iterations": iterations,
            "batch_size": batch_size,
            "data_type": "uint8",
            "device_ids": device_ids,
            "contexts": contexts,
            "input_shape": shape,
            "percentiles": percentiles,
            "queue_size": queue_size,
        }
    )
    profiler = ModelProfiler(config, models)
    print(profiler.profiling())
