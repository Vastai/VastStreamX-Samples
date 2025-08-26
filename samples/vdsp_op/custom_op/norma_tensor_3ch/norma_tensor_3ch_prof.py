import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../../..")
sys.path.append(common_path)

from norma_tensor_3ch import NormaTensor3ChOp
from common.model_profiler import ModelProfiler
from easydict import EasyDict as edict
import argparse
import ast


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elf_file",
        default="/opt/vastai/vaststreamx/data/elf/norma_tensor_3ch",
        help="elf file",
    )
    parser.add_argument(
        "-d",
        "--device_ids",
        default="[0]",
        help="device ids to run",
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
        default=10240,
        type=int,
        help="iterations count for one profiling",
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
    elf_file = args.elf_file
    device_ids = ast.literal_eval(args.device_ids)
    batch_size = 1
    instance = args.instance
    iterations = args.iterations
    queue_size = 0
    input_host = args.input_host
    percentiles = ast.literal_eval(args.percentiles)
    shape = ast.literal_eval(args.shape)

    ops = []
    contexts = []
    for i in range(instance):
        device_id = device_ids[i % len(device_ids)]
        op = NormaTensor3ChOp("norma_tensor_3ch_op", elf_file, device_id)
        ops.append(op)
        if input_host:
            contexts.append("CPU")
        else:
            contexts.append("VACC")

    config = edict(
        {
            "instance": instance,
            "iterations": iterations,
            "batch_size": batch_size,
            "data_type": "float16",
            "device_ids": device_ids,
            "contexts": contexts,
            "input_shape": shape,
            "percentiles": percentiles,
            "queue_size": queue_size,
        }
    )

    profiler = ModelProfiler(config, ops)
    print(profiler.profiling())
