#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import vaststreamx as vsx

import argparse
import numpy as np


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--input_npz",
        default="",
        help="input npz",
    )
    parser.add_argument(
        "--output_npz",
        default="./tensor_out.npz",
        help="output file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    # init env
    vsx.set_device(args.device_id)

    # load npz file
    tensor_map = np.load(args.input_npz)
    input_tensors = []
    for k, v in tensor_map.items():
        print(f"key:{k}, tensor shape:{v.shape}")
        input_tensors.append(v)
    # copy to device
    tensors_vacc = []
    for tensor in input_tensors:
        tensors_vacc.append(vsx.from_numpy(tensor, args.device_id))
    # copy to host
    tensors_cpu = []
    for tensor in tensors_vacc:
        tensors_cpu.append(vsx.as_numpy(tensor))
    # change data
    if tensors_cpu[0].dtype == np.int32:
        tensors_cpu[0][0][0:10] = 10
    # write to npz file
    out_map = {}
    for i, tensor in enumerate(tensors_cpu):
        out_map["output_" + str(i)] = tensor
    np.savez(args.output_npz, **out_map)
