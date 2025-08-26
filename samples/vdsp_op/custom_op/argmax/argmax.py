#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import argparse
import numpy as np
from argmax_op import ArgmaxOp
import ast

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../../../")
sys.path.append(common_path)


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
        "--elf_file",
        default="/opt/vastai/vaststreamx/data/elf/planar_argmax",
        help="elf file",
    )
    parser.add_argument(
        "-s",
        "--shape",
        help="model input shape",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()

    argmax_op = ArgmaxOp("planar_argmax_op", args.elf_file, args.device_id)
    shape = ast.literal_eval(args.shape)

    input = np.random.rand(*shape).astype(np.float16)
    output = argmax_op.process(input)

    print("output shape: ", output.shape)
