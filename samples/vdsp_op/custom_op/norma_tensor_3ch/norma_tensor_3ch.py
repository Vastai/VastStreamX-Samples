import os
import sys
import argparse
import numpy as np
from norma_tensor_3ch_op import NormaTensor3ChOp
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
        default="/opt/vastai/vaststreamx/data/elf/norma_tensor_3ch",
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

    argmax_op = NormaTensor3ChOp("norma_tensor_3ch_op", args.elf_file, args.device_id)
    shape = ast.literal_eval(args.shape)

    input = np.random.rand(*shape).astype(np.int8)

    output = argmax_op.process(input)

    print("output shape: ", output.shape)
