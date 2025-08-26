import vaststreamx as vsx
import os
import sys
import cv2
import argparse
import numpy as np
from brightness_op import BrightnessOp

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../../../")
sys.path.append(common_path)

import common.utils as utils


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
        "--input_file",
        default="data/images/dog.jpg",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="./custom_op_result.jpg",
        help="output file",
    )
    parser.add_argument(
        "--elf_file",
        default="/opt/vastai/vaststreamx/data/elf/brightness",
        help="elf file",
    )
    parser.add_argument(
        "--scale",
        default=2.2,
        type=float,
        help="brightness scale coefficient",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()

    brightness_op = BrightnessOp(
        "img_brightness_adjust", args.elf_file, args.device_id, args.scale
    )

    bgr888 = cv2.imread(args.input_file)
    yuv_nv12 = utils.cv_bgr888_to_nv12(bgr888)

    out_nv12 = brightness_op.process(yuv_nv12)
    cv_nv12 = vsx.as_numpy(out_nv12).squeeze()
    out_bgr888 = utils.cv_nv12_to_bgr888(cv_nv12)

    cv2.imwrite(args.output_file, out_bgr888)
    print("Write result to ", args.output_file)
