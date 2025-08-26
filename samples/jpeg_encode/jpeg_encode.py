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

import argparse
import vaststreamx as vsx


def argument_parser():
    parser = argparse.ArgumentParser("JPEG_ENCODER")
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=354,
        help="image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=474,
        help="image width",
    )
    parser.add_argument(
        "--input_file",
        default="../../../data/images/cat_354x474_nv12.yuv",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="jpeg_encode_result.jpg",
        help="output file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    # init card
    vsx.set_device(args.device_id)

    encoder = vsx.JpegEncoder()

    # prepare input data
    vsxImage = vsx.create_image(
        args.input_file,
        vsx.ImageFormat.YUV_NV12,
        args.width,
        args.height,
        vsx.Context.CPU(),
    )
    encoder.send_image(vsxImage)
    encoder.stop_send_image()

    # get output bytes
    output = encoder.recv_data()
    print(f"Encoded data bytes: {len(output)}")

    # write encoded bytes to output file
    with open(args.output_file, "wb") as f:
        f.write(output)
