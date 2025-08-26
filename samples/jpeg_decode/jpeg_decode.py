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


def argument_parser():
    parser = argparse.ArgumentParser("JPEG_DECODER")
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--input_file",
        default="../../../data/images/cat.jpg",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="jpeg_decode_result.bmp",
        help="output file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()

    # init card
    vsx.set_device(args.device_id)

    decoder = vsx.JpegDecoder()

    # prepare input bytes
    data = None
    with open(args.input_file, "rb") as f:
        data = f.read()

    decoder.send_data(data)
    decoder.stop_send_data()

    # get decoded image
    output = decoder.recv_image()
    # width_align: 64， height_align 4
    print(
        f"Output image width: {output.width}, height: {output.height}, width_pitch: {output.widthpitch}, height_pitch: {output.heightpitch}"
    )
    print(f"Decoded image format is {output.format}")

    # copy to cpu memory
    cpu_image = vsx.as_numpy(output).squeeze()
    print(f"cpu_image shape: {cpu_image.shape}")

    # unpitch
    y = cpu_image[: output.height, : output.width]
    uv = cpu_image[
        output.heightpitch : output.heightpitch + output.height // 2, : output.width
    ]
    unpitch = np.vstack((y, uv))

    # write nv12 to binary file
    with open(args.output_file, "wb") as f:
        f.write(unpitch)

    # convert nv12 to bgr and write to file
    cv_bgr888 = cv2.cvtColor(unpitch, cv2.COLOR_YUV2BGR_NV12)
    cv2.imwrite(args.output_file.split(".")[-2] + ".bmp", cv_bgr888)
