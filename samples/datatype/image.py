#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import vaststreamx as vsx

import cv2
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
        "--input_file",
        default="data/images/dog.jpg",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="./image_out.jpg",
        help="output file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    # init env
    vsx.set_device(args.device_id)
    # read file by opencv
    cv_image = cv2.imread(args.input_file)
    print(f"input opencv image shape:{cv_image.shape}")
    # make vsx image, memory is in device
    vsx_image = vsx.create_image(
        cv_image,
        vsx.ImageFormat.BGR_INTERLEAVE,
        cv_image.shape[1],
        cv_image.shape[0],
        args.device_id,
    )
    print(
        f"vsx_image shape: {vsx_image.shape},width: {vsx_image.width}, height: {vsx_image.height}"
    )
    # copy to host
    image_cpu_np = vsx.as_numpy(vsx_image)
    print(f"image_cpu_np shape:{image_cpu_np.shape}")

    # change pexils
    for h in range(10):
        for w in range(10):
            image_cpu_np[h][w][:] = [w, w, w]
    # write to file
    cv2.imwrite(args.output_file, image_cpu_np)
