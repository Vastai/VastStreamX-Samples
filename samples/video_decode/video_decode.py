#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
from threading import Thread
import vaststreamx as vsx
import hashlib

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

from common.vdecoder import Vdecoder
from common.utils import load_labels
from pathlib import Path

import numpy as np
import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--codec_type",
        default="H264",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--input_file",
        default="",
        help="video file",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        help="output folder",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    device_id = args.device_id

    assert vsx.set_device(device_id) == 0, f"Failed to set device id {device_id}"

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

    input_file = args.input_file
    output_folder = args.output_folder
    vdecoder = Vdecoder(codec_type, device_id, input_file)

    def get_frame_thread():
        index = 0
        while True:
            try:
                vast_image = vdecoder.get_result()
            except ValueError as e:
                break
            cpu_image = vsx.as_numpy(vast_image).squeeze()
            width, height = vast_image.width, vast_image.height
            height_pitch = vast_image.heightpitch

            y = cpu_image[:height, :width]
            uv = cpu_image[height_pitch : height_pitch + height // 2, :width]
            nv12 = np.concatenate((y, uv), axis=0)

            output_file = (
                output_folder
                + "/"
                + str(vast_image.width)
                + "x"
                + str(vast_image.height)
                + "_"
                + str(index)
                + ".yuv"
            )
            with open(output_file, "wb") as f:
                f.write(nv12)
            print(f"write yuv file: {output_file}")
            index += 1

    recv_thread = Thread(target=get_frame_thread)
    recv_thread.start()

    while True:
        media_data = vdecoder.get_test_data(False)
        if media_data is None:
            vdecoder.stop()
            break
        else:
            vdecoder.process(media_data)

    recv_thread.join()
