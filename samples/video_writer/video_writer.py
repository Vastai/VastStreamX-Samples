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
import numpy as np


current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

from common.media_encode import MediaEncode
import argparse


def read_frames(input_file, width, height):
    frames = []
    frames_num = 0
    with open(input_file, "rb") as file:
        while True:
            data = file.read((int)(width * height * 3 / 2))
            if len(data) == 0:
                break
            frames.append(data[:])
            frames_num += 1
    return frames, frames_num


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
        "--width",
        default=0,
        type=int,
        help="frame width",
    )
    parser.add_argument(
        "--height",
        default=0,
        type=int,
        help="frame height",
    )
    parser.add_argument(
        "--frame_rate",
        default=30,
        type=int,
        help="frame rate",
    )
    parser.add_argument(
        "--input_file",
        default="",
        help="video file",
    )
    parser.add_argument(
        "--output_uri",
        default="",
        help="output uri",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    codec_type = vsx.CodecType.CODEC_TYPE_H264
    if args.codec_type == "H264" or args.codec_type == "h264":
        codec_type = vsx.CodecType.CODEC_TYPE_H264
    elif args.codec_type == "H265" or args.codec_type == "h265":
        codec_type = vsx.CodecType.CODEC_TYPE_H265
    else:
        print(f"undefined codec_type:{args.codec_type}")
        exit(-1)

    input_file = args.input_file
    uri = args.output_uri

    frames, frames_num = read_frames(input_file, args.width, args.height)

    writer = vsx.VideoWriter(
        uri, args.frame_rate, codec_type, 4000000, args.frame_rate, args.device_id
    )

    pts = 0

    for frame in frames:
        img_arr = np.frombuffer(frame, dtype=np.uint8)
        writable_array = np.require(
            img_arr, dtype=img_arr.dtype, requirements=["O", "w"]
        )
        vsx_image = vsx.create_image(
            array=writable_array,
            format=vsx.ImageFormat.YUV_NV12,
            width=args.width,
            height=args.height,
            device_id=args.device_id,
        )
        frame_attr = vsx.FrameAttr()
        frame_attr.frame_dts = pts
        frame_attr.frame_pts = pts
        pts += 1
        writer.write(vsx_image, frame_attr)
    writer.release()
