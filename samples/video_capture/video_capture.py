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
    parser = argparse.ArgumentParser("VIDEO_CAPTURE")
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--input_uri",
        default="../../../data/videos/test.mp4",
        help="input file",
    )
    parser.add_argument(
        "--frame_count",
        default=10,
        type=int,
        help="frame count to save",
    )
    parser.add_argument(
        "--output_folder",
        default="output",
        help="output folder",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    # initialize decoder
    cap = vsx.VideoCapture(
        args.input_uri, vsx.CaptureMode.FULLSPEED_MODE, args.device_id
    )
    frame_count = args.frame_count
    os.makedirs(args.output_folder, exist_ok=True)
    count = 0
    while count < frame_count:
        ret, frame, frame_attr = cap.read()
        if ret:
            rgb_image = vsx.cvtcolor(
                frame, vsx.ImageFormat.RGB_PLANAR, vsx.ImageColorSpace.COLOR_SPACE_BT601
            )
            cv_mat = utils.vsximage_to_cv_bgr888(rgb_image)
            cv2.imwrite(os.path.join(args.output_folder, f"frame_{count}.bmp"), cv_mat)
            count += 1
        else:
            break
    print(f"Read {count} frames.")
    cap.release()
