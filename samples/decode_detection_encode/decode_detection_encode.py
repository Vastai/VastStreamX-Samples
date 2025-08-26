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

from common.detector import Detector
import argparse
import vaststreamx as vsx
import time
import cv2
import os
import numpy as np
import common.utils as utils


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/"
        "yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--vdsp_params",
        default="./data/configs/yolo_div255_bgr888.json",
        help="vdsp preprocess parameter file",
    )
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--threshold",
        default=0.1,
        type=float,
        help="threshold for detection",
    )
    parser.add_argument(
        "--label_file",
        default="data/labels/coco2id.txt",
        help="label file",
    )
    parser.add_argument(
        "--input_uri",
        default="data/videos/test.mp4",
        type=str,
        help="uri to decode",
    )
    parser.add_argument(
        "--output_file",
        default="",
        type=str,
        help="output path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1
    detector = Detector(args.model_prefix, args.vdsp_params, batch_size, args.device_id)
    detector.set_threshold(args.threshold)
    labels = utils.load_labels(args.label_file)

    cap = vsx.VideoCapture(
        args.input_uri, vsx.CaptureMode.FULLSPEED_MODE, args.device_id, True
    )

    ret, frame, frame_attr = cap.read()
    assert ret

    print(f"Frame width: {frame.width}, height: {frame.height}")
    print(f"Frame rate: {frame_attr.video_fps}")
    print(f"Frame codec_info: {frame_attr.codec_info}")
    print(f"Frame color_space: {frame_attr.color_space}")

    frame_rate = int(frame_attr.video_fps)
    if frame_attr.codec_info.find("avc1") != -1:
        codec_type = vsx.CODEC_TYPE_H264
    elif frame_attr.codec_info.find("hevc") != -1:
        codec_type = vsx.CODEC_TYPE_HEVC
    else:
        print(f"undefined codec_type:{frame_attr.codec_info}")
        exit(-1)

    writer = None
    if args.output_file:
        default_bit_rate = 0
        key_frame_interval = 25
        writer = vsx.VideoWriter(
            args.output_file,
            frame_rate,
            codec_type,
            default_bit_rate,
            key_frame_interval,
            args.device_id,
        )

    while True:
        objects = detector.process(frame)
        bgr_planar = vsx.cvtcolor(frame, vsx.ImageFormat.BGR_PLANAR)
        print(f"bgr_planar width:{bgr_planar.width}")
        print(f"bgr_planar height:{bgr_planar.height}")
        print(f"bgr_planar widthpitch:{bgr_planar.widthpitch}")
        print(f"bgr_planar heightpitch:{bgr_planar.heightpitch}")
        print(f"bgr_planar format:{bgr_planar.format}")
        b_plane, g_plane, r_plane = vsx.as_numpy(bgr_planar)
        bgr_interleave = np.dstack((b_plane, g_plane, r_plane))
        for obj in objects:
            if obj[1] >= 0:
                bbox = [int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])]
                print(
                    f"Object class: {labels[int(obj[0])]}, score: {obj[1]:.4f}, bbox: {bbox}"
                )
                cv2.rectangle(
                    bgr_interleave,
                    (bbox[0], bbox[1]),
                    (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    (255, 0, 0),
                    2,
                )
            else:
                break
        yuv_nv12 = utils.cv_bgr888_to_nv12(bgr_interleave)
        vsx_image = vsx.create_image(
            yuv_nv12,
            vsx.ImageFormat.YUV_NV12,
            frame.width,
            frame.height,
            vsx.Context.CPU(),
        )
        if writer:
            writer.write(vsx_image, frame_attr)

        ret, frame, frame_attr = cap.read()
        if ret == 0:
            break
    cap.release()
    if writer:
        writer.release()
