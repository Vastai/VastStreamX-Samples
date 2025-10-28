#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import shutil

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

from common.detector import Detector
import argparse
import vaststreamx as vsx
import multiprocessing
import time
import cv2
import os
import numpy as np
import common.utils as utils
from threading import Thread
from pydantic import BaseModel
from typing import List


class DetectionRect(BaseModel):
    object_id: int
    score: float
    bbox: List[int]


class DetectionSummary(BaseModel):
    channel_index: int
    frame_index: int
    rect: List[DetectionRect]


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        type=str,
        required=True,
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--vdsp_params",
        type=str,
        required=True,
        help="vdsp preprocess parameter file",
    )
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run [ default: 0 ]",
    )
    parser.add_argument(
        "--threshold",
        default=0.1,
        type=float,
        help="threshold for detection [default: 0.1]",
    )
    parser.add_argument(
        "--uri",
        type=str,
        required=True,
        help="uri to decode",
    )
    parser.add_argument(
        "--output_path",
        default="output_result",
        type=str,
        help="output path [ default: output_result ]",
    )
    parser.add_argument(
        "--num_channels",
        default=1,
        type=int,
        help="number of channels to decode [ default: 1 ]",
    )
    parser.add_argument(
        "--loop",
        default=1,
        type=int,
        help="loop count for each channel [ default: 1 ]",
    )
    parser.add_argument("--keep", default=1, type=int, help="keep num [default:1]")
    parser.add_argument("--drop", default=0, type=int, help="drop num [default:0]")
    parser.add_argument("--uri_list", default="", help="input uri list file")
    parser.add_argument(
        "--disable_encoder", action="store_true", help="enable encoder or not"
    )
    parser.add_argument("--save_output", action="store_true", help="save output or not")

    args = parser.parse_args()
    return args


def process(args, input_uri, cus_device_id, gap, index=0, loop=1):
    save_path = os.path.join(args.output_path, f"channel_{index}")
    if args.save_output:
        os.makedirs(save_path, exist_ok=True)
    vsx.set_device(cus_device_id)
    for inn in range(loop):
        print(
            f"Process {index} on device {cus_device_id} for uri:{input_uri}, loop {inn}"
        )
        batch_size = 1
        detector = Detector(
            args.model_prefix, args.vdsp_params, batch_size, cus_device_id
        )
        detector.set_threshold(args.threshold)

        cap = vsx.VideoCapture(
            input_uri, vsx.CaptureMode.FULLSPEED_MODE, cus_device_id, True
        )

        ret, frame, frame_attr = cap.read()
        assert ret

        frame_rate = int(frame_attr.video_fps / gap)
        suffix = "h264"
        if frame_attr.codec_info.find("avc1") != -1:
            codec_type = vsx.CODEC_TYPE_H264
        elif frame_attr.codec_info.find("hevc") != -1:
            codec_type = vsx.CODEC_TYPE_HEVC
            suffix = "h265"
        else:
            print(f"undefined codec_type:{frame_attr.codec_info}")
            exit(-1)

        encoder = None
        consumer = None
        if not args.disable_encoder:
            encoder = vsx.VideoEncoder(
                codec_type,
                frame.width,
                frame.height,
                {"frame_rate_denominator": 1, "frame_rate_numerator": frame_rate},
            )

            def encoder_consumer():
                idx = 0
                while True:
                    try:
                        datas = encoder.recv_data()
                        if datas is None:
                            break
                        filename = os.path.join(save_path, f"frame_{idx:05d}.{suffix}")
                        if args.save_output:
                            with open(filename, "wb") as file:
                                file.write(datas)
                        idx += 1
                    except:
                        print("receive all for channel ", index)
                        break

            consumer = Thread(target=encoder_consumer)
            consumer.start()

        count = 0
        objs_result = []
        while True:
            count += 1
            if count <= args.keep:
                objects = detector.process(frame)
                objs_result.append(objects)
                bboxs = []
                for obj in objects:
                    if obj[1] >= 0:
                        # x,y,w,h
                        bbox = [int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])]
                        bboxs.append(bbox)
                    else:
                        break
                vsx_image = vsx.draw_rectangle(frame, bboxs, vsx.YUV_BLUE, 2, True)

                if encoder:
                    encoder.send_image(vsx_image, frame_attr)
            else:
                vsx_image = vsx.draw_rectangle(frame, bboxs, vsx.YUV_BLUE, 2, False)
                if encoder:
                    encoder.send_image(vsx_image, frame_attr)

            ret, frame, frame_attr = cap.read()
            if ret == 0:
                break

            if count == args.keep + args.drop:
                count = 0

        cap.release()
        if encoder:
            encoder.stop_send_image()
        if consumer:
            consumer.join()

        if args.save_output:
            summary = os.path.join(save_path, "detection_summary.txt")
            with open(summary, "w") as f:
                for frame_idx, objs in enumerate(objs_result):
                    rects = []
                    for obj in objs:
                        if obj[1] >= 0:
                            bbox = [
                                int(obj[2]),  # x
                                int(obj[3]),  # y
                                int(obj[4]),  # w
                                int(obj[5]),  # h
                            ]
                            rects.append(
                                DetectionRect(object_id=obj[0], score=obj[1], bbox=bbox)
                            )
                    detection_summary = DetectionSummary(
                        channel_index=index, frame_index=frame_idx, rect=rects
                    )
                    f.write(detection_summary.model_dump_json() + "\n")


if __name__ == "__main__":
    args = argument_parser()
    process_list = []

    input_list = []
    if args.uri_list != "":
        with open(args.uri_list, "rt") as f:
            input_list = [line.strip() for line in f.readlines()]
        print(f"input_list:{input_list}")
    elif args.uri != "":
        for i in range(args.num_channels):
            input_list.append(args.uri)
    if args.save_output:
        if args.output_path == "":
            print("please set output_path")
            exit(-1)
        if os.path.exists(args.output_path):
            shutil.rmtree(args.output_path)
        os.makedirs(args.output_path, exist_ok=True)
    assert len(input_list) == args.num_channels
    device_id = args.device_id
    split_num = args.num_channels / 2
    if split_num == 0:
        split_num = 1

    gap = args.drop / args.keep + 1

    for index, input_uri in enumerate(input_list):
        cus_device_id = device_id + int(index / split_num)
        print(f"input_uri:{input_uri},device_id:{cus_device_id}")
        p = multiprocessing.Process(
            target=process, args=(args, input_uri, cus_device_id, gap, index, args.loop)
        )
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()
