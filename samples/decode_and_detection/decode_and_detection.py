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
import multiprocessing
import time
import cv2
import os


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
        "--uri",
        default="data/videos/test.mp4",
        type=str,
        help="uri to decode",
    )
    parser.add_argument(
        "--output_path",
        default="",
        type=str,
        help="output path",
    )
    parser.add_argument(
        "--num_channels",
        default=1,
        type=int,
        help="number of channels to decode",
    )
    args = parser.parse_args()
    return args


def draw_results(yuv_frame, objects, idx, output_path):
    yuv_npy = vsx.as_numpy(yuv_frame).squeeze()
    img_rgb = cv2.cvtColor(yuv_npy, cv2.COLOR_YUV2BGR_NV12)
    for obj in objects:
        if obj[1] >= 0:
            bbox = [int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])]
            cv2.rectangle(
                img_rgb,
                (bbox[0], bbox[1]),
                (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                (255, 0, 0),
                2,
            )
        else:
            break
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(f"{output_path}/img_{idx}.png", img_rgb)
    return img_rgb


def process(args, index=0):
    batch_size = 1
    detector = Detector(args.model_prefix, args.vdsp_params, batch_size, args.device_id)
    detector.set_threshold(args.threshold)

    cap = vsx.VideoCapture(
        args.uri, vsx.CaptureMode.FULLSPEED_MODE, args.device_id, True
    )

    cnt = 0
    tick = time.time()
    while True:
        ret, image, _ = cap.read()
        if ret:
            objects = detector.process(image)
            if args.output_path:
                draw_results(image, objects, cnt, args.output_path)
            tock = time.time()
            cnt += 1
            print(f"{index}th Decode+AI @ {cnt/(tock-tick)} fps")
        else:
            print("cap.read() returns 0")
            break

    cap.release()


if __name__ == "__main__":
    args = argument_parser()
    process_list = []
    for i in range(args.num_channels):
        p = multiprocessing.Process(target=process, args=(args, i))
        process_list.append(p)
        p.start()
        time.sleep(0.5)

    for p in process_list:
        p.join()
