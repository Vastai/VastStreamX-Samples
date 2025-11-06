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
import vaststreamx as vsx
import multiprocessing
from tracker.byte_tracker import BYTETracker
import argparse
import shutil


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_prefix", default="", help="model prefix of the model suite files"
    )
    parser.add_argument(
        "--vdsp_params", default="", help="vdsp preprocess parameter file"
    )
    parser.add_argument(
        "-d", "--device_id", default=0, type=int, help="device id to run"
    )
    parser.add_argument(
        "--detect_thresh", default=0.01, type=float, help="detector threshold"
    )
    # tracker params
    parser.add_argument(
        "--track_thresh", type=float, default=0.6, help="tracking confidence threshold"
    )
    parser.add_argument(
        "--track_buffer", type=int, default=30, help="the frames for keep lost tracks"
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=0.9,
        help="matching threshold for tracking",
    )
    parser.add_argument(
        "--min_box_area", type=float, default=100, help="filter out tiny boxes"
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
    parser.add_argument("--uri_list", default="", help="input uri list file")
    parser.add_argument(
        "--disable_encoder", action="store_true", help="enable encoder or not"
    )
    parser.add_argument("--save_output", action="store_true", help="save output or not")

    args = parser.parse_args()
    return args

def process(args, input_uri, cus_device_id, index=0, loop=1):
    vsx.set_device(cus_device_id)
    for inn in range(loop):
        print(
            f"Process {index} on device {cus_device_id} for uri:{input_uri}, loop {inn}"
        )
        batch_size = 1
        detector = Detector(
            args.model_prefix, args.vdsp_params, batch_size, cus_device_id
        )
        detector.set_threshold(args.detect_thresh)
        tracker = BYTETracker(
            args.track_thresh + 0.1, args.track_thresh, args.track_buffer, args.match_thresh
        )

        cap = vsx.VideoCapture(
            input_uri, vsx.CaptureMode.FULLSPEED_MODE, cus_device_id, True
        )

        ret, frame, frame_attr = cap.read()
        assert ret

        frame_rate = int(frame_attr.video_fps)
        if frame_attr.codec_info.find("avc1") != -1:
            codec_type = vsx.CODEC_TYPE_H264
        elif frame_attr.codec_info.find("hevc") != -1:
            codec_type = vsx.CODEC_TYPE_HEVC
        else:
            print(f"undefined codec_type:{frame_attr.codec_info}")
            exit(-1)

        writer = None
        if not args.disable_encoder:
            default_bit_rate = 8000000
            key_frame_interval = frame_rate
            writer = vsx.VideoWriter(
                args.output_path + f"/channel_{index}.ts",
                frame_rate,
                codec_type,
                default_bit_rate,
                key_frame_interval,
                cus_device_id,
            )
        
        text_file = open(args.output_path + f"/channel_{index}.txt","wt")
        save_format = "{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n"

        frame_id = 0
        while True:
            frame_id += 1
            objects = detector.process(frame)
            track_targets = tracker.update(objects)
            track_tlwhs = []
            track_ids = []
            track_scores = []
            bboxes=[]
            for t in track_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    track_tlwhs.append(tlwh)
                    track_ids.append(tid)
                    track_scores.append(float(t.score))
                    bboxes.append([int(tlwh[0]),int(tlwh[1]),int(tlwh[2]),int(tlwh[3])])
            # write video frame
            if writer:
                vsx_image = vsx.draw_rectangle(frame, bboxes, vsx.YUV_RED, 2, True)
                writer.write(vsx_image, frame_attr)
            # write track result
            for tlwh, track_id, score in zip(track_tlwhs, track_ids, track_scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=round(x1, 1),
                    y1=round(y1, 1),
                    w=round(w, 1),
                    h=round(h, 1),
                    s=round(score, 2),
                )
                text_file.write(line)

            ret, frame, frame_attr = cap.read()
            if ret == 0:
                break

        cap.release()
        if writer:
            writer.release()
        text_file.close()

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

    for index, input_uri in enumerate(input_list):
        cus_device_id = device_id + int(index / split_num)
        print(f"input_uri:{input_uri},device_id:{cus_device_id}")
        p = multiprocessing.Process(
            target=process, args=(args, input_uri, cus_device_id, index, args.loop)
        )
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()