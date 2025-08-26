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
import common.utils as utils
from tracker.byte_tracker import BYTETracker
import argparse
import cv2


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_prefix", default="", help="model prefix of the model suite files"
    )
    parser.add_argument("--hw_config", help="hw-config file of the model suite")
    parser.add_argument(
        "--vdsp_params", default="", help="vdsp preprocess parameter file"
    )
    parser.add_argument(
        "-d", "--device_id", default=0, type=int, help="device id to run"
    )
    parser.add_argument(
        "--detect_threshold", default=0.01, type=float, help="detector threshold"
    )
    parser.add_argument(
        "--label_file", default="../../data/labels/coco2id.txt", help="label file"
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
    parser.add_argument("--input_file", default="", help="input file")
    parser.add_argument("--output_file", default="", help="output file")
    parser.add_argument("--dataset_filelist", default="", help="dataset image filelist")
    parser.add_argument("--dataset_root", default="", help="dataset image root")
    parser.add_argument("--dataset_result_file", default="", help="dataset result file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    labels = utils.load_labels(args.label_file)
    batch_size = 1
    model = Detector(args.model_prefix, args.vdsp_params, batch_size, args.device_id)
    model.set_threshold(args.detect_threshold)
    tracker = BYTETracker(
        args.track_thresh + 0.1, args.track_thresh, args.track_buffer, args.match_thresh
    )
    image_format = model.get_fusion_op_iimage_format()

    if args.dataset_filelist == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Failed to read {args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        objects = model.process(vsx_image)
        online_targets = tracker.update(objects)
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(float(t.score))
            print(
                f"Object class: {labels[int(t.det_class)]}, score: {t.score:.6f}, id: {tid}, bbox: [{tlwh[0]:.3f}, {tlwh[1]:.3f}, {tlwh[2]:.3f}, {tlwh[3]:.3f}]"
            )
            if args.output_file != "":
                cv2.rectangle(
                    cv_image,
                    (int(tlwh[0]), int(tlwh[1])),
                    (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])),
                    color=[0, 255, 0],
                    thickness=2,
                )
                cv2.putText(
                    cv_image,
                    labels[int(t.det_class)] + " " + f"{t.score:0.2f}",
                    (int(tlwh[0]), int(tlwh[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    [0, 255, 0],
                    thickness=1,
                )
        if args.output_file != "":
            cv2.imwrite(args.output_file, cv_image)
    else:
        filelist = []
        results = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = [line.strip() for line in f.readlines()]

        for frame_id, image_file in enumerate(filelist):
            fullname = os.path.join(args.dataset_root, image_file)
            cv_image = cv2.imread(fullname)
            assert cv_image is not None, f"Failed to read {fullname}"
            print(fullname)
            vsx_image = utils.cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )
            objects = model.process(vsx_image)
            online_targets = tracker.update(objects)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(float(t.score))
            # save results
            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))

        save_format = "{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n"
        with open(args.dataset_result_file, "w") as f:
            for frame_id, tlwhs, track_ids, scores in results:
                for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
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
                    f.write(line)
