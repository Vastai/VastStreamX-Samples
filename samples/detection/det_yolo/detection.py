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
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)

from common.detector import Detector
import common.utils as utils

import cv2
import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/yolov5m-int8-max-1_3_640_640-vacc-pipeline/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        help="hw-config file of the model suite",
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
        default=0.5,
        type=float,
        help="device id to run",
    )
    parser.add_argument(
        "--label_file",
        default="data/labels/coco2id.txt",
        help="label file",
    )
    parser.add_argument(
        "--input_file",
        default="data/images/dog.jpg",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="",
        help="output file",
    )
    parser.add_argument(
        "--dataset_filelist",
        default="",
        help="dataset filename list",
    )
    parser.add_argument(
        "--dataset_root",
        default="",
        help="dataset root",
    )
    parser.add_argument(
        "--dataset_output_folder",
        default="",
        help="dataset output folder path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    labels = utils.load_labels(args.label_file)
    batch_size = 1
    model = Detector(args.model_prefix, args.vdsp_params, batch_size, args.device_id)
    model.set_threshold(args.threshold)
    image_format = model.get_fusion_op_iimage_format()

    if args.dataset_filelist == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Failed to read input file: {args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        objects = model.process(vsx_image)
        print("Detection objects:")
        for obj in objects:
            if obj[1] >= 0:
                bbox_str = f"[{obj[2]:.2f}, {obj[3]:.2f}, {obj[4]:.2f}, {obj[5]:.2f}]"
                print(
                    f"Object class: {labels[int(obj[0])]}, score: {obj[1]:.4f}, bbox: {bbox_str}"
                )
            else:
                break
        if args.output_file != "":
            for obj in objects:
                if obj[1] >= 0:
                    cv2.rectangle(
                        cv_image,
                        pt1=(int(obj[2]), int(obj[3])),
                        pt2=(int(obj[2] + obj[4]), int(obj[3] + obj[5])),
                        color=(0, 0, 255),
                        thickness=1,
                    )
                else:
                    break
            cv2.imwrite(args.output_file, cv_image)
    else:
        filelist = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = [line.strip() for line in f.readlines()]
        for image_file in filelist:
            fullname = os.path.join(args.dataset_root, image_file)
            cv_image = cv2.imread(fullname)
            assert cv_image is not None, f"Failed to read image:{fullname}"
            vsx_image = utils.cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )
            objects = model.process(vsx_image)
            base_name, _ = os.path.splitext(os.path.basename(image_file))
            outfile = open(
                os.path.join(args.dataset_output_folder, base_name + ".txt"), "wt"
            )
            print(f"{image_file} detection objects:")
            for obj in objects:
                if obj[1] >= 0:
                    bbox_str = (
                        f"[{obj[2]:.2f}, {obj[3]:.2f}, {obj[4]:.2f}, {obj[5]:.2f}]"
                    )
                    print(
                        f"Object class: {labels[int(obj[0])]}, score: {obj[1]:.4f}, bbox: {bbox_str}"
                    )
                    outfile.write(
                        f"{labels[int(obj[0])]} {obj[1]} {obj[2]:.2f} {obj[3]:.2f} {(obj[2]+obj[4]):.2f} {(obj[3]+obj[5]):.2f}\n"
                    )
                else:
                    break
            outfile.close()
