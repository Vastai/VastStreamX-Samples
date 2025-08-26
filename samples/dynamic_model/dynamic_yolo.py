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

from common.dynamic_detector import DynamicDetector
from common.utils import load_labels
import common.utils as utils
import cv2
import argparse
import ast


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--module_info",
        default="/opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json",
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
        default=0.5,
        type=float,
        help="device id to run",
    )
    parser.add_argument(
        "--max_input_shape",
        default="[1,3,640,640]",
        help="model max input shape",
    )
    parser.add_argument(
        "--label_file",
        default="../../../data/labels/coco2id.txt",
        help="label file",
    )
    parser.add_argument(
        "--input_file",
        default="../../../data/images/dog.jpg",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="dynamic_result.jpg",
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
    labels = load_labels(args.label_file)
    max_input_shape = ast.literal_eval(args.max_input_shape)
    batch_size = 1
    dynamic_model = DynamicDetector(
        args.module_info,
        args.vdsp_params,
        [max_input_shape],
        batch_size,
        args.device_id,
    )
    model_min_input_size, model_max_input_size = 320, 640
    dynamic_model.set_threshold(args.threshold)
    image_format = dynamic_model.get_fusion_op_iimage_format()

    if args.dataset_filelist == "":
        image = cv2.imread(args.input_file)
        assert image is not None, f"Failed to open {args.input_file}"
        ori_h, ori_w, c = image.shape
        input_size = ori_w if ori_w > ori_h else ori_h
        if input_size % 2 != 0:
            input_size += 1
        if input_size > model_max_input_size:
            input_size = model_max_input_size
        elif input_size < model_min_input_size:
            input_size = model_min_input_size
        dynamic_model.set_model_input_shape([[1, 3, input_size, input_size]])
        vsx_image = utils.cv_bgr888_to_vsximage(image, image_format, args.device_id)
        objects = dynamic_model.process(vsx_image)
        print("Detection objects:")
        for obj in objects:
            if obj[0] > 0:
                bbox = [int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])]
                print(
                    f"Object class: {labels[int(obj[0])]}, score: {obj[1]}, bbox: {bbox}"
                )
            else:
                break
        if args.output_file != "":
            for obj in objects:
                if obj[0] > 0:
                    cv2.rectangle(
                        image,
                        pt1=(int(obj[2]), int(obj[3])),
                        pt2=(int(obj[2] + obj[4]), int(obj[3] + obj[5])),
                        color=(0, 0, 255),
                        thickness=1,
                    )
                else:
                    break
            cv2.imwrite(args.output_file, image)
    else:
        filelist = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = [line.strip() for line in f.readlines()]
        for image_file in filelist:
            fullname = os.path.join(args.dataset_root, image_file)
            image = cv2.imread(fullname)
            assert image is not None, f"Failed to open {fullname}"
            ori_h, ori_w, c = image.shape
            input_size = ori_w if ori_w > ori_h else ori_h
            if input_size % 2 != 0:
                input_size += 1
            if input_size > model_max_input_size:
                input_size = model_max_input_size
            elif input_size < model_min_input_size:
                input_size = model_min_input_size
            dynamic_model.set_model_input_shape([[1, 3, input_size, input_size]])
            vsx_image = utils.cv_bgr888_to_vsximage(image, image_format, args.device_id)
            objects = dynamic_model.process(vsx_image)
            base_name, _ = os.path.splitext(os.path.basename(image_file))
            outfile = open(
                os.path.join(args.dataset_output_folder, base_name + ".txt"), "wt"
            )
            print(f"{image_file} detection objects:")
            for obj in objects:
                if obj[1] >= 0:
                    bbox = [int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])]
                    print(
                        f"Object class: {labels[int(obj[0])]}, score: {obj[1]}, bbox: {bbox}"
                    )
                    outfile.write(
                        f"{labels[int(obj[0])]} {obj[1]} {obj[2]:.3f} {obj[3]:.3f} {(obj[2]+obj[4]):.3f} {(obj[3]+obj[5]):.3f}\n"
                    )
                else:
                    break
            outfile.close()
