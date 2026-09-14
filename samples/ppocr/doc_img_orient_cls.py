#
# Copyright (C) 2026 Vastai-tech Company.
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

from common.doc_img_orient_cls import DocImgOrientClassifier
import common.utils as utils

import cv2
import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/doc_ori_int8_mse_1-3-224-224/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--vdsp_params",
        default="./data/configs/doc_ori_bgr888.json",
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
        "--label_file",
        default="data/labels/ppocr_cls_labels.txt",
        help="label file",
    )
    parser.add_argument(
        "--input_file",
        default="data/images/img_rot180.jpg",
        help="input file",
    )
    parser.add_argument(
        "--dataset_val_file",
        default="",
        help="dataset validation file",
    )
    parser.add_argument(
        "--dataset_root",
        default="",
        help="input dataset root",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    lines = utils.load_labels(args.label_file)
    labels={}
    for line in lines:
        label, angle = line.strip().split()
        labels[int(label)] = int(angle)
    
    batch_size = 1
    model = DocImgOrientClassifier(args.model_prefix, args.vdsp_params, batch_size, args.device_id)
    image_format = model.get_fusion_op_iimage_format()

    if args.dataset_val_file == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Failed to read input file: {args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        index, score = model.process(vsx_image)
        print(
            f"Image angle: {labels[index]}, confidence: {score:.4f}"
        )
    else:
        filelist = []
        gt={}
        with open(args.dataset_val_file, "rt") as f:
            lines = f.readlines()
            for line in lines:
                filename, label = line.strip().split()
                filelist.append(filename)
                gt[filename] = int(label)        
        correct_count = 0
    
        for file in filelist:
            fullname = os.path.join(args.dataset_root, file)
            print(fullname)
            cv_image = cv2.imread(fullname)
            assert cv_image is not None, f"Failed to read input file: {fullname}"
            vsx_image = utils.cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )
            index, score = model.process(vsx_image)
            if index == gt.get(file, -1):
                correct_count += 1
            print(f"Image: {file}, Label: {gt.get(file, -1)}, Predicted: {index}, Correct: {correct_count}")
        print(f"Accuracy: {correct_count / len(filelist):.4f}")
