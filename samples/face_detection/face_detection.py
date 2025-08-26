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

from common.face_detector import FaceDetector
import common.utils as utils
import numpy as np
import cv2
import argparse
import vaststreamx as vsx


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--vdsp_params",
        default="./data/configs/retinaface_rgbplanar.json",
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
        "--input_file",
        default="../../../data/images/face.jpg",
        help="input file",
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
    parser.add_argument(
        "--output_file",
        default="face_det_result.jpg",
        help="output file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1
    model = FaceDetector(
        args.model_prefix, args.vdsp_params, batch_size, args.device_id
    )
    model.set_threshold(args.threshold)
    image_format = model.get_fusion_op_iimage_format()

    if args.dataset_filelist == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Failed to read input file: {args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        objects = model.process(vsx_image)

        print("Face bboxes and landmarks:")
        for i in range(len(objects)):
            if objects[i][0] < 0:
                break
            bbox = [
                objects[i][1],
                objects[i][2],
                objects[i][3],
                objects[i][4],
            ]
            landmarks = []
            point_count = int((len(objects[0]) - 5) / 2)
            for s in range(point_count):
                landmarks.append([objects[i][5 + s * 2 + 0], objects[i][5 + s * 2 + 1]])
            print(
                f"Index:{i},score: {objects[i][0]}, bbox: {bbox}, landmarks:{landmarks}"
            )
            cv2.rectangle(
                cv_image,
                pt1=(int(bbox[0]), int(bbox[1])),
                pt2=(int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                color=(0, 0, 255),
                thickness=1,
            )
            for landmark in landmarks:
                cv2.circle(
                    cv_image,
                    (int(landmark[0]), int(landmark[1])),
                    color=(0, 255, 0),
                    radius=2,
                    thickness=2,
                )
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
            save_dir = os.path.join(
                args.dataset_output_folder, fullname.split(os.sep)[-2]
            )
            os.makedirs(save_dir, exist_ok=True)
            outfile = open(os.path.join(save_dir, base_name + ".txt"), "wt")

            outfile.write(f"{image_file}\n")
            outfile.write(f"{len(objects)}\n")

            for i in range(len(objects)):
                if objects[i][0] < 0:
                    break
                bbox = [
                    objects[i][1],
                    objects[i][2],
                    objects[i][3],
                    objects[i][4],
                ]
                landmarks = []
                point_count = int((len(objects[0]) - 5) / 2)
                for s in range(point_count):
                    landmarks.append(
                        [objects[i][5 + s * 2 + 0], objects[i][5 + s * 2 + 1]]
                    )
                print(
                    f"Index:{i},score: {objects[i][0]}, bbox: {bbox}, landmarks:{landmarks}"
                )
                outfile.write(
                    f"{objects[i][1]} {objects[i][2]} {objects[i][3]} {objects[i][4]} {objects[i][0]}\n"
                )
            outfile.close()
