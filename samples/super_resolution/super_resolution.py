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

from common.model_cv import ModelCV
import numpy as np
import cv2
import argparse
import ast
import common.utils as utils


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--vdsp_params",
        default="./data/configs/rcan_bgr888.json",
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
        "--denorm",
        default="[0,1,1]",
        help="denormalization params [mean, std, scale]",
    )
    parser.add_argument(
        "--input_file",
        default="../../../data/images/hd_1920x1080.png",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="./sr_result.png",
        help="output file",
    )
    parser.add_argument(
        "--dataset_filelist",
        default="",
        help="dataset filelst",
    )
    parser.add_argument(
        "--dataset_root",
        default="",
        help="input dataset root",
    )
    parser.add_argument(
        "--dataset_output_folder",
        default="",
        help="dataset output folder",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1
    model = ModelCV(
        args.model_prefix, args.vdsp_params, batch_size, args.device_id, args.hw_config
    )
    image_format = model.get_fusion_op_iimage_format()

    mu, std, scale = ast.literal_eval(args.denorm)

    if args.dataset_filelist == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Failed to read input file: {args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        output = model.process(vsx_image)
        output = np.array(output).squeeze().astype(np.float32)
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        output = np.clip(np.floor((output * std + mu) * scale + 0.5), 0, 255).astype(
            np.uint8
        )
        cv2.imwrite(args.output_file, output)
    else:
        filelist = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = [line.strip() for line in f.readlines()]
        for file in filelist:
            fullname = os.path.join(args.dataset_root, file)
            print(fullname)
            cv_image = cv2.imread(fullname)
            assert cv_image is not None, f"Failed to read input file: {fullname}"
            vsx_image = utils.cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )
            output = model.process(vsx_image)
            output = np.array(output).squeeze().astype(np.float32)
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
            output = np.clip(
                np.floor((output * std + mu) * scale + 0.5), 0, 255
            ).astype(np.uint8)
            outfile = os.path.join(
                args.dataset_output_folder, os.path.basename(fullname)
            )
            cv2.imwrite(outfile, output)
