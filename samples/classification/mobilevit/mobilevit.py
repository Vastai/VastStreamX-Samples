import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)

from common.mobile_vit import MobileVit
import common.utils as utils

import numpy as np
import cv2
import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/mobilevit_s-fp16-none-1_3_224_224-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--vdsp_params",
        default="../../../../data/configs/mobilevit_rgbplanar.json",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "-d",
        "--device_id",
        default=0,
        type=int,
        help="device id to run",
    )
    parser.add_argument(
        "--input_file",
        default="../../../../data/images/cat.jpg",
        help="input file",
    )
    parser.add_argument(
        "--label_file",
        default="../../../../data/labels/imagenet.txt",
        help="label file",
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
        "--dataset_output_file",
        default="",
        help="dataset output file",
    )
    args = parser.parse_args()
    return args


def load_labels(file):
    with open(file) as fin:
        return [line.strip() for line in fin.readlines()]


if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1
    labels = load_labels(args.label_file)

    model = MobileVit(
        args.model_prefix,
        args.vdsp_params,
        batch_size=batch_size,
        device_id=args.device_id,
    )
    image_format = model.get_fusion_op_iimage_format()

    if args.dataset_filelist == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Failed to read input file: {args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        result = model.process(vsx_image)
        index = np.argsort(result)[0][::-1]
        print("Top5:")
        for i in range(5):
            print(
                f"{i}th, score: {result[0, index[i]]:.4f}, class name: {labels[index[i]]}"
            )
    else:
        with open(args.dataset_filelist, "rt") as f:
            filelist = [line.strip() for line in f.readlines()]
        with open(args.dataset_output_file, "wt") as fout:
            for file in filelist:
                fullname = os.path.join(args.dataset_root, file)
                print(fullname)
                cv_image = cv2.imread(fullname)
                assert cv_image is not None, f"Failed to read input file: {fullname}"
                vsx_image = utils.cv_bgr888_to_vsximage(
                    cv_image, image_format, args.device_id
                )
                result = model.process(vsx_image)
                index = np.argsort(result)[0][::-1]
                for i in range(5):
                    fout.write(
                        f"{file}: top-{i} id: {index[i]}, prob: {result[0, index[i]]}, class name: {labels[index[i]]}\n"
                    )
