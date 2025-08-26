import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

from common.text_rec import TextRecognizer
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
        default="/opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--vdsp_params",
        default="./data/configs/crnn_rgbplanar.json",
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
        default="../data/labels/key_37.txt",
        help="label file",
    )
    parser.add_argument(
        "--input_file",
        default="../data/images/word_336.png",
        help="input file",
    )
    parser.add_argument(
        "--dataset_filelist",
        default="",
        help="dataset filelist",
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


if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1
    model = TextRecognizer(
        args.model_prefix,
        args.vdsp_params,
        args.label_file,
        batch_size,
        args.device_id,
        args.hw_config,
    )
    image_format = model.get_fusion_op_iimage_format()

    if args.dataset_filelist == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Failed to read input file: {args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        res = model.process(vsx_image)
        print(res)
    else:
        filelist = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = f.readlines()
        with open(args.dataset_output_file, "wt") as outfile:
            for filename in filelist:
                fullname = os.path.join(args.dataset_root, filename.replace("\n", ""))
                print("fullname:", fullname)
                cv_image = cv2.imread(fullname)
                assert cv_image is not None, f"Read image failed:{filename}"
                vsx_image = utils.cv_bgr888_to_vsximage(
                    cv_image, image_format, args.device_id
                )
                result = model.process(vsx_image)
                basename, _ = os.path.splitext(os.path.basename(fullname))
                result_str = result[0][0]
                outfile.write(f"{basename} {result_str}\n")
        outfile.close()
