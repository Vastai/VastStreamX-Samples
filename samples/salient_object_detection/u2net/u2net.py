import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)
from common.u2net_detector import U2netDetector
import common.utils as utils
import numpy as np
import cv2
import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--vdsp_params",
        default="../../../../data/configs/u2net_bgr888.json",
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
        "--input_file",
        default="../../../../data/images/cat.jpg",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="./u2net_result.png",
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
    model = U2netDetector(
        args.model_prefix, args.vdsp_params, batch_size, args.device_id, args.hw_config
    )
    image_format = model.get_fusion_op_iimage_format()
    if args.dataset_filelist == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Failed to read input file: {args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        output = model.process(vsx_image)
        output = np.uint8(np.clip(np.floor(0.5 + output * 255), 0, 255))
        cv2.imwrite(args.output_file, output)
    else:
        filelist = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = [line.strip() for line in f.readlines()]
        for file in filelist:
            fullname = os.path.join(args.dataset_root, file)
            # print(f'{os.path.abspath(fullname)}')
            cv_image = cv2.imread(fullname)
            assert cv_image is not None, f"Failed to read input file: {fullname}"
            vsx_image = utils.cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )
            output = model.process(vsx_image)
            output = np.uint8(np.clip(np.floor(0.5 + output * 255), 0, 255))
            outfile = os.path.join(
                args.dataset_output_folder,
                os.path.basename(fullname).split(".")[0] + ".png",
            )
            cv2.imwrite(outfile, output)
