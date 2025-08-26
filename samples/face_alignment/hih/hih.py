import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)

import cv2
import argparse
from hih_aligner.hih_aligner import Hih
import common.utils as utils


def argument_parser():
    parser = argparse.ArgumentParser(description="FACE_ALIGNMENT")
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--vdsp_params",
        default="../data/configs/hih_bgr888.json",
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
        default="../../../../data/images/face.jpg",
        type=str,
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="./face_out.png",
        type=str,
        help="output file",
    )
    parser.add_argument(
        "--dataset_filelist",
        default="",
        type=str,
        help="input dataset image list",
    )
    parser.add_argument(
        "--dataset_root",
        default="",
        help="dataset root",
    )
    parser.add_argument(
        "--dataset_output_file",
        default="",
        type=str,
        help="dataset output file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1
    face_aligner = Hih(args.model_prefix, args.vdsp_params, batch_size, args.device_id)
    image_format = face_aligner.get_fusion_op_iimage_format()
    text = []
    if args.dataset_filelist == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Read image failed:{args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        landmarks = face_aligner.process(vsx_image)
        print("Face alignment results:")
        print(landmarks)
    else:
        filelist = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = [line.strip() for line in f.readlines()]
        for _, file in enumerate(filelist):
            filename = os.path.basename(file)
            fullname = os.path.join(args.dataset_root, filename)
            cv_image = cv2.imread(fullname)
            assert cv_image is not None, f"Read image failed:{fullname}"
            vsx_image = utils.cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )
            landmarks = face_aligner.process(vsx_image)
            text.append(filename)
            for i in landmarks.reshape((1, -1)).squeeze():
                text.append(f" {i}")
            text.append("\n")

    if args.dataset_output_file != "":
        with open(args.dataset_output_file, "w+") as f:
            f.writelines(text)
