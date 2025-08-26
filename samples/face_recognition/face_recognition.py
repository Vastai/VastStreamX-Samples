import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

from common.classifier import Classifier
import common.utils as utils
import numpy as np
import cv2
import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--vdsp_params",
        default="../data/configs/facenet_bgr888.json",
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
        default="../../data/images/face.jpg",
        help="input file",
    )
    parser.add_argument(
        "--dataset_filelist",
        default="",
        help="input dataset image list",
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
    facenet = Classifier(
        args.model_prefix, args.vdsp_params, batch_size, args.device_id
    )
    image_format = facenet.get_fusion_op_iimage_format()

    if args.dataset_filelist == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Failed to read input file: {args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        result = facenet.process(vsx_image)
        print("Face feature:")
        print(result)
    else:
        filelist = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = f.readlines()
        for i, filename in enumerate(filelist):
            filename = os.path.join(args.dataset_root, filename.replace("\n", ""))
            print(filename)
            cv_image = cv2.imread(filename)
            assert cv_image is not None, f"Read image failed:{filename}"
            vsx_image = utils.cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )
            result = facenet.process(vsx_image)
            npz_file = os.path.join(
                args.dataset_output_folder, "output_" + str(i).zfill(6) + ".npz"
            )
            np.savez(npz_file, output_0=result)
