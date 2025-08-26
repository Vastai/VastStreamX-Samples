import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

import numpy as np
import cv2
import argparse
from dbnet_detector.dbnet_detector import DbnetDetector
import vaststreamx as vsx
import common.utils as utils


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--vdsp_params",
        default="./data/configs/dbnet_rgbplanar.json",
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
        "--elf_file",
        default="/opt/vastai/vaststreamx/data/elf/find_contours_ext_op",
        help="input file",
    )
    parser.add_argument(
        "--input_file",
        default="../../../data/images/detect.jpg",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="./dbnet_result.jpg",
        help="output file",
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
    detector = DbnetDetector(
        args.model_prefix,
        args.vdsp_params,
        batch_size,
        args.device_id,
        args.hw_config,
        elf_file=args.elf_file,
    )
    input_shape = detector.input_shape[0]
    image_format = detector.get_fusion_op_iimage_format()

    if args.dataset_filelist == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Read image failed:{args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        [bboxes, scores] = detector.process(vsx_image)
        for i in range(len(bboxes)):
            print(
                f"index:{i}, score:{scores[i]},bbox:[{bboxes[i][0]},{bboxes[i][1]},{bboxes[i][2]},{bboxes[i][3]}]"
            )
        if args.output_file != "":
            for bbox in bboxes:
                for i in range(len(bbox)):
                    t = (i + 1) % len(bbox)
                    pt1 = (bbox[i][0], bbox[i][1])
                    pt2 = (bbox[t][0], bbox[t][1])
                    cv2.line(cv_image, pt1, pt2, color=(0, 0, 255))
            cv2.imwrite(args.output_file, cv_image)
    else:
        filelist = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = f.readlines()
        for filename in filelist:
            filename = os.path.join(args.dataset_root, filename.replace("\n", ""))
            print(filename)
            cv_image = cv2.imread(filename)
            assert cv_image is not None, f"Read image failed:{filename}"
            vsx_image = utils.cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )
            [bboxes, scores] = detector.process(vsx_image)
            basename, _ = os.path.splitext(os.path.basename(filename))
            npz_file = os.path.join(args.dataset_output_folder, basename + ".npz")
            np.savez(npz_file, output_0=bboxes)
