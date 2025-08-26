import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)

from common.model_cv import ModelCV
import numpy as np
import cv2
import argparse
import common.utils as utils


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--vdsp_params",
        default="./data/configs/bisenet_bgr888.json ",
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
        default="data/images/face.jpg",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="./bisenet_result.jpg",
        help="output file",
    )
    parser.add_argument(
        "--dataset_filelist",
        default="",
        help="dataset filelist",
    )
    parser.add_argument(
        "--dataset_root",
        default="",
        help="dataset root",
    )
    parser.add_argument(
        "--dataset_output_folder",
        default="",
        help="dataset output folder",
    )
    args = parser.parse_args()
    return args


part_colors = [
    [0, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 0, 85],
    [255, 0, 170],
    [0, 255, 0],
    [85, 255, 0],
    [170, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [0, 85, 255],
    [0, 170, 255],
    [255, 255, 0],
    [255, 255, 85],
    [255, 255, 170],
    [255, 0, 255],
    [255, 85, 255],
    [255, 170, 255],
    [0, 255, 255],
    [85, 255, 255],
    [170, 255, 255],
]


if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1
    model = ModelCV(
        args.model_prefix, args.vdsp_params, batch_size, args.device_id, args.hw_config
    )
    image_format = model.get_fusion_op_iimage_format()

    if args.dataset_filelist == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Read image failed: {args.input_image}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)

        result = model.process(vsx_image)
        class_map = np.array(result).squeeze().argmax(0)
        num_of_class = np.max(class_map)
        class_map = cv2.resize(
            class_map,
            (cv_image.shape[1], cv_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        segmentation_image = np.zeros(cv_image.shape, dtype=np.uint8)
        for pi in range(1, num_of_class + 1):
            index = np.where(class_map == pi)
            segmentation_image[index[0], index[1], :] = part_colors[pi]
        cv2.imwrite(args.output_file, segmentation_image)
        print(f"Write result to: {args.output_file}")
    else:
        filelist = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = f.readlines()
        for image_file in filelist:
            image_file = os.path.join(
                args.dataset_root, image_file.strip(" ").strip("\n")
            )
            print(image_file)
            cv_image = cv2.imread(image_file)
            assert cv_image is not None, f"Read image failed: {image_file}"
            vsx_image = utils.cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )

            result = model.process(vsx_image)

            base_name = os.path.basename(image_file).split(".")[0]
            npz_file = os.path.join(args.dataset_output_folder, base_name + ".npz")
            np.savez(npz_file, output_0=result[0])
