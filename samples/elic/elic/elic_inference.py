#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
from typing import List

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)

import cv2
import argparse
from elic_compress import ElicCompress
from elic_decompress import ElicDecompress
import common.utils as utils
import torchvision
import torch
import math
import time
import numpy as np

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]


def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.nn.functional.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gaha_model_prefix",
        default="/opt/vastai/vaststreamx/data/models/elic-compress-g_a-h_a-fp16-1_3_512_512-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--gaha_hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--gaha_vdsp_params",
        default="./data/configs/elic_compress_gaha_rgb888.json",
        help="vdsp preprocess parameter file",
    )
    parser.add_argument(
        "--hs_model_prefix",
        default="/opt/vastai/vaststreamx/data/models/elic-compress-h_s_chunk-fp16-1_192_8_8-vacc/mod",
        help="h_s model prefix of the model suite files",
    )
    parser.add_argument(
        "--hs_hw_config",
        default="",
        help="hs_hw-config file of the model suite",
    )
    parser.add_argument(
        "--gs_model_prefix",
        default="/opt/vastai/vaststreamx/data/models/elic-compress-g_s_chunk-fp16-1_192_8_8-vacc/mod",
        help="g_s model prefix of the model suite files",
    )
    parser.add_argument(
        "--gs_hw_config",
        default="",
        help="gs_hw-config file of the model suite",
    )
    parser.add_argument(
        "--torch_model",
        default="",
        help="torch model file",
    )
    parser.add_argument(
        "--tensorize_elf_path",
        default="",
        help="tensorize elf file",
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
        default="../../../data/images/cycling.jpg",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="./elic_compress_result.jpg",
        help="output file",
    )
    parser.add_argument(
        "--dataset_path",
        default="",
        help="input dataset path",
    )
    parser.add_argument(
        "--dataset_output_path",
        default="",
        help="dataset output path",
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=256,
        help="padding patch size (default: %(default)s)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1

    compressor = ElicCompress(
        args.gaha_model_prefix,
        args.gaha_vdsp_params,
        args.hs_model_prefix,
        args.torch_model,
        batch_size,
        args.device_id,
        args.gaha_hw_config,
        args.hs_hw_config,
        args.patch,
    )
    image_format = compressor.get_fusion_op_iimage_format()

    decompressor = ElicDecompress(
        args.hs_model_prefix,
        args.gs_model_prefix,
        args.torch_model,
        args.tensorize_elf_path,
        batch_size,
        args.device_id,
        args.hs_hw_config,
        args.gs_hw_config,
    )
    if args.dataset_path == "":
        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Read image failed:{args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        p = args.patch
        h = vsx_image.height
        w = vsx_image.width
        new_h = (h + p - 1) // p * p
        new_w = (w + p - 1) // p * p
        padding_left = 0
        padding_right = new_w - w - padding_left
        padding_top = 0
        padding_bottom = new_h - h - padding_top
        com_out = compressor.process(vsx_image)
        decom_out = decompressor.decompress(com_out["strings"], com_out["shape"])

        decom_out["x_hat"] = torch.nn.functional.pad(
            decom_out["x_hat"],
            (-padding_left, -padding_right, -padding_top, -padding_bottom),
        )
        torchvision.utils.save_image(decom_out["x_hat"], args.output_file, nrow=1)
    else:
        filepaths = collect_images(args.dataset_path)
        filepaths = sorted(filepaths)
        if len(filepaths) == 0:
            print(
                f"Error: no images found in directory:{args.dataset_path}.",
                file=sys.stderr,
            )
            sys.exit(1)
        os.makedirs(args.dataset_output_path, exist_ok=True)
        compress_times = []
        decompress_times = []
        pnsrs = []

        for file in filepaths:
            print(f"image file:{file}")
            cv_image = cv2.imread(file)
            assert cv_image is not None, f"Read image failed:{file}"
            vsx_image = utils.cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )

            p = args.patch
            h = vsx_image.height
            w = vsx_image.width
            new_h = (h + p - 1) // p * p
            new_w = (w + p - 1) // p * p
            padding_left = 0
            padding_right = new_w - w - padding_left
            padding_top = 0
            padding_bottom = new_h - h - padding_top
            start = time.time()
            com_out = compressor.process(vsx_image)
            compress_times.append(time.time() - start)

            start = time.time()
            decom_out = decompressor.decompress(com_out["strings"], com_out["shape"])
            decompress_times.append(time.time() - start)
            decom_out["x_hat"] = torch.nn.functional.pad(
                decom_out["x_hat"],
                (-padding_left, -padding_right, -padding_top, -padding_bottom),
            )

            cv_image = np.array(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)).transpose(
                2, 0, 1
            )
            cv_image = cv_image[np.newaxis, :] / 255.0
            psnr = compute_psnr(decom_out["x_hat"], torch.from_numpy(cv_image))
            print(f"psnr:{psnr}")
            pnsrs.append(psnr)

            out_file = os.path.join(args.dataset_output_path, os.path.basename(file))
            torchvision.utils.save_image(decom_out["x_hat"], out_file, nrow=1)

        average_compress_time = sum(compress_times) / float(len(compress_times))
        average_decompress_time = sum(decompress_times) / float(len(decompress_times))
        average_pnsr = sum(pnsrs) / float(len(pnsrs))
        print(f"    Ave Compress time:{average_compress_time*1000} ms")
        print(f"    Ave Decompress time:{average_decompress_time*1000} ms")
        print(f"    Ave PNSR:{average_pnsr}")
