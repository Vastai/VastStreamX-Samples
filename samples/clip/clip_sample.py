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

from common.utils import load_labels
from common.clip_model import ClipModel

import vaststreamx as vsx
import cv2
import argparse
import numpy as np


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgmod_prefix",
        default="/opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod",
        help="image model prefix of the model suite files",
    )
    parser.add_argument(
        "--imgmod_hw_config",
        help="image model hw-config file of the model suite",
        default="",
    )
    parser.add_argument(
        "--norm_elf",
        default="/opt/vastai/vaststreamx/data/elf/normalize",
        help="image model elf file",
    )
    parser.add_argument(
        "--space2depth_elf",
        default="/opt/vastai/vaststreamx/data/elf/space_to_depth",
        help="image model elf file",
    )
    parser.add_argument(
        "--txtmod_prefix",
        default="/opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod",
        help="text model prefix of the model suite files",
    )
    parser.add_argument(
        "--txtmod_hw_config",
        help="text model hw-config file of the model suite",
        default="",
    )
    parser.add_argument(
        "--txtmod_vdsp_params",
        default="./data/configs/clip_txt_vdsp.json",
        help="text model vdsp preprocess parameter file",
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
        default="data/images/CLIP.png",
        help="input file",
    )
    parser.add_argument(
        "--label_file",
        default="data/labels/imagenet.txt",
        help="label file",
    )
    parser.add_argument(
        "--dataset_filelist",
        default="",
        help="input dataset filelist",
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
    parser.add_argument(
        "--strings",
        default="[a diagram,a dog,a cat]",
        help='test strings, split by ","',
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1
    assert vsx.set_device(args.device_id) == 0
    model = ClipModel(
        args.imgmod_prefix,
        args.norm_elf,
        args.space2depth_elf,
        args.txtmod_prefix,
        args.txtmod_vdsp_params,
        batch_size,
        args.device_id,
    )

    if args.dataset_filelist == "":
        image = cv2.imread(args.input_file)
        assert image is not None, f"Failed to read input file: {args.input_file}"
        texts = args.strings.strip("[").strip("]").split(",")
        print(f"intput texts:{texts}")

        result = model.process(image=image, texts=texts)
        index = np.argsort(result)[::-1]
        n = 5 if len(index) >= 5 else len(index)
        print(f"Top{n}:")
        for i in range(n):
            print(f"{i}th, string: {texts[index[i]]}, score: {result[index[i]]}")
    else:
        labels = load_labels(args.label_file)
        texts_features = model.process_texts(labels)
        filelist = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = [line.strip() for line in f.readlines()]
        with open(args.dataset_output_file, "wt") as fout:
            for file in filelist:
                fullname = os.path.join(args.dataset_root, file)
                print(fullname)
                image = cv2.imread(fullname)
                assert image is not None, f"Failed to read input file: {fullname}"
                image_feature = model.process_image(image)
                result = model.post_process(image_feature, texts_features)
                index = np.argsort(result)[::-1]
                for i in range(5):
                    fout.write(
                        f"{file}: top-{i} id: {index[i]}, prob: {result[index[i]]}, class name: {labels[index[i]]}\n"
                    )
