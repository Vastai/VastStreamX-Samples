import os
import sys
import cv2
import argparse

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)

from common.vit_model import VitModel
from common.utils import load_labels

import numpy as np
import cv2


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/vit-b-fp16-none-1_3_224_224-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--norm_elf_file",
        default="/opt/vastai/vaststreamx/data/elf/normalize",
        help="normalize op elf file",
    )
    parser.add_argument(
        "--space_to_depth_elf_file",
        default="/opt/vastai/vaststreamx/data/elf/space_to_depth",
        help="space_to_depth op elf files",
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


if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1
    labels = load_labels(args.label_file)

    vit = VitModel(
        args.model_prefix,
        args.norm_elf_file,
        args.space_to_depth_elf_file,
        batch_size=batch_size,
        device_id=args.device_id,
    )

    if args.dataset_filelist == "":
        image = cv2.imread(args.input_file)
        assert image is not None, f"Failed to read input file: {args.input_file}"
        output = vit.process(image)
        index = np.argsort(output)[0][::-1]
        print("Top5:")
        for i in range(5):
            print(
                f"{i}th: score: {output[0, index[i]]:0.4f}, class name: {labels[index[i]]}"
            )
    else:
        with open(args.dataset_filelist, "rt") as f:
            filelist = [line.strip() for line in f.readlines()]
        with open(args.dataset_output_file, "wt") as fout:
            for file in filelist:
                fullname = os.path.join(args.dataset_root, file)
                print(fullname)
                image = cv2.imread(fullname)
                assert image is not None, f"Failed to read input file: {fullname}"
                output = vit.process(image)
                index = np.argsort(output)[0][::-1]
                for i in range(5):
                    fout.write(
                        f"{file}: top-{i} id: {index[i]}, prob: {output[0, index[i]]}, class name: {labels[index[i]]}\n"
                    )
