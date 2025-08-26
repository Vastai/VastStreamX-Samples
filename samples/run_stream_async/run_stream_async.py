#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
from threading import Thread
import queue

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

from common.classifier_async import ClassifierAsync
import common.utils as utils
import numpy as np
import cv2
import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--vdsp_params",
        default="./data/configs/resnet_bgr888.json",
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
        default="data/labels/imagenet.txt",
        help="label file",
    )
    parser.add_argument(
        "--input_file",
        default="data/images/cat.jpg",
        help="input file",
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    labels = utils.load_labels(args.label_file)
    batch_size = 1
    classifier = ClassifierAsync(
        args.model_prefix, args.vdsp_params, batch_size, args.device_id
    )
    image_format = classifier.get_fusion_op_iimage_format()

    if args.dataset_filelist == "":

        def get_output_thread():
            while True:
                try:
                    output = classifier.get_output()[0]
                    index = np.argsort(output)[0][::-1]
                    print("Top5:")
                    for i in range(5):
                        print(
                            f"{i}th, class name: {labels[index[i]]}, score: {output[0, index[i]]}"
                        )
                except ValueError:
                    break

        thread = Thread(target=get_output_thread)
        thread.start()

        cv_image = cv2.imread(args.input_file)
        assert cv_image is not None, f"Failed to read input file: {args.input_file}"
        vsx_image = utils.cv_bgr888_to_vsximage(cv_image, image_format, args.device_id)
        classifier.process_async(vsx_image)
        classifier.close_input()
        thread.join()
        classifier.wait_until_done()
    else:
        filelist = []
        with open(args.dataset_filelist, "rt") as f:
            filelist = [line.strip() for line in f.readlines()]

        def get_output_thread(que):
            with open(args.dataset_output_file, "wt") as fout:
                while True:
                    try:
                        output = classifier.get_output()[0]
                        file = que.get()
                        index = np.argsort(output)[0][::-1]
                        for i in range(5):
                            fout.write(
                                f"{file}: top-{i} id: {index[i]}, prob: {output[0, index[i]]}, class name: {labels[index[i]]}\n"
                            )
                    except ValueError:
                        break

        que = queue.Queue()
        thread = Thread(target=get_output_thread, args=(que,))
        thread.start()

        for file in filelist:
            fullname = os.path.join(args.dataset_root, file)
            print(fullname)
            cv_image = cv2.imread(fullname)
            assert cv_image is not None, f"Failed to read input file: {fullname}"
            vsx_image = utils.cv_bgr888_to_vsximage(
                cv_image, image_format, args.device_id
            )

            que.put(file)
            result = classifier.process_async(vsx_image)
        classifier.close_input()
        thread.join()
        classifier.wait_until_done()
