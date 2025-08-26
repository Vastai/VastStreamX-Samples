import os
import sys
import cv2
import argparse
import pickle
import torch
import utils

from torch import nn

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../")
sys.path.append(common_path)

from common.dinov2_model import Dinov2Model
from common.utils import load_labels

import numpy as np
import cv2


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vaststreamx/data/models/dinov2-b-fp16-none-1_3_224_224-vacc/mod",
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
        default="../../../data/images/oxford_003681.jpg",
        help="input file",
    )

    parser.add_argument(
        "--dataset_root",
        default="",
        help="input dataset root",
    )
    parser.add_argument(
        "--dataset_conf",
        default="./gnd_roxford5k.pkl",
        help="dataset conf pkl file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1

    dinov2 = Dinov2Model(
        args.model_prefix,
        args.norm_elf_file,
        args.space_to_depth_elf_file,
        batch_size=batch_size,
        device_id=args.device_id,
    )
    print(dinov2.output_shape)
    if args.dataset_root == "":
        image = cv2.imread(args.input_file)
        assert image is not None, f"Failed to read input file: {args.input_file}"
        outputs = dinov2.process(image)
        for output in outputs:
            print(f"output:{output} ")
    else:
        train_list = []
        query_list = []
        with open(args.dataset_conf, "rb") as f:
            cfg = pickle.load(f)
        query_list = cfg["qimlist"]
        train_list = cfg["imlist"]

        train_features = []
        for file in train_list:
            fullname = os.path.join(args.dataset_root, file + ".jpg")
            print(fullname)
            image = cv2.imread(fullname)
            assert image is not None, f"Failed to read input file: {fullname}"
            outputs = dinov2.process(image)
            train_features.append(torch.from_numpy(outputs[0]))
        # normalize features
        train_features = torch.stack(train_features, dim=1)
        train_features = torch.squeeze(train_features)
        train_features = nn.functional.normalize(
            train_features.to(torch.float32), dim=1, p=2
        )

        query_features = []
        for file in query_list:
            fullname = os.path.join(args.dataset_root, file + ".jpg")
            print(fullname)
            image = cv2.imread(fullname)
            assert image is not None, f"Failed to read input file: {fullname}"
            outputs = dinov2.process(image)
            query_features.append(torch.from_numpy(outputs[0]))

        query_features = torch.stack(query_features, dim=1)
        query_features = torch.squeeze(query_features)
        query_features = nn.functional.normalize(
            query_features.to(torch.float32), dim=1, p=2
        )

        # Step 2: similarity
        sim = torch.mm(train_features.T, query_features)
        ranks = torch.argsort(-sim, dim=0).cpu().numpy()

        # Step 3: evaluate
        gnd = cfg["gnd"]
        # evaluate ranks
        ks = [1, 5, 10]
        # search for easy & hard
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g["ok"] = np.concatenate([gnd[i]["easy"], gnd[i]["hard"]])
            g["junk"] = np.concatenate([gnd[i]["junk"]])
            gnd_t.append(g)

        mapM, apsM, mprM, prsM = utils.compute_map(ranks, gnd_t, ks)

        # search for hard
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g["ok"] = np.concatenate([gnd[i]["hard"]])
            g["junk"] = np.concatenate([gnd[i]["junk"], gnd[i]["easy"]])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = utils.compute_map(ranks, gnd_t, ks)
        print(
            "mAP M: {}, H: {}".format(
                np.around(mapM * 100, decimals=2),
                np.around(mapH * 100, decimals=2),
            )
        )
        print(
            "mP@k{} M: {}, H: {}".format(
                np.array(ks),
                np.around(mprM * 100, decimals=2),
                np.around(mprH * 100, decimals=2),
            )
        )
