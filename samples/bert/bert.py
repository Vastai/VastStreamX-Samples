import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

from common.bert_base import Bert
import numpy as np
import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_prefix",
        default="/opt/vastai/vastpipe/data/models/bert_base_en_qa-384-int8-max-1_384_1_384_1_384-vacc/mod",
        help="model prefix of the model suite files",
    )
    parser.add_argument(
        "--hw_config",
        default="",
        help="hw-config file of the model suite",
    )
    parser.add_argument(
        "--vdsp_params",
        default="./data/configs/bert_vdsp.json",
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
        default="",
        help="input file",
    )
    parser.add_argument(
        "--output_file",
        default="",
        help="output file",
    )
    parser.add_argument(
        "--dataset_filelist",
        default="",
        help="input_npz filename list",
    )
    parser.add_argument(
        "--dataset_root",
        default="",
        help="dataset root",
    )
    parser.add_argument(
        "--dataset_output_folder",
        default="",
        help="dataset output folder save result to",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    batch_size = 1
    model = Bert(
        args.model_prefix,
        args.vdsp_params,
        batch_size,
        args.device_id,
        args.hw_config,
    )
    if args.dataset_filelist != "":
        npz_datalist_fr = open(args.dataset_filelist, "rt")
        npz_datalist = npz_datalist_fr.readlines()
        os.makedirs(args.dataset_output_folder, exist_ok=True)
        for i, npz_file in enumerate(npz_datalist):
            data = []
            npz_file = os.path.join(args.dataset_root, npz_file.strip())
            print(npz_file)
            npz_data = np.load(npz_file)
            for k, v in npz_data.items():
                data.append(v)
            result = model.process(data)
            out = {}
            out["output_0"] = result[0]
            np.savez(os.path.join(args.dataset_output_folder, str(i).zfill(6)), **out)
    else:
        data = []
        npz_data = np.load(args.input_file)
        for k, v in npz_data.items():
            data.append(v)
        result = model.process(data)
        out = {}
        out["output_0"] = result[0]
        np.savez(args.output_file, **out)
        print(f"save result to: {args.output_file}")
