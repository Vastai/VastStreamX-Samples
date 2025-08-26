import clip
import argparse
import numpy as np
import copy
import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../..")
sys.path.append(common_path)

from common.utils import load_labels


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_file",
        default="",
        help="label file",
    )
    parser.add_argument(
        "--npz_files_path",
        default="",
        help="path to save npz file",
    )
    args = parser.parse_args()
    return args


def make_tokens(text):
    assert isinstance(text, str), f"input type must be str"
    token = clip.tokenize(text)[0]
    token_padding = np.pad(token.numpy(), pad_width=(0, 3)).astype(np.int32)
    # make mask
    index = np.argmax(token_padding)
    token_mask = copy.deepcopy(token_padding)
    token_mask[: index + 1] = 1
    # make input
    zero_arr = np.zeros(token_padding.shape, dtype=np.int32)
    tokens = []
    tokens.append(token_padding)
    tokens.append(zero_arr)
    tokens.append(zero_arr)
    tokens.append(token_mask)
    tokens.append(zero_arr)
    tokens.append(zero_arr)

    return tokens


if __name__ == "__main__":
    args = argument_parser()
    labels = load_labels(args.label_file)

    if not os.path.exists(args.npz_files_path):
        os.makedirs(args.npz_files_path)

    for i, txt in enumerate(labels):
        tokens = make_tokens(txt)
        out = {}
        out["input_0"] = tokens[0]
        out["input_1"] = tokens[1]
        out["input_2"] = tokens[2]
        out["input_3"] = tokens[3]
        out["input_4"] = tokens[4]
        out["input_5"] = tokens[5]

        npz_file = os.path.join(args.npz_files_path, txt + ".npz")
        print(f"npz_file:{npz_file}")
        np.savez(npz_file, **out)
