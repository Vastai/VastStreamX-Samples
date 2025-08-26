#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import pickle


def read_pickle_file(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data
