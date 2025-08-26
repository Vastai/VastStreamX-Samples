
# 
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 
from run_cmd import run_cmd
import os
import pytest

################# c++ test  #######################
@pytest.mark.fast
@pytest.mark.ai_integration
def test_crop_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/crop \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file crop_result.jpg \
    --crop_rect [33,65,416,416]
    """
    run_cmd(cmd)

    assert os.path.exists("crop_result.jpg"), "crop c++:can't find crop_result.jpg"

    os.system("rm crop_result.jpg")


################# python test  #######################
@pytest.mark.fast
@pytest.mark.ai_integration
def test_crop_py(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/crop/crop.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file crop_result.jpg \
    --crop_rect [33,65,416,416]
    """

    run_cmd(cmd)

    assert os.path.exists("crop_result.jpg"), "crop python:can't find crop_result.jpg"

    os.system("rm crop_result.jpg")
