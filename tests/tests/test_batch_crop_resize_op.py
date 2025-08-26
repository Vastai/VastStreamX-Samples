
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
def test_batch_crop_resize_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/batch_crop_resize \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_size [512,512] \
    --crop_rect1 [50,70,131,230] \
    --crop_rect2 [60,90,150,211] \
    --output_file1 batch_crop_resize_result1.jpg \
    --output_file2 batch_crop_resize_result2.jpg 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "batch_crop_resize_result1.jpg"
    ), "batch_crop_resize c++:can't find batch_crop_resize_result1.jpg"


    assert os.path.exists(
        "batch_crop_resize_result2.jpg"
    ), "batch_crop_resize c++:can't find batch_crop_resize_result2.jpg"

    os.system("rm batch_crop_resize_result1.jpg")
    os.system("rm batch_crop_resize_result2.jpg")



################# python test  #######################
@pytest.mark.fast
@pytest.mark.ai_integration
def test_batch_crop_resize_py(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/batch_crop_resize/batch_crop_resize.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_size [512,512] \
    --crop_rect1 [50,70,131,230] \
    --crop_rect2 [60,90,150,211] \
    --output_file1 batch_crop_resize_result1.jpg \
    --output_file2 batch_crop_resize_result2.jpg 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "batch_crop_resize_result1.jpg"
    ), "batch_crop_resize python:can't find batch_crop_resize_result1.jpg"


    assert os.path.exists(
        "batch_crop_resize_result2.jpg"
    ), "batch_crop_resize python:can't find batch_crop_resize_result2.jpg"

    os.system("rm batch_crop_resize_result1.jpg")
    os.system("rm batch_crop_resize_result2.jpg")

