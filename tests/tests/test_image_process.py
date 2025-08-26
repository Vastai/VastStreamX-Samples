
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
def test_image_process_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/image_process \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file ./image_process_result.jpg
    """

    run_cmd(cmd)

    assert os.path.exists(
        "image_process_result.jpg"
    ), "image_process c++:can't find image_process_result.jpg"

    os.system("rm image_process_result.jpg")


################# python test  #######################
@pytest.mark.fast
@pytest.mark.ai_integration
def test_image_process_py(device_id):
    cmd = f"""
    python3 ./samples/image_process/image_process.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file ./image_process_result.jpg
    """

    run_cmd(cmd)

    assert os.path.exists(
        "image_process_result.jpg"
    ), "image_process python:can't find image_process_result.jpg"

    os.system("rm image_process_result.jpg")
