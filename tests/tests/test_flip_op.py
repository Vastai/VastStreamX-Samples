
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
def test_flip_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/flip \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file flip_result.jpg \
    --flip_type y 
    """
    run_cmd(cmd)

    assert os.path.exists("flip_result.jpg"), "flip c++:can't find flip_result.jpg"

    os.system("rm flip_result.jpg")


################# python test  #######################
@pytest.mark.fast
@pytest.mark.ai_integration
def test_flip_py(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/flip/flip.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file flip_result.jpg \
    --flip_type y 
    """

    run_cmd(cmd)

    assert os.path.exists("flip_result.jpg"), "flip python:can't find flip_result.jpg"

    os.system("rm flip_result.jpg")
