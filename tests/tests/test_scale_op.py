
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
def test_scale_op_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/scale \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_size1 [512,512] \
    --output_size2 [600,800] \
    --output_file1 scale_result1.jpg \
    --output_file2 scale_result2.jpg 
    """
    run_cmd(cmd)

    assert os.path.exists("scale_result1.jpg"), "scale c++:can't find scale_result1.jpg"
    assert os.path.exists("scale_result2.jpg"), "scale c++:can't find scale_result2.jpg"

    os.system("rm scale_result1.jpg")
    os.system("rm scale_result2.jpg")



################# python test  #######################
@pytest.mark.fast
@pytest.mark.ai_integration
def test_scale_op_py(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/scale/scale.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_size1 [512,512] \
    --output_size2 [600,800] \
    --output_file1 scale_result1.jpg \
    --output_file2 scale_result2.jpg 
    """
    run_cmd(cmd)

    assert os.path.exists("scale_result1.jpg"), "scale python:can't find scale_result1.jpg"
    assert os.path.exists("scale_result2.jpg"), "scale python:can't find scale_result2.jpg"

    os.system("rm scale_result1.jpg")
    os.system("rm scale_result2.jpg")


