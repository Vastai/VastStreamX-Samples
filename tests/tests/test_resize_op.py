
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
def test_resize_op_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/resize \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_size "[512,512]"   \
    --output_file resize_result.jpg
    """
    run_cmd(cmd)

    assert os.path.exists("resize_result.jpg"), "resize c++:can't find resize_result.jpg"

    os.system("rm resize_result.jpg")


################# python test  #######################
@pytest.mark.fast
@pytest.mark.ai_integration
def test_resize_op_py(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/resize/resize.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_size "[600,800]"   \
    --output_file resize_result.jpg
    """
    run_cmd(cmd)

    assert os.path.exists("resize_result.jpg"), "resize python:can't find resize_result.jpg"

    os.system("rm resize_result.jpg")
