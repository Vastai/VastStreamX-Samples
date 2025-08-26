
# 
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 
from run_cmd import run_cmd
import re
import os
import pytest

################# c++ test  #######################
@pytest.mark.fast
@pytest.mark.ai_integration
def test_copy_make_border_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/copy_make_border \
    --device_id {device_id} \
    --input_file ./data/images/cat.jpg \
    --output_file copy_make_border_result.jpg \
    --output_size "[640,640]"
    """
    run_cmd(cmd)

    assert os.path.exists(
        "copy_make_border_result.jpg"
    ), "copy_make_border c++:can't find copy_make_border_result.jpg"

    os.system("rm copy_make_border_result.jpg")


################# python test  #######################
@pytest.mark.fast
@pytest.mark.ai_integration
def test_copy_make_border_py(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/copy_make_border/copy_make_border.py \
    --device_id {device_id} \
    --input_file ./data/images/cat.jpg \
    --output_file copy_make_border_result.jpg \
    --output_size "[640,640]"
    """

    run_cmd(cmd)

    assert os.path.exists(
        "copy_make_border_result.jpg"
    ), "copy_make_border python:can't find copy_make_border_result.jpg"

    os.system("rm copy_make_border_result.jpg")
