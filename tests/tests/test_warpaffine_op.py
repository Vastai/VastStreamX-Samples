
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
def test_warpaffine_op_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/warpaffine \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --matrix [0.7890625,-0.611328125,56.0,0.611328125,0.7890625,-416.0] \
    --output_size [640,640] \
    --output_file warpaffine_reusult.jpg
    """

    run_cmd(cmd)

    assert os.path.exists(
        "warpaffine_reusult.jpg"
    ), "warpaffine c++:can't find warpaffine_reusult.jpg"

    os.system("rm warpaffine_reusult.jpg")


################# python test  #######################
@pytest.mark.fast
@pytest.mark.ai_integration
def test_warpaffine_op_py(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/warpaffine/warpaffine.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --matrix [0.7890625,-0.611328125,56.0,0.611328125,0.7890625,-416.0] \
    --output_size [640,640] \
    --output_file warpaffine_reusult.jpg
    """
    run_cmd(cmd)

    assert os.path.exists(
        "warpaffine_reusult.jpg"
    ), "warpaffine python:can't find warpaffine_reusult.jpg"

    os.system("rm warpaffine_reusult.jpg")
