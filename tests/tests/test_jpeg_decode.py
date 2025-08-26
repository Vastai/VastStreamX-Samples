
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
@pytest.mark.codec
@pytest.mark.codec_integration
def test_jpeg_decode_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/jpeg_decode \
    --device_id {device_id} \
    --input_file ./data/images/cat.jpg \
    --output_file ./jpeg_decode_result.yuv
    """

    run_cmd(cmd)

    assert os.path.exists(
        "jpeg_decode_result.yuv"
    ), "jpeg_decode c++:can't find jpeg_decode_result.yuv"

    os.system("rm jpeg_decode_result.yuv   jpeg_decode_result.bmp")


################# python test  #######################
@pytest.mark.codec
@pytest.mark.codec_integration
def test_jpeg_decode_py(device_id):
    cmd = f"""
    python3 ./samples/jpeg_decode/jpeg_decode.py \
    --device_id {device_id} \
    --input_file ./data/images/cat.jpg \
    --output_file ./jpeg_decode_result.yuv
    """

    run_cmd(cmd)

    assert os.path.exists(
        "jpeg_decode_result.yuv"
    ), "jpeg_decode python:can't find jpeg_decode_result.yuv"

    os.system("rm jpeg_decode_result.yuv  jpeg_decode_result.bmp")
