
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
def test_jpeg_encode_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/jpeg_encode \
    --device_id {device_id} \
    --height 354 \
    --width 474 \
    --input_file ./data/images/cat_354x474_nv12.yuv \
    --output_file ./jpeg_encode_result.jpg
    """

    run_cmd(cmd)

    assert os.path.exists(
        "jpeg_encode_result.jpg"
    ), "jpeg_encode c++:can't find jpeg_encode_result.jpg"

    os.system("rm jpeg_encode_result.jpg")


################# c++ performance test  #######################
@pytest.mark.codec
def test_jpeg_encode_cpp_performance(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/jpeg_encode_prof \
    --device_ids [{device_id}] \
    --percentiles "[50,90,95,99]" \
    --input_file ./data/images/plate_1920_1080.yuv \
    --width 1920 \
    --height 1080 \
    --instance 10 \
    --iterations 3000
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 900
    assert (
        float(qps) > qps_thresh
    ), f"jpeg_encode c++:best performance qps {qps} is smaller than {qps_thresh}"

    cmd = f"""
    ./build/vaststreamx-samples/bin/jpeg_encode_prof \
    --device_ids [{device_id}] \
    --percentiles "[50,90,95,99]" \
    --input_file ./data/images/plate_1920_1080.yuv \
    --width 1920 \
    --height 1080 \
    --instance 1 \
    --iterations 500
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 155
    assert (
        float(qps) > qps_thresh
    ), f"jpeg_encode c++:best latancy qps {qps} is smaller than {qps_thresh}"



################# python test  #######################
@pytest.mark.codec
@pytest.mark.codec_integration
def test_jpeg_encode_py(device_id):
    cmd = f"""
    python3 ./samples/jpeg_encode/jpeg_encode.py \
    --device_id {device_id} \
    --input_file ./data/images/cat_354x474_nv12.yuv \
    --output_file ./jpeg_encode_result.jpg
    """

    run_cmd(cmd,False)

    assert os.path.exists(
        "jpeg_encode_result.jpg"
    ), "jpeg_encode python:can't find jpeg_encode_result.jpg"

    os.system("rm jpeg_encode_result.jpg")

################# python  performance test  #######################
@pytest.mark.codec
def test_jpeg_encode_py_performance(device_id):
    cmd = f"""
    python3 ./samples/jpeg_encode/jpeg_encode_prof.py \
    --device_ids [{device_id}] \
    --percentiles "[50,90,95,99]" \
    --input_file ./data/images/plate_1920_1080.yuv \
    --width 1920 \
    --height 1080 \
    --instance 10 \
    --iterations 3000
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 900
    assert (
        float(qps) > qps_thresh
    ), f"jpeg_encode python:best performance qps {qps} is smaller than {qps_thresh}"

    cmd = f"""
    python3 ./samples/jpeg_encode/jpeg_encode_prof.py \
    --device_ids [{device_id}] \
    --percentiles "[50,90,95,99]" \
    --input_file ./data/images/plate_1920_1080.yuv \
    --width 1920 \
    --height 1080 \
    --instance 1 \
    --iterations 500
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 155
    assert (
        float(qps) > qps_thresh
    ), f"jpeg_encode python:best latancy qps {qps} is smaller than {qps_thresh}"


