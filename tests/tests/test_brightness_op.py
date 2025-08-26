
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
################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_brightness_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/brightness  \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file brightness_result.jpg \
    --elf_file /opt/vastai/vaststreamx/data/elf/brightness \
    --scale 2.2
    """

    res = run_cmd(cmd)

    assert os.path.exists(
        "brightness_result.jpg"
    ), "brightness_op c++:can't find brightness_result.jpg"

    os.system("rm brightness_result.jpg")

#################  performance test
@pytest.mark.fast
def test_brightness_cpp_performance(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/brightness_prof \
    --elf_file /opt/vastai/vaststreamx/data/elf/brightness \
    --device_ids [{device_id}] \
    --shape "[3,640,640]" \
    --instance 8 \
    --iterations 50000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 9000
    assert (
        float(qps) > qps_thresh
    ), f"brightness_op c++:best prof qps {qps} is smaller than {qps_thresh}"


################# python test  #######################
################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_brightness_py(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/custom_op/brightness/brightness.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file brightness_result.jpg \
    --elf_file /opt/vastai/vaststreamx/data/elf/brightness \
    --scale 2.2
    """


    res = run_cmd(cmd)

    assert os.path.exists(
        "brightness_result.jpg"
    ), "brightness_op python:can't find brightness_result.jpg"

    os.system("rm brightness_result.jpg")

#################  performance test
@pytest.mark.fast
def test_brightness_py_performance(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/custom_op/brightness/brightness_prof.py \
    --elf_file /opt/vastai/vaststreamx/data/elf/brightness \
    --device_ids [{device_id}] \
    --shape "[3,640,640]" \
    --instance 8 \
    --iterations 50000 \
    --percentiles "[50,90,95,99]" \
    --input_host 0 
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 9000
    assert (
        float(qps) > qps_thresh
    ), f"brightness_op python:best prof qps {qps} is smaller than {qps_thresh}"

