
# 
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 
from run_cmd import run_cmd
import re
import pytest

################# c++ test  #######################
@pytest.mark.fast
@pytest.mark.ai_integration
def test_argmax_op_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/argmax \
    --device_id {device_id} \
    --elf_file /opt/vastai/vaststreamx/data/elf/planar_argmax \
    --input_shape "[19,512,512]"
    """

    res = run_cmd(cmd)

    assert (
        re.findall(pattern="1,1,512,512", string=res) != []
    ), 'argmax c++:can\'t find "1,1,512,512" in result string'

@pytest.mark.fast
def test_argmax_op_cpp_performance(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/argmax_prof \
    --device_ids [{device_id}] \
    --elf_file /opt/vastai/vaststreamx/data/elf/planar_argmax \
    --shape "[19,512,512]" \
    --instance 4 \
    --iterations 10000 \
    --percentiles "[50,90,95,99]" \
    --input_host 0 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 2900
    assert (
        float(qps) > qps_thresh
    ), f"argmax c++: performance qps {qps} is smaller than {qps_thresh}"




################# Python test  #######################

@pytest.mark.fast
@pytest.mark.ai_integration
def test_argmax_op_py(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/custom_op/argmax/argmax.py \
    --device_id {device_id} \
    --elf_file /opt/vastai/vaststreamx/data/elf/planar_argmax \
    --shape "[19,512,512]"
    """

    res = run_cmd(cmd)

    assert (
        re.findall(pattern="1, 1, 512, 512", string=res) != []
    ), 'argmax python:can\'t find "1, 1, 512, 512" in result string'


@pytest.mark.fast
def test_argmax_op_py_performance(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/custom_op/argmax/argmax_prof.py \
    --elf_file /opt/vastai/vaststreamx/data/elf/planar_argmax \
    --device_ids [{device_id}] \
    --shape "[19,512,512]" \
    --instance 4 \
    --iterations 10000 \
    --percentiles "[50,90,95,99]" \
    --input_host 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 2900
    assert (
        float(qps) > qps_thresh
    ), f"argmax python: performance qps {qps} is smaller than {qps_thresh}"


