
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
def test_dinov2_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/dinov2 \
    -m /opt/vastai/vaststreamx/data/models/dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc/mod \
    --norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
    --space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
    --device_id {device_id} \
    --input_file ./data/images/oxford_003681.jpg
    """

    res = run_cmd(cmd)

    feature_str = res.split("output:")[-1]
    feature = re.findall(pattern=r"-?\d+.\d+", string=feature_str)

    assert len(feature) == 1024, f"dinov2 c++: output len {len(feature)} is not 1024"



#################  performance test

@pytest.mark.fast
def test_dinov2_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/dinov2_prof \
    -m /opt/vastai/vaststreamx/data/models/dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc/mod \
    --norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
    --space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 200 \
    --shape [3,224,224] \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 25
    assert (
        float(qps) > qps_thresh
    ), f"dinov2 c++:best prof qps {qps} is smaller than {qps_thresh}"


    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/dinov2_prof \
    -m /opt/vastai/vaststreamx/data/models/dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc/mod \
    --norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
    --space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 200 \
    --shape [3,224,224] \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 24
    assert (
        float(qps) > qps_thresh
    ), f"dinov2 c++:best latancy qps {qps} is smaller than {qps_thresh}"


############################# dataset test

@pytest.mark.fast
def test_dinov2_cpp_precision(device_id):
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_path = os.path.abspath("./samples/dinov2/")
    os.environ['PYTHONPATH'] = f"{current_pythonpath}:{new_path}" 

    cmd = f"""
    ./build/vaststreamx-samples/bin/dinov2 \
    -m /opt/vastai/vaststreamx/data/models/dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc/mod \
    --norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
    --space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
    --device_id {device_id} \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/oxbuild_images-v1 \
    --dataset_conf ./data/labels/gnd_roxford5k.pkl
    """
    res = run_cmd(cmd)

    mAP_m = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"M: \d+.\d+", string=res).group(),
    ).group()
    mAP_h = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"H: \d+.\d+", string=res).group(),
    ).group()

    mpk_m = re.findall(
        pattern=r"\d+.\d+", string=re.findall(pattern=r"M: \[([^\]]*)\]", string=res)[0]
    )

    mpk_h = re.findall(
        pattern=r"\d+.\d+", string=re.findall(pattern=r"H: \[([^\]]*)\]", string=res)[0]
    )
    mAP_m_thresh = 79.6
    mAP_h_thresh = 58.1

    assert (
        float(mAP_m) > mAP_m_thresh
    ), f"dinov2 c++: mAP_m:{mAP_m} is smaller than {mAP_m_thresh}"
    assert (
        float(mAP_h) > mAP_h_thresh
    ), f"dinov2 c++: mAP_h:{mAP_h} is smaller than {mAP_h_thresh}"

    mpk_m_thresh_0 = 98.5
    mpk_m_thresh_1 = 94.5
    mpk_m_thresh_2 = 91.3
    assert (
        float(mpk_m[0]) > mpk_m_thresh_0
    ), f"dinov2 c++: mpk_m[0]:{mpk_m[0]} is smaller than {mpk_m_thresh_0}"
    assert (
        float(mpk_m[1]) > mpk_m_thresh_1
    ), f"dinov2 c++: mpk_m[1]:{mpk_m[1]} is smaller than {mpk_m_thresh_1}"
    assert (
        float(mpk_m[2]) > mpk_m_thresh_2
    ), f"dinov2 c++: mpk_m[2]:{mpk_m[2]} is smaller than {mpk_m_thresh_2}"


    mpk_h_thresh_0 = 92.8
    mpk_h_thresh_1 = 80.0
    mpk_h_thresh_2 = 70.0
    assert (
        float(mpk_h[0]) > mpk_h_thresh_0
    ), f"dinov2 c++: mpk_h[0]:{mpk_h[0]} is smaller than {mpk_h_thresh_0}"
    assert (
        float(mpk_h[1]) > mpk_h_thresh_1
    ), f"dinov2 c++: mpk_h[1]:{mpk_h[1]} is smaller than {mpk_h_thresh_1}"
    assert (
        float(mpk_h[2]) > mpk_h_thresh_2
    ), f"dinov2 c++: mpk_h[2]:{mpk_h[2]} is smaller than {mpk_h_thresh_2}"



#####################  python test #####################


##############################  one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_dinov2_py(device_id):
    cmd = f"""
    python3 ./samples/dinov2/dinov2.py \
    -m /opt/vastai/vaststreamx/data/models/dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc/mod \
    --norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
    --space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
    --device_id {device_id} \
    --input_file ./data/images/oxford_003681.jpg
    """
    res = run_cmd(cmd)

    feature = re.findall(
        pattern=r"-?\d+.\d+",
        string=re.search(pattern=r"output:\[([^\]]*)\]", string=res).group(),
    )

    assert len(feature) == 6, f"dinov2 python: output len {len(feature)} is not 6"



#################  performance test
@pytest.mark.fast
def test_dinov2_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/dinov2/dinov2_prof.py \
    -m /opt/vastai/vaststreamx/data/models/dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc/mod \
    --norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
    --space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 200 \
    --shape [3,224,224] \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 25
    assert (
        float(qps) > qps_thresh
    ), f"dinov2 python:best prof qps {qps} is smaller than {qps_thresh}"


    # 测试最小时延
    cmd = f"""
    python3 ./samples/dinov2/dinov2_prof.py \
    -m /opt/vastai/vaststreamx/data/models/dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc/mod \
    --norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
    --space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 200 \
    --shape [3,224,224] \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 24
    assert (
        float(qps) > qps_thresh
    ), f"dinov2 python:best latancy qps {qps} is smaller than {qps_thresh}"


############################# dataset test
@pytest.mark.fast
def test_dinov2_py_precision(device_id):
    cmd = f"""
    python3 ./samples/dinov2/dinov2.py \
    -m /opt/vastai/vaststreamx/data/models/dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc/mod \
    --norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
    --space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
    --device_id {device_id} \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/oxbuild_images-v1 \
    --dataset_conf ./data/labels/gnd_roxford5k.pkl
    """

    res = run_cmd(cmd)

    mAP_m = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"M: \d+.\d+", string=res).group(),
    ).group()
    mAP_h = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"H: \d+.\d+", string=res).group(),
    ).group()

    mpk_m = re.findall(
        pattern=r"\d+.\d+", string=re.findall(pattern=r"M: \[([^\]]*)\]", string=res)[0]
    )

    mpk_h = re.findall(
        pattern=r"\d+.\d+", string=re.findall(pattern=r"H: \[([^\]]*)\]", string=res)[0]
    )
    mAP_m_thresh = 79.5
    mAP_h_thresh = 58.1

    assert (
        float(mAP_m) > mAP_m_thresh
    ), f"dinov2 c++: mAP_m:{mAP_m} is smaller than {mAP_m_thresh}"
    assert (
        float(mAP_h) > mAP_h_thresh
    ), f"dinov2 c++: mAP_h:{mAP_h} is smaller than {mAP_h_thresh}"

    mpk_m_thresh_0 = 98.5
    mpk_m_thresh_1 = 94.5
    mpk_m_thresh_2 = 91.3
    assert (
        float(mpk_m[0]) > mpk_m_thresh_0
    ), f"dinov2 c++: mpk_m[0]:{mpk_m[0]} is smaller than {mpk_m_thresh_0}"
    assert (
        float(mpk_m[1]) > mpk_m_thresh_1
    ), f"dinov2 c++: mpk_m[1]:{mpk_m[1]} is smaller than {mpk_m_thresh_1}"
    assert (
        float(mpk_m[2]) > mpk_m_thresh_2
    ), f"dinov2 c++: mpk_m[2]:{mpk_m[2]} is smaller than {mpk_m_thresh_2}"


    mpk_h_thresh_0 = 92.8
    mpk_h_thresh_1 = 80.0
    mpk_h_thresh_2 = 70.0
    assert (
        float(mpk_h[0]) > mpk_h_thresh_0
    ), f"dinov2 c++: mpk_h[0]:{mpk_h[0]} is smaller than {mpk_h_thresh_0}"
    assert (
        float(mpk_h[1]) > mpk_h_thresh_1
    ), f"dinov2 c++: mpk_h[1]:{mpk_h[1]} is smaller than {mpk_h_thresh_1}"
    assert (
        float(mpk_h[2]) > mpk_h_thresh_2
    ), f"dinov2 c++: mpk_h[2]:{mpk_h[2]} is smaller than {mpk_h_thresh_2}"

