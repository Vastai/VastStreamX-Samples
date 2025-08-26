
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
def test_rcan_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/super_resolution \
    -m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
    --vdsp_params ./data/configs/rcan_bgr888.json  \
    --device_id {device_id} \
    --input_file  ./data/images/hd_1920x1080.png \
    --output_file rcan_result.jpg
    """
    res = run_cmd(cmd)

    assert os.path.exists("rcan_result.jpg"), "rcan c++:can't find rcan_result.jpg"

    os.system("rm rcan_result.jpg")

#################  performance test
@pytest.mark.fast
def test_rcan_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/sr_prof \
    -m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
    --vdsp_params ./data/configs/rcan_bgr888.json  \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 100 \
    --shape "[3,1080,1920]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 20
    assert (
        float(qps) > qps_thresh
    ), f"rcan c++:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/sr_prof \
    -m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
    --vdsp_params ./data/configs/rcan_bgr888.json  \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 100 \
    --shape "[3,1080,1920]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 12
    assert (
        float(qps) > qps_thresh
    ), f"rcan c++:best prof qps {qps} is smaller than {qps_thresh}"


############################# dataset test
@pytest.mark.slow
def test_rcan_cpp_precision(device_id):
    os.system("mkdir -p rcan_output")
    cmd = f"""
    ./build/vaststreamx-samples/bin/super_resolution \
    -m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
    --vdsp_params ./data/configs/rcan_bgr888.json  \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/sr4k/DIV2K_valid_LR_bicubic_X2_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/sr4k/ \
    --dataset_output_folder rcan_output
    """
    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/super_resolution/evaluation.py \
    --gt_dir /opt/vastai/vaststreamx/data/datasets/sr4k/DIV2K_valid_HR \
    --input_filelist /opt/vastai/vaststreamx/data/datasets/sr4k/DIV2K_valid_LR_bicubic_X2_filelist.txt  \
    --output_dir rcan_output
    """
    res = run_cmd(cmd, check_stderr=False)

    psnr = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"psnr: \d+.\d+", string=res).group()
    ).group()

    ssim = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"ssim: \d+.\d+", string=res).group()
    ).group()

    psnr_thresh = 30.0
    ssim_thresh = 0.85

    assert float(psnr) > psnr_thresh, f"rcan c++: psnr {psnr} is smaller than {psnr_thresh}"
    assert float(ssim) > ssim_thresh, f"rcan c++: ssim {ssim} is smaller than {ssim_thresh}"

    os.system("rm -rf rcan_output")


######################  python test #####################

################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_rcan_py(device_id):
    cmd = f"""
    python3 ./samples/super_resolution/super_resolution.py \
    -m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
    --vdsp_params ./data/configs/rcan_bgr888.json  \
    --device_id {device_id} \
    --input_file  ./data/images/hd_1920x1080.png \
    --output_file rcan_result.jpg
    """
    res = run_cmd(cmd)

    assert os.path.exists("rcan_result.jpg"), "rcan python:can't find rcan_result.jpg"

    os.system("rm rcan_result.jpg")

#################  performance test
@pytest.mark.fast
def test_rcan_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/super_resolution/sr_prof.py \
    -m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
    --vdsp_params ./data/configs/rcan_bgr888.json  \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 100 \
    --shape "[3,1080,1920]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 20
    assert (
        float(qps) > qps_thresh
    ), f"rcan python:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    python3 ./samples/super_resolution/sr_prof.py \
    -m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
    --vdsp_params ./data/configs/rcan_bgr888.json  \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 100 \
    --shape "[3,1080,1920]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 12
    assert (
        float(qps) > qps_thresh
    ), f"rcan python:best prof qps {qps} is smaller than {qps_thresh}"


############################# dataset test
@pytest.mark.slow
def test_rcan_py_precision(device_id):
    os.system("mkdir -p rcan_output")
    cmd = f"""
    python3 ./samples/super_resolution/super_resolution.py \
    -m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
    --vdsp_params ./data/configs/rcan_bgr888.json  \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/sr4k/DIV2K_valid_LR_bicubic_X2_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/sr4k/ \
    --dataset_output_folder rcan_output
    """
    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/super_resolution/evaluation.py \
    --gt_dir /opt/vastai/vaststreamx/data/datasets/sr4k/DIV2K_valid_HR \
    --input_filelist /opt/vastai/vaststreamx/data/datasets/sr4k/DIV2K_valid_LR_bicubic_X2_filelist.txt  \
    --output_dir rcan_output
    """
    res = run_cmd(cmd, check_stderr=False)

    psnr = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"psnr: \d+.\d+", string=res).group()
    ).group()

    ssim = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"ssim: \d+.\d+", string=res).group()
    ).group()

    psnr_thresh = 30.0
    ssim_thresh = 0.85

    assert (
        float(psnr) > psnr_thresh
    ), f"rcan python: psnr {psnr} is smaller than {psnr_thresh}"

    assert (
        float(ssim) > ssim_thresh
    ), f"rcan python: ssim {ssim} is smaller than {ssim_thresh}"

    os.system("rm -rf rcan_output")
