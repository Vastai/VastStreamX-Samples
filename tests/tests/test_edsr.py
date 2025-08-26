
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

################ one image test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_super_resolution_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/super_resolution \
    -m /opt/vastai/vaststreamx/data/models/edsr_x2-int8-percentile-1_3_256_256-vacc/mod \
    --vdsp_params ./data/configs/edsr_bgr888.json \
    --device_id {device_id} \
    --denorm "[0,1,1]" \
    --input_file ./data/images/hd_1920x1080.png \
    --output_file sr_result.jpg
    """
    res = run_cmd(cmd)

    assert os.path.exists("sr_result.jpg"), "sr:can't find sr_result.jpg"

    os.system("rm sr_result.jpg")

#################  performance test
@pytest.mark.fast
def test_super_resolution_cpp_performance(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/sr_prof \
    -m /opt/vastai/vaststreamx/data/models/edsr_x2-int8-percentile-1_3_256_256-vacc/mod \
    --vdsp_params ./data/configs/edsr_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 100 \
    --shape [3,256,256] \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 7
    assert float(qps) > qps_thresh, f"edsr:best prof qps {qps} is smaller than {qps_thresh}"


############################# dataset test
@pytest.mark.slow
def test_super_resolution_cpp_precision(device_id):
    os.system("mkdir -p sr_output")
    cmd = f"""
    ./build/vaststreamx-samples/bin/super_resolution \
    -m /opt/vastai/vaststreamx/data/models/edsr_x2-int8-percentile-1_3_256_256-vacc/mod \
    --vdsp_params ./data/configs/edsr_bgr888.json \
    --device_id {device_id} \
    --denorm "[0,1,1]" \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/sr4k/DIV2K_valid_LR_bicubic_X2_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/sr4k/ \
    --dataset_output_folder sr_output
    """

    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/super_resolution/eval_div2k.py \
    --gt /opt/vastai/vaststreamx/data/datasets/sr4k/DIV2K_valid_HR/ \
    --result ./sr_output
    """

    res = run_cmd(cmd)

    accuracys = re.findall(pattern=r"\d+.\d+", string=res)

    mean_psnr = 22.0
    mean_ssim = 0.73

    assert (
        float(accuracys[0]) >= mean_psnr
    ), f"edsr: mean psnr {accuracys[0]} is smaller than {mean_psnr}"
    assert (
        float(accuracys[1]) >= mean_ssim
    ), f"edsr: mean ssim {accuracys[1]} is smaller than {mean_ssim}"

    os.system("rm -r sr_output")


#####################  python test #####################


##############################  one image test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_super_resolution_py(device_id):
    cmd = f"""
    python3 ./samples/super_resolution/super_resolution.py \
    -m /opt/vastai/vaststreamx/data/models/edsr_x2-int8-percentile-1_3_256_256-vacc/mod \
    --vdsp_params ./data/configs/edsr_bgr888.json \
    --device_id {device_id} \
    --denorm "[0,1,1]" \
    --input_file  ./data/images/hd_1920x1080.png \
    --output_file sr_result.jpg
    """
    res = run_cmd(cmd)

    assert os.path.exists("sr_result.jpg"), "sr:can't find sr_result.jpg"

    os.system("rm sr_result.jpg")


#################  performance test
@pytest.mark.fast
def test_super_resolution_py_performance(device_id):
    cmd = f"""
    python3 ./samples/super_resolution/sr_prof.py \
    -m /opt/vastai/vaststreamx/data/models/edsr_x2-int8-percentile-1_3_256_256-vacc/mod \
    --vdsp_params ./data/configs/edsr_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 100 \
    --shape [3,256,256] \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 7
    assert float(qps) > qps_thresh, f"edsr:best prof qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.slow
def test_super_resolution_py_precision(device_id):
    os.system("mkdir -p sr_output")
    cmd = f"""
    python3 ./samples/super_resolution/super_resolution.py \
    -m /opt/vastai/vaststreamx/data/models/edsr_x2-int8-percentile-1_3_256_256-vacc/mod \
    --vdsp_params ./data/configs/edsr_bgr888.json \
    --device_id {device_id} \
    --denorm "[0,1,1]" \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/sr4k/DIV2K_valid_LR_bicubic_X2_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/sr4k/ \
    --dataset_output_folder sr_output
    """

    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/super_resolution/eval_div2k.py \
    --gt /opt/vastai/vaststreamx/data/datasets/sr4k/DIV2K_valid_HR/ \
    --result ./sr_output
    """

    res = run_cmd(cmd)

    accuracys = re.findall(pattern=r"\d+.\d+", string=res)

    mean_psnr = 22.0
    mean_ssim = 0.73

    assert (
        float(accuracys[0]) >= mean_psnr
    ), f"edsr: mean psnr {accuracys[0]} is smaller than {mean_psnr}"
    assert (
        float(accuracys[1]) >= mean_ssim
    ), f"edsr: mean ssim {accuracys[1]} is smaller than {mean_ssim}"

    os.system("rm -r sr_output")
