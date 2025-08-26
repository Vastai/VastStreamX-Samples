
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
def test_gpen_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/face_enhancement \
    -m /opt/vastai/vaststreamx/data/models/gpen-int8-mse-1_3_512_512-vacc/mod \
    --vdsp_params ./data/configs/gpen_bgr888.json \
    --device_id {device_id} \
    --input_file ./data/images/face.jpg \
    --output_file gpen_result.jpg
    """
    res = run_cmd(cmd)

    assert os.path.exists("gpen_result.jpg"), "sr:can't find gpen_result.jpg"

    os.system("rm gpen_result.jpg")

#################  performance test
@pytest.mark.fast
def test_gpen_cpp_performance(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/face_enhancement_prof \
    -m /opt/vastai/vaststreamx/data/models/gpen-int8-mse-1_3_512_512-vacc/mod \
    --vdsp_params ./data/configs/gpen_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 100 \
    --shape [3,512,512] \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 19
    assert (
        float(qps) >= qps_thresh
    ), f"face_enhancement:best prof qps {qps} is smaller than {qps_thresh}"


############################# dataset test
@pytest.mark.fast
def test_gpen_cpp_precision(device_id):
    os.system("mkdir -p gpen_output")
    cmd = f"""
    ./build/vaststreamx-samples/bin/face_enhancement \
    -m /opt/vastai/vaststreamx/data/models/gpen-int8-mse-1_3_512_512-vacc/mod \
    --vdsp_params ./data/configs/gpen_bgr888.json \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/GPEN/filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/GPEN/lq \
    --dataset_output_folder gpen_output
    """

    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/face_enhancement/eval_celeb.py \
    --result ./gpen_output \
    --gt /opt/vastai/vaststreamx/data/datasets/GPEN/hq
    """

    res = run_cmd(cmd)

    accuracys = re.findall(pattern=r"\d+.\d+", string=res)

    mean_psnr = 25.8
    mean_ssim = 0.6

    assert (
        float(accuracys[0]) >= mean_psnr
    ), f"face_enhancement: mean psnr {accuracys[0]} is smaller than {mean_psnr}"
    assert (
        float(accuracys[1]) >= mean_ssim
    ), f"face_enhancement: mean ssim {accuracys[1]} is smaller than {mean_ssim}"

    os.system("rm -r gpen_output")


#####################  python test #####################


##############################  one image test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_gpen_py(device_id):
    cmd = f"""
    python3 ./samples/face_enhancement/face_enhancement.py \
    -m /opt/vastai/vaststreamx/data/models/gpen-int8-mse-1_3_512_512-vacc/mod \
    --vdsp_params ./data/configs/gpen_bgr888.json \
    --device_id {device_id} \
    --input_file  ./data/images/face.jpg \
    --output_file gpen_result.jpg
    """
    res = run_cmd(cmd)

    assert os.path.exists("gpen_result.jpg"), "sr:can't find gpen_result.jpg"

    os.system("rm gpen_result.jpg")


#################  performance test
@pytest.mark.fast
def test_gpen_py_performance(device_id):
    cmd = f"""
    python3 ./samples/face_enhancement/face_enhancement_prof.py \
    -m /opt/vastai/vaststreamx/data/models/gpen-int8-mse-1_3_512_512-vacc/mod \
    --vdsp_params ./data/configs/gpen_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 100 \
    --shape [3,512,512] \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 19
    assert (
        float(qps) >= qps_thresh
    ), f"face_enhancement:best prof qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_gpen_py_precision(device_id):
    os.system("mkdir -p gpen_output")
    cmd = f"""
    python3 ./samples/face_enhancement/face_enhancement.py \
    -m /opt/vastai/vaststreamx/data/models/gpen-int8-mse-1_3_512_512-vacc/mod \
    --vdsp_params ./data/configs/gpen_bgr888.json \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/GPEN/filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/GPEN/lq \
    --dataset_output_folder gpen_output
    """

    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/face_enhancement/eval_celeb.py \
    --result ./gpen_output \
    --gt /opt/vastai/vaststreamx/data/datasets/GPEN/hq
    """

    res = run_cmd(cmd)

    accuracys = re.findall(pattern=r"\d+.\d+", string=res)

    mean_psnr = 25.8
    mean_ssim = 0.6

    assert (
        float(accuracys[0]) >= mean_psnr
    ), f"face_enhancement: mean psnr {accuracys[0]} is smaller than {mean_psnr}"
    assert (
        float(accuracys[1]) >= mean_ssim
    ), f"face_enhancement: mean ssim {accuracys[1]} is smaller than {mean_ssim}"

    os.system("rm -r gpen_output")
