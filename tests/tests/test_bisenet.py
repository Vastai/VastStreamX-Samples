
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
def test_bisenet_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/bisenet \
    --model_prefix /opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod \
    --vdsp_params ./data/configs/bisenet_bgr888.json \
    --device_id {device_id} \
    --input_file ./data/images/face.jpg \
    --output_file bisenet_result.jpg
    """

    res = run_cmd(cmd)

    assert os.path.exists("bisenet_result.jpg"), "bisenet c++:can't find bisenet_result.jpg"

    os.system("rm bisenet_result.jpg")

#################  performance test
@pytest.mark.fast
def test_bisenet_cpp_performance(device_id):

    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/bisenet_prof  \
    --model_prefix /opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod \
    --vdsp_params ./data/configs/bisenet_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 2 \
    --instance 2 \
    --shape [3,512,512] \
    --iterations 3000 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 650
    assert (
        float(qps) > qps_thresh
    ), f"bisenet:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/bisenet_prof  \
    --model_prefix /opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod \
    --vdsp_params ./data/configs/bisenet_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --shape [3,512,512] \
    --iterations 2000 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 170
    assert (
        float(qps) > qps_thresh
    ), f"bisenet:best latancy qps {qps} is smaller than {qps_thresh}"




############################# dataset test
@pytest.mark.fast
def test_bisenet_cpp_percision(device_id):
    os.system("mkdir -p bisenet_output")

    cmd = f"""
    ./build/vaststreamx-samples/bin/bisenet \
    --model_prefix /opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod \
    --vdsp_params ./data/configs/bisenet_bgr888.json \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/bisenet/bisegnet_test_img_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/bisenet/ \
    --dataset_output_folder ./bisenet_output
    """
    run_cmd(cmd)

    cmd = f"""
    python3 evaluation/bisenet/zllrunning_vamp_eval.py \
    --src_dir /opt/vastai/vaststreamx/data/datasets/bisenet/bisegnet_test_img \
    --gt_dir /opt/vastai/vaststreamx/data/datasets/bisenet/bisegnet_test_mask \
    --input_npz_path /opt/vastai/vaststreamx/data/datasets/bisenet/bisegnet_test_img_filelist.txt \
    --out_npz_dir ./bisenet_output \
    --input_shape 512 512 \
    --vamp_flag
    """
    res = run_cmd(cmd)

    iou = re.search(
        pattern=r"IoU:\d+.\d+", string=res.replace(" ", "").replace("\t", "")
    ).group()
    mean_iou = re.search(pattern=r"\d+.\d+", string=iou).group()
    mean_iou_thresh = 0.73

    assert (
        float(mean_iou) > mean_iou_thresh
    ), f"bisenet:mean_iou {mean_iou} is smaller than {mean_iou_thresh}"

    os.system("rm -rf bisenet_output")

#####################  python test #####################
###############################  one image  test

@pytest.mark.fast
@pytest.mark.ai_integration
def test_bisenet_py(device_id):
    cmd = f"""
    python3 samples/segmentation/bisenet/bisenet.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod \
    --vdsp_params ./data/configs/bisenet_bgr888.json \
    --device_id {device_id} \
    --input_file ./data/images/face.jpg \
    --output_file bisenet_result.jpg
    """
    res = run_cmd(cmd)

    assert os.path.exists("bisenet_result.jpg"), "bisenet c++:can't find bisenet_result.jpg"

    os.system("rm bisenet_result.jpg")

#################  performance test
@pytest.mark.fast
def test_bisenet_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 samples/segmentation/bisenet/bisenet_prof.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod \
    --vdsp_params ./data/configs/bisenet_bgr888.json \
    --device_ids [{device_id}]  \
    --batch_size 2 \
    --instance 2 \
    --shape [3,512,512] \
    --iterations 3000 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1
    """


    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 650
    assert (
        float(qps) > qps_thresh
    ), f"bisenet:best prof qps {qps} is smaller than {qps_thresh}"


    # 测试最小时延
    cmd = f"""
    python3 samples/segmentation/bisenet/bisenet_prof.py  \
    --model_prefix /opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod \
    --vdsp_params ./data/configs/bisenet_bgr888.json \
    --device_ids [{device_id}]  \
    --batch_size 1 \
    --instance 1 \
    --shape [3,512,512] \
    --iterations 2000 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 170
    assert (
        float(qps) > qps_thresh
    ), f"bisenet:best latancy qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_bisenet_py_percision(device_id):
    os.system("mkdir -p bisenet_output")

    cmd = f"""
    python3 samples/segmentation/bisenet/bisenet.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod \
    --vdsp_params ./data/configs/bisenet_bgr888.json \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/bisenet/bisegnet_test_img_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/bisenet/ \
    --dataset_output_folder ./bisenet_output
    """
    run_cmd(cmd)

    cmd = f"""
    python3 evaluation/bisenet/zllrunning_vamp_eval.py \
    --src_dir /opt/vastai/vaststreamx/data/datasets/bisenet/bisegnet_test_img \
    --gt_dir /opt/vastai/vaststreamx/data/datasets/bisenet/bisegnet_test_mask \
    --input_npz_path /opt/vastai/vaststreamx/data/datasets/bisenet/bisegnet_test_img_filelist.txt \
    --out_npz_dir ./bisenet_output \
    --input_shape 512 512 \
    --vamp_flag
    """
    res = run_cmd(cmd)

    iou = re.search(
        pattern=r"IoU:\d+.\d+", string=res.replace(" ", "").replace("\t", "")
    ).group()
    mean_iou = re.search(pattern=r"\d+.\d+", string=iou).group()
    mean_iou_thresh = 0.73

    assert (
        float(mean_iou) > mean_iou_thresh
    ), f"bisenet:mean_iou {mean_iou} is smaller than {mean_iou_thresh}"

    os.system("rm -rf bisenet_output")
