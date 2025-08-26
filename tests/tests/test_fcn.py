
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
def test_fcn_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/fcn \
    -m /opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod \
    --vdsp_params ./data/configs/fcn_bgr888.json \
    --device_id {device_id} \
    --input_file ./data/images/cycling.jpg \
    --output_file fcn_result.jpg
    """
    res = run_cmd(cmd)

    assert os.path.exists("fcn_result.jpg"), "fcn:can't fcn_result.jpg"

    os.system("rm fcn_result.jpg")

#################  performance test
@pytest.mark.fast
def test_fcn_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/fcn_prof \
    -m /opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod \
    --vdsp_params ./data/configs/fcn_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 6 \
    --instance 1 \
    --shape [3,320,320] \
    --iterations 1000 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 650
    assert float(qps) > qps_thresh, f"fcn:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/fcn_prof  \
    -m /opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod \
    --vdsp_params ./data/configs/fcn_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --shape [3,320,320] \
    --iterations 1000 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 250
    assert (
        float(qps) > qps_thresh
    ), f"fcn:best latancy qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_fcn_cpp_precision(device_id):
    os.system("mkdir -p fcn_out")
    cmd = f"""
    ./build/vaststreamx-samples/bin/fcn \
    -m /opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod \
    --vdsp_params ./data/configs/fcn_bgr888.json \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/segmentation/JPEGImages_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/segmentation/ \
    --dataset_output_folder fcn_out
    """
    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/fcn/awesome_vamp_eval.py \
    --src_dir /opt/vastai/vaststreamx/data/datasets/segmentation/JPEGImages_val \
    --gt_dir /opt/vastai/vaststreamx/data/datasets/segmentation/SegmentationClass \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/segmentation/JPEGImages_val_filelist.txt \
    --out_npz_dir ./fcn_out \
    --input_shape 320 320 \
    --vamp_flag
    """

    res = run_cmd(cmd)

    acc_strs = re.findall(pattern=r"pixAcc: \d+.\d+", string=res)
    iou_strs = re.findall(pattern=r"mIoU: \d+.\d+", string=res)

    acc_len = 0
    acc_sum = 0.0
    for acc_str in acc_strs:
        acc = float(re.findall(pattern=r"\d+.\d+", string=acc_str)[0])
        acc_sum += acc
        acc_len += 1

    iou_len = 0
    iou_sum = 0.0
    for iou_str in iou_strs:
        iou = float(re.findall(pattern=r"\d+.\d+", string=iou_str)[0])
        iou_sum += iou
        iou_len += 1

    assert iou_len == acc_len, f"iou_len {iou_len} != acc_len {acc_len}"

    acc_ave = acc_sum / acc_len
    iou_ave = iou_sum / iou_len
    acc_thresh = 87
    iou_thresh = 48

    assert acc_ave > acc_thresh, f"acc_ave {acc_ave} is smaller than {acc_thresh}"
    assert iou_ave > iou_thresh, f"iou_ave {iou_ave} is smaller than {iou_thresh}"

    os.system("rm -rf fcn_out")

# #####################  python test #####################


###############################  one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_fcn_py(device_id):
    cmd = f"""
    python3 samples/segmentation/fcn/fcn.py \
    -m /opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod \
    --vdsp_params ./data/configs/fcn_bgr888.json \
    --device_id {device_id} \
    --input_file ./data/images/cycling.jpg \
    --output_file fcn_result.jpg
    """
    res = run_cmd(cmd)

    assert os.path.exists("fcn_result.jpg"), "fcn:can't fcn_result.jpg"

    os.system("rm fcn_result.jpg")

#################  performance test

@pytest.mark.fast
def test_fcn_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 samples/segmentation/fcn/fcn_prof.py \
    -m /opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod \
    --vdsp_params ./data/configs/fcn_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 6 \
    --instance 1 \
    --shape [3,320,320] \
    --iterations 1000 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 600
    assert float(qps) > qps_thresh, f"fcn:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    python3 samples/segmentation/fcn/fcn_prof.py  \
    -m /opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod \
    --vdsp_params ./data/configs/fcn_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --shape [3,320,320] \
    --iterations 1000 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 250
    assert (
        float(qps) > qps_thresh
    ), f"fcn:best latancy qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_fcn_py_precision(device_id):
    os.system("mkdir -p fcn_out")
    cmd = f"""
    python3 samples/segmentation/fcn/fcn.py \
    -m /opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod \
    --vdsp_params ./data/configs/fcn_bgr888.json \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/segmentation/JPEGImages_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/segmentation/ \
    --dataset_output_folder fcn_out
    """
    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/fcn/awesome_vamp_eval.py \
    --src_dir /opt/vastai/vaststreamx/data/datasets/segmentation/JPEGImages_val \
    --gt_dir /opt/vastai/vaststreamx/data/datasets/segmentation/SegmentationClass \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/segmentation/JPEGImages_val_filelist.txt \
    --out_npz_dir ./fcn_out \
    --input_shape 320 320 \
    --vamp_flag
    """

    res = run_cmd(cmd)

    acc_strs = re.findall(pattern=r"pixAcc: \d+.\d+", string=res)
    iou_strs = re.findall(pattern=r"mIoU: \d+.\d+", string=res)

    acc_len = 0
    acc_sum = 0.0
    for acc_str in acc_strs:
        acc = float(re.findall(pattern=r"\d+.\d+", string=acc_str)[0])
        acc_sum += acc
        acc_len += 1

    iou_len = 0
    iou_sum = 0.0
    for iou_str in iou_strs:
        iou = float(re.findall(pattern=r"\d+.\d+", string=iou_str)[0])
        iou_sum += iou
        iou_len += 1

    assert iou_len == acc_len, f"iou_len {iou_len} != acc_len {acc_len}"

    acc_ave = acc_sum / acc_len
    iou_ave = iou_sum / iou_len
    acc_thresh = 87
    iou_thresh = 48

    assert acc_ave > acc_thresh, f"acc_ave {acc_ave} is smaller than {acc_thresh}"
    assert iou_ave > iou_thresh, f"iou_ave {iou_ave} is smaller than {iou_thresh}"

    os.system("rm -rf fcn_out")
