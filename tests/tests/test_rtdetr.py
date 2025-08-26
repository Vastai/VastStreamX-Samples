
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
def test_rtdetr_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/rtdetr \
    -m /opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod \
    --vdsp_params ./data/configs/rtdetr_bgr888.json  \
    --device_id {device_id} \
    --threshold 0.5 \
    --label_file ./data/labels/coco2id.txt \
    --input_file ./data/images/dog.jpg \
    --output_file rtdetr_result.jpg
    """
    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    assert len(scores) == 3, f"rtdetr c++:detected object count={len(scores)} is not 3"
    assert os.path.exists("rtdetr_result.jpg"), "rtdetr c++:can't find rtdetr_result.jpg"
    assert (
        re.findall(pattern=r"bicycle", string=res) != []
    ), "rtdetr c++:can't find bicycle in result string"
    assert (
        re.findall(pattern=r"dog", string=res) != []
    ), "rtdetr c++:can't find dog in result string"
    assert (
        re.findall(pattern=r"truck", string=res) != []
    ), "rtdetr c++:can't find truck in result string"

    os.system("rm rtdetr_result.jpg")


#################  performance test

@pytest.mark.fast
def test_rtdetr_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/rtdetr_prof \
    -m /opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod \
    --vdsp_params ./data/configs/rtdetr_bgr888.json  \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 100 \
    --shape "[3,640,640]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 16
    assert (
        float(qps) > qps_thresh
    ), f"rtdetr c++:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/rtdetr_prof \
    -m /opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod \
    --vdsp_params ./data/configs/rtdetr_bgr888.json  \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 100 \
    --shape "[3,640,640]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 15
    assert (
        float(qps) > qps_thresh
    ), f"rtdetr c++:best latency qps {qps} is smaller than {qps_thresh}"


############################# dataset test
@pytest.mark.fast
def test_rtdetr_cpp_precision(device_id):
    os.system("mkdir -p rtdetr_out")
    cmd = f"""
    ./build/vaststreamx-samples/bin/rtdetr \
    -m /opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod \
    --vdsp_params ./data/configs/rtdetr_bgr888.json  \
    --device_id {device_id} \
    --threshold 0.001 \
    --label_file ./data/labels/coco2id.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder ./rtdetr_out
    """
    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/detection/eval_map.py \
    --gt ./evaluation/detection/instances_val2017.json \
    --txt ./rtdetr_out
    """
    res = run_cmd(cmd)

    bbox_mAP = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"'bbox_mAP': \d+.\d+", string=res).group(),
    ).group()

    bbox_mAP_50 = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"'bbox_mAP_50': \d+.\d+", string=res).group(),
    ).group()

    bbox_mAP_thres = 0.44
    bbox_mAP_50_thres = 0.61

    assert (
        float(bbox_mAP) > bbox_mAP_thres
    ), f"rtdetr c++: bbox_mAP {bbox_mAP} is smaller than {bbox_mAP_thres}"
    assert (
        float(bbox_mAP_50) > bbox_mAP_50_thres
    ), f"rtdetr c++: bbox_mAP_50 {bbox_mAP_50} is smaller than {bbox_mAP_50_thres}"

    os.system("rm -rf rtdetr_out")


# #####################  python test #####################

################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_rtdetr_py(device_id):
    cmd = f"""
    python3 ./samples/detection/rtdetr/rtdetr.py \
    -m /opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod \
    --vdsp_params ./data/configs/rtdetr_bgr888.json  \
    --device_id {device_id} \
    --threshold 0.5 \
    --label_file ./data/labels/coco2id.txt \
    --input_file ./data/images/dog.jpg \
    --output_file rtdetr_result.jpg
    """
    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    assert len(scores) == 3, f"rtdetr python:detected object count={len(scores)} is not 3"
    assert os.path.exists("rtdetr_result.jpg"), "rtdetr python:can't find rtdetr_result.jpg"
    assert (
        re.findall(pattern=r"bicycle", string=res) != []
    ), "rtdetr python:can't find bicycle in result string"
    assert (
        re.findall(pattern=r"dog", string=res) != []
    ), "rtdetr python:can't find dog in result string"
    assert (
        re.findall(pattern=r"truck", string=res) != []
    ), "rtdetr python:can't find truck in result string"

    os.system("rm rtdetr_result.jpg")




#################  performance test

@pytest.mark.fast
def test_rtdetr_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/detection/rtdetr/rtdetr_prof.py \
    -m /opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod \
    --vdsp_params ./data/configs/rtdetr_bgr888.json  \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 100 \
    --shape "[3,640,640]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 16
    assert (
        float(qps) > qps_thresh
    ), f"rtdetr python:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    python3 ./samples/detection/rtdetr/rtdetr_prof.py \
    -m /opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod \
    --vdsp_params ./data/configs/rtdetr_bgr888.json  \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 100 \
    --shape "[3,640,640]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 15
    assert (
        float(qps) > qps_thresh
    ), f"rtdetr python:best latency qps {qps} is smaller than {qps_thresh}"




############################# dataset test
@pytest.mark.fast
def test_rtdetr_py_precision(device_id):
    os.system("mkdir -p rtdetr_out")
    cmd = f"""
    python3 ./samples/detection/rtdetr/rtdetr.py \
    -m /opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod \
    --vdsp_params ./data/configs/rtdetr_bgr888.json  \
    --device_id {device_id} \
    --threshold 0.001 \
    --label_file ./data/labels/coco2id.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder ./rtdetr_out
    """
    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/detection/eval_map.py \
    --gt ./evaluation/detection/instances_val2017.json \
    --txt ./rtdetr_out
    """
    res = run_cmd(cmd)

    bbox_mAP = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"'bbox_mAP': \d+.\d+", string=res).group(),
    ).group()

    bbox_mAP_50 = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"'bbox_mAP_50': \d+.\d+", string=res).group(),
    ).group()

    bbox_mAP_thres = 0.44
    bbox_mAP_50_thres = 0.61

    assert (
        float(bbox_mAP) > bbox_mAP_thres
    ), f"rtdetr python: bbox_mAP {bbox_mAP} is smaller than {bbox_mAP_thres}"
    assert (
        float(bbox_mAP_50) > bbox_mAP_50_thres
    ), f"rtdetr python: bbox_mAP_50 {bbox_mAP_50} is smaller than {bbox_mAP_50_thres}"

    os.system("rm -rf rtdetr_out")
