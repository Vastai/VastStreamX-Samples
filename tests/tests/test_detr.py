
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
def test_detr_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/detr \
    -m /opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod \
    --vdsp_params ./data/configs/detr_bgr888.json  \
    --device_id {device_id} \
    --threshold 0.5 \
    --label_file ./data/labels/coco91.txt \
    --input_file ./data/images/dog.jpg \
    --output_file detr_result.jpg
    """

    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)


    assert len(scores) == 5, f"detr c++:detected object count={len(scores)} is not 5"
    assert os.path.exists("detr_result.jpg"), "detr c++:can't find detr_result.jpg"
    assert (
        re.findall(pattern=r"bicycle", string=res) != []
    ), "detr c++:can't find bicycle in result string"
    assert (
        re.findall(pattern=r"dog", string=res) != []
    ), "detr c++:can't find dog in result string"
    assert (
        re.findall(pattern=r"truck", string=res) != []
    ), "detr c++:can't find truck in result string"

    os.system("rm detr_result.jpg")


#################  performance test
@pytest.mark.fast
def test_detr_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/detr_prof \
    -m /opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod \
    --vdsp_params ./data/configs/detr_bgr888.json  \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 200 \
    --shape "[3,1066,800]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 27
    assert (
        float(qps) > qps_thresh
    ), f"detr c++:best prof qps {qps} is smaller than {qps_thresh}"


    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/detr_prof \
    -m /opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod \
    --vdsp_params ./data/configs/detr_bgr888.json  \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 150 \
    --shape "[3,1066,800]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 25
    assert (
        float(qps) > qps_thresh
    ), f"detr c++:best latency qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_detr_cpp_precision(device_id):
    os.system("mkdir -p detr_out")
    cmd = f"""
    ./build/vaststreamx-samples/bin/detr \
    -m /opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod \
    --vdsp_params ./data/configs/detr_bgr888.json  \
    --device_id {device_id} \
    --threshold 0.01 \
    --label_file ./data/labels/coco91.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder ./detr_out
    """

    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/detection/eval_map.py \
    --gt ./evaluation/detection/instances_val2017.json \
    --txt ./detr_out
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


    bbox_mAP_thres = 0.37
    bbox_mAP_50_thres = 0.58

    assert (
        float(bbox_mAP) > bbox_mAP_thres
    ), f"detr c++: bbox_mAP {bbox_mAP} is smaller than {bbox_mAP_thres}"
    assert (
        float(bbox_mAP_50) > bbox_mAP_50_thres
    ), f"detr c++: bbox_mAP_50 {bbox_mAP_50} is smaller than {bbox_mAP_50_thres}"

    os.system("rm -rf detr_out")
# #####################  python test #####################

################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_detr_py(device_id):
    cmd = f"""
    python3 ./samples/detection/detr/detr.py \
    -m /opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod \
    --vdsp_params ./data/configs/detr_bgr888.json  \
    --device_id {device_id} \
    --threshold 0.5 \
    --label_file ./data/labels/coco91.txt \
    --input_file ./data/images/dog.jpg \
    --output_file detr_result.jpg
    """
    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)


    assert len(scores) == 5, f"detr python:detected object count={len(scores)} is not 5"
    assert os.path.exists("detr_result.jpg"), "detr python:can't find detr_result.jpg"
    assert (
        re.findall(pattern=r"bicycle", string=res) != []
    ), "detr python:can't find bicycle in result string"
    assert (
        re.findall(pattern=r"dog", string=res) != []
    ), "detr python:can't find dog in result string"
    assert (
        re.findall(pattern=r"truck", string=res) != []
    ), "detr python:can't find truck in result string"

    os.system("rm detr_result.jpg")


#################  performance test

@pytest.mark.fast
def test_detr_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/detection/detr/detr_prof.py \
    -m /opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod \
    --vdsp_params ./data/configs/detr_bgr888.json  \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 200 \
    --shape "[3,1066,800]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 27
    assert (
        float(qps) > qps_thresh
    ), f"detr python:best prof qps {qps} is smaller than {qps_thresh}"


    # 测试最小时延
    cmd = f"""
    python3 ./samples/detection/detr/detr_prof.py \
    -m /opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod \
    --vdsp_params ./data/configs/detr_bgr888.json  \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 150 \
    --shape "[3,1066,800]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 25
    assert (
        float(qps) > qps_thresh
    ), f"detr python:best latency qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_detr_py_precision(device_id):
    os.system("mkdir -p detr_out")
    cmd = f"""
    python3 ./samples/detection/detr/detr.py \
    -m /opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod \
    --vdsp_params ./data/configs/detr_bgr888.json  \
    --device_id {device_id} \
    --threshold 0.01 \
    --label_file ./data/labels/coco91.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder ./detr_out
    """
    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/detection/eval_map.py \
    --gt ./evaluation/detection/instances_val2017.json \
    --txt ./detr_out
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


    bbox_mAP_thres = 0.37
    bbox_mAP_50_thres = 0.58

    assert (
        float(bbox_mAP) > bbox_mAP_thres
    ), f"detr python: bbox_mAP {bbox_mAP} is smaller than {bbox_mAP_thres}"
    assert (
        float(bbox_mAP_50) > bbox_mAP_50_thres
    ), f"detr python: bbox_mAP_50 {bbox_mAP_50} is smaller than {bbox_mAP_50_thres}"

    os.system("rm -rf detr_out")
