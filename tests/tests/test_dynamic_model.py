
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
def test_dynamic_yolo_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/dynamic_yolo \
    -m /opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json \
    --vdsp_params ./data/configs/yolo_div255_bgr888.json \
    --device_id {device_id} \
    --max_input_shape "[1,3,640,640]" \
    --threshold 0.5 \
    --label_file ./data/labels/coco2id.txt \
    --input_file ./data/images/dog.jpg \
    --output_file dynamic_model_result.jpg
    """
    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.5

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(f"dynamic_yolo:score {score} is smaller than {score_thres}")
    obj_count = 4
    assert len(scores) == obj_count, f"dynamic_yolo:detected object count={len(scores)} is not {obj_count}"
    assert os.path.exists(
        "dynamic_model_result.jpg"
    ), "dynamic_yolo:can't find dynamic_model_result.jpg"
    assert (
        re.findall(pattern=r"bicycle", string=res) != []
    ), "dynamic_yolo c++:can't find bicycle in result string"
    assert (
        re.findall(pattern=r"dog", string=res) != []
    ), "dynamic_yolo c++:can't find dog in result string"
    assert (
        re.findall(pattern=r"car", string=res) != []
    ), "dynamic_yolo c++:can't find car in result string"


    os.system("rm dynamic_model_result.jpg")

#################  performance test

@pytest.mark.fast
def test_dynamic_yolo_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/dynamic_yolo_prof \
    -m /opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json \
    --vdsp_params ./data/configs/yolo_div255_bgr888.json \
    --device_ids [{device_id}] \
    --max_input_shape "[1,3,640,640]" \
    --threshold 0.5 \
    --batch_size 1 \
    --instance 2 \
    --iterations 5000 \
    --shape "[1,3,640,640]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 850
    assert (
        float(qps) > qps_thresh
    ), f"dynamic_yolo c++:best prof qps {qps} is smaller than {qps_thresh}"


    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/dynamic_yolo_prof \
    -m /opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json \
    --vdsp_params ./data/configs/yolo_div255_bgr888.json \
    --device_ids [{device_id}] \
    --max_input_shape "[1,3,640,640]" \
    --threshold 0.5 \
    --batch_size 1 \
    --instance 1 \
    --iterations 2000 \
    --shape "[1,3,640,640]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 270
    assert (
        float(qps) > qps_thresh
    ), f"dynamic_yolo c++ :best prof qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_dynamic_yolo_cpp_precision(device_id):
    os.system("mkdir -p dynamic_out")
    cmd = f"""
    ./build/vaststreamx-samples/bin/dynamic_yolo \
    -m /opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json \
    --vdsp_params ./data/configs/yolo_div255_bgr888.json \
    --device_id {device_id} \
    --max_input_shape "[1,3,640,640]" \
    --threshold 0.01 \
    --label_file ./data/labels/coco2id.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder ./dynamic_out
    """
    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/detection/eval_map.py \
    --gt ./evaluation/detection/instances_val2017.json \
    --txt ./dynamic_out
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

    bbox_mAP_thres = 0.355
    bbox_mAP_50_thres = 0.545

    assert (
        float(bbox_mAP) > bbox_mAP_thres
    ), f"dynamic_yolo c++: bbox_mAP {bbox_mAP} is smaller than {bbox_mAP_thres}"
    assert (
        float(bbox_mAP_50) > bbox_mAP_50_thres
    ), f"dynamic_yolo c++ : bbox_mAP_50 {bbox_mAP_50} is smaller than {bbox_mAP_50_thres}"

    os.system("rm -rf dynamic_out")

# #####################  python test #####################

################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_dynamic_yolo_py(device_id):
    cmd = f"""
    python3 ./samples/dynamic_model/dynamic_yolo.py \
    -m /opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json \
    --vdsp_params ./data/configs/yolo_div255_bgr888.json \
    --device_id {device_id} \
    --max_input_shape "[1,3,640,640]" \
    --threshold 0.5 \
    --label_file ./data/labels/coco2id.txt \
    --input_file ./data/images/dog.jpg \
    --output_file dynamic_model_result.jpg
    """
    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)
    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.5

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(f"dynamic_yolo:score {score} is smaller than {score_thres}")

    obj_count = 4
    assert len(scores) == obj_count, f"dynamic_yolo:detected object count={len(scores)} is not {obj_count}"
    assert os.path.exists(
        "dynamic_model_result.jpg"
    ), "dynamic_yolo:can't find dynamic_model_result.jpg"
    assert (
        re.findall(pattern=r"bicycle", string=res) != []
    ), "dynamic_yolo python:can't find bicycle in result string"
    assert (
        re.findall(pattern=r"dog", string=res) != []
    ), "dynamic_yolo python:can't find dog in result string"
    assert (
        re.findall(pattern=r"car", string=res) != []
    ), "dynamic_yolo python:can't find car in result string"

    os.system("rm dynamic_model_result.jpg")

#################  performance test

@pytest.mark.fast
def test_dynamic_yolo_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/dynamic_model/dynamic_yolo_prof.py \
    -m /opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json \
    --vdsp_params ./data/configs/yolo_div255_bgr888.json \
    --max_input_shape "[1,3,640,640]" \
    --model_input_shape "[1,3,640,640]" \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 2 \
    --iterations 5000 \
    --shape "[3,640,640]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 850
    assert (
        float(qps) > qps_thresh
    ), f"dynamic_yolo python:best prof qps {qps} is smaller than {qps_thresh}"


    # 测试最小时延
    cmd = f"""
    python3 ./samples/dynamic_model/dynamic_yolo_prof.py \
    -m /opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json \
    --vdsp_params ./data/configs/yolo_div255_bgr888.json \
    --max_input_shape "[1,3,640,640]" \
    --model_input_shape "[1,3,640,640]" \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 2000 \
    --shape "[3,640,640]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 265
    assert (
        float(qps) > qps_thresh
    ), f"dynamic_yolo python:best prof qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_dynamic_yolo_py_precision(device_id):
    os.system("mkdir -p dynamic_out")
    cmd = f"""
    python3 ./samples/dynamic_model/dynamic_yolo.py \
    -m /opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json \
    --vdsp_params ./data/configs/yolo_div255_bgr888.json \
    --device_id {device_id} \
    --max_input_shape "[1,3,640,640]" \
    --threshold 0.01 \
    --label_file ./data/labels/coco2id.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder ./dynamic_out
    """
    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/detection/eval_map.py \
    --gt ./evaluation/detection/instances_val2017.json \
    --txt ./dynamic_out
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

    bbox_mAP_thres = 0.355
    bbox_mAP_50_thres = 0.545

    assert (
        float(bbox_mAP) > bbox_mAP_thres
    ), f"dynamic_yolo python: bbox_mAP {bbox_mAP} is smaller than {bbox_mAP_thres}"
    assert (
        float(bbox_mAP_50) > bbox_mAP_50_thres
    ), f"dynamic_yolo python: bbox_mAP_50 {bbox_mAP_50} is smaller than {bbox_mAP_50_thres}"

    os.system("rm -rf dynamic_out")
