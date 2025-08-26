
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
def test_yoloxm_640_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/detection \
    -m /opt/vastai/vaststreamx/data/models/yolox_m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolox_equal_bgrplanar.json \
    --device_id {device_id} \
    --threshold 0.5 \
    --label_file ./data/labels/coco2id.txt \
    --input_file ./data/images/dog.jpg \
    --output_file result.png
    """
    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thresh = 0.75

    for score in scores:
        if float(score) < score_thresh:
            raise RuntimeError(f"detection:score {score} is smaller than {score_thresh}")

    assert len(scores) == 3, f"detection:detected object count={len(scores)} is not 3"
    assert os.path.exists("result.png"), "detection:can't find result.png"
    assert (
        re.findall(pattern=r"bicycle", string=res) != []
    ), "detection:can't find bicycle in result string"
    assert (
        re.findall(pattern=r"dog", string=res) != []
    ), "detection:can't find dog in result string"
    assert (
        re.findall(pattern=r"truck", string=res) != []
    ), "detection:can't find truck in result string"

    os.system("rm result.png")

#################  performance test
@pytest.mark.fast
def test_yoloxm_640_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/det_prof \
    -m /opt/vastai/vaststreamx/data/models/yolox_m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolox_equal_bgrplanar.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 2000 \
    --shape [3,640,640] \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 370
    assert (
        float(qps) > qps_thresh
    ), f"detection:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/det_prof \
    -m /opt/vastai/vaststreamx/data/models/yolox_m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolox_equal_bgrplanar.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 1000 \
    --shape [3,640,640] \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 260
    assert (
        float(qps) > qps_thresh
    ), f"detection:best latancy qps {qps} is smaller than {qps_thresh}"


############################# dataset test
@pytest.mark.fast
def test_yoloxm_640_cpp_precision(device_id):
    os.system("mkdir -p yoloxm_out")
    cmd = f"""
    ./build/vaststreamx-samples/bin/detection \
    -m /opt/vastai/vaststreamx/data/models/yolox_m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolox_equal_bgrplanar.json \
    --device_id {device_id} \
    --threshold 0.01 \
    --label_file ./data/labels/coco2id.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder ./yoloxm_out
    """
    run_cmd(cmd,False)

    cmd = f"""
    python3 ./evaluation/detection/eval_map.py \
    --gt ./evaluation/detection/instances_val2017.json \
    --txt ./yoloxm_out
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

    bbox_mAP_thres = 0.41
    bbox_mAP_50_thres = 0.57

    assert (
        float(bbox_mAP) > bbox_mAP_thres
    ), f"detection: bbox_mAP {bbox_mAP} is smaller than {bbox_mAP_thres}"
    assert (
        float(bbox_mAP_50) > bbox_mAP_50_thres
    ), f"detection: bbox_mAP_50 {bbox_mAP_50} is smaller than {bbox_mAP_50_thres}"

    os.system("rm -rf yoloxm_out")
# #####################  python test #####################

################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_yoloxm_640_py(device_id):
    cmd = f"""
    python3 ./samples/detection/det_yolo/detection.py \
    -m /opt/vastai/vaststreamx/data/models/yolox_m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolox_equal_bgrplanar.json \
    --device_id {device_id} \
    --threshold 0.5 \
    --label_file ./data/labels/coco2id.txt \
    --input_file ./data/images/dog.jpg \
    --output_file result.png
    """
    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.75

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(f"detection:score {score} is smaller than {score_thres}")

    assert len(scores) == 3, f"detection:detected object count={len(scores)} is not 3"
    assert os.path.exists("result.png"), "detection:can't find result.png"
    assert (
        re.findall(pattern=r"bicycle", string=res) != []
    ), "detection:can't find bicycle in result string"
    assert (
        re.findall(pattern=r"dog", string=res) != []
    ), "detection:can't find dog in result string"
    assert (
        re.findall(pattern=r"truck", string=res) != []
    ), "detection:can't find truck in result string"

    os.system("rm result.png")
#################  performance test
@pytest.mark.fast
def test_yoloxm_640_py_performance(device_id):
    # 最佳性能与时延
    cmd = f"""
    python3 ./samples/detection/det_yolo/det_prof.py \
    -m /opt/vastai/vaststreamx/data/models/yolox_m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolox_equal_bgrplanar.json \
    --device_ids [{device_id}]  \
    --batch_size 1 \
    --instance 1 \
    --iterations 2000 \
    --shape [3,640,640] \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 370
    assert (
        float(qps) > qps_thresh
    ), f"detection:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最佳时延与吞吐
    cmd = f"""
    python3 ./samples/detection/det_yolo/det_prof.py \
    -m /opt/vastai/vaststreamx/data/models/yolox_m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolox_equal_bgrplanar.json \
    --device_ids [{device_id}]  \
    --batch_size 1 \
    --instance 1 \
    --iterations 1000 \
    --shape [3,640,640] \
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
    ), f"detection:best latancy qps {qps} is smaller than {qps_thresh}"


############################# dataset test
@pytest.mark.fast
def test_yoloxm_640_py_precision(device_id):
    os.system("mkdir -p yoloxm_out")
    cmd = f"""
    python3 ./samples/detection/det_yolo/detection.py \
    -m /opt/vastai/vaststreamx/data/models/yolox_m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolox_equal_bgrplanar.json \
    --device_id {device_id} \
    --threshold 0.01 \
    --label_file ./data/labels/coco2id.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder ./yoloxm_out
    """
    run_cmd(cmd,False)

    cmd = f"""
    python3 ./evaluation/detection/eval_map.py \
    --gt ./evaluation/detection/instances_val2017.json \
    --txt ./yoloxm_out
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

    bbox_mAP_thres = 0.41
    bbox_mAP_50_thres = 0.57

    assert (
        float(bbox_mAP) > bbox_mAP_thres
    ), f"detection: bbox_mAP {bbox_mAP} is smaller than {bbox_mAP_thres}"
    assert (
        float(bbox_mAP_50) > bbox_mAP_50_thres
    ), f"detection: bbox_mAP_50 {bbox_mAP_50} is smaller than {bbox_mAP_50_thres}"

    os.system("rm -rf yoloxm_out")
