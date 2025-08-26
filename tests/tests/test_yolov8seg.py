
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
def test_yolov8seg_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/yolov8_seg \
    -m /opt/vastai/vaststreamx/data/models/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolov8seg_bgr888.json \
    --device_id {device_id} \
    --elf_file /opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc \
    --threshold 0.4 \
    --label_file ./data/labels/coco2id.txt \
    --input_file ./data/images/cycling.jpg \
    --output_file yolov8_seg_result.jpg
    """
    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.43

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(
                f"yolov8_seg c++:score {score} is smaller than {score_thres}"
            )

    assert len(scores) == 5, f"yolov8_seg c++:detected object count={len(scores)} is not 5"
    assert os.path.exists(
        "yolov8_seg_result.jpg"
    ), "yolov8_seg c++:can't yolov8_seg_result.jpg"
    assert (
        re.findall(pattern=r"bicycle", string=res) != []
    ), "yolov8_seg c++:can't find bicycle in result string"
    assert (
        re.findall(pattern=r"person", string=res) != []
    ), "yolov8_seg c++:can't find dog in result string"

    os.system("rm yolov8_seg_result.jpg")


#################  performance test
@pytest.mark.fast
def test_yolov8seg_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/yolov8_seg_prof \
    -m /opt/vastai/vaststreamx/data/models/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolov8seg_bgr888.json \
    --elf_file /opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc \
    --device_ids [{device_id}] \
    --shape "[3,640,640]" \
    --batch_size 1 \
    --instance 6 \
    --iterations 600 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 95
    assert (
        float(qps) > qps_thresh
    ), f"yolov8_seg c++:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/yolov8_seg_prof \
    -m /opt/vastai/vaststreamx/data/models/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolov8seg_bgr888.json \
    --elf_file /opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc \
    --device_ids [{device_id}] \
    --shape "[3,640,640]" \
    --batch_size 1 \
    --instance 1 \
    --iterations 100 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 15
    assert (
        float(qps) > qps_thresh
    ), f"yolov8_seg c++:best latency qps {qps} is smaller than {qps_thresh}"


############################# dataset test
@pytest.mark.slow
def test_yolov8seg_cpp_precision(device_id):
    os.system("mkdir -p yolov8_seg_out")
    cmd = f"""
    ./build/vaststreamx-samples/bin/yolov8_seg \
    -m /opt/vastai/vaststreamx/data/models/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolov8seg_bgr888.json \
    --device_id {device_id} \
    --elf_file /opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc \
    --threshold 0.01 \
    --label_file ./data/labels/coco2id.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder ./yolov8_seg_out
    """
    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/yolov8_seg/yolov8_seg_eval.py \
    --output_path yolov8_seg_out \
    --gt ./evaluation/yolov8_seg/instances_val2017.json
    """
    res = run_cmd(cmd)

    mAP_strs = re.findall(pattern=r"] = \d+.\d+", string=res)

    mAPs = []
    for map_str in mAP_strs:
        mAP = re.search(pattern=r"\d+.\d+", string=map_str).group()
        mAPs.append(mAP)

    det_mAP_50_95_thres = 0.47
    det_mAP_50_thres = 0.63

    seg_mAP_50_95_thres = 0.37
    seg_mAP_50_thres = 0.59

    assert len(mAPs) == 24, f"len(mAPs)={len(mAPs)} is not equal 24"
    assert (
        float(mAPs[0]) > det_mAP_50_95_thres
    ), f"yolov8_seg c++: (mAPs[0] {mAPs[0]} is smaller than {det_mAP_50_95_thres}"

    assert (
        float(mAPs[1]) > det_mAP_50_thres
    ), f"yolov8_seg c++: (mAPs[1] {mAPs[1]} is smaller than {det_mAP_50_thres}"

    assert (
        float(mAPs[12]) > seg_mAP_50_95_thres
    ), f"yolov8_seg c++: (mAPs[12] {mAPs[12]} is smaller than {seg_mAP_50_95_thres}"

    assert (
        float(mAPs[13]) > seg_mAP_50_thres
    ), f"yolov8_seg c++: (mAPs[13] {mAPs[13]} is smaller than {seg_mAP_50_thres}"

    os.system("rm -rf yolov8_seg_out yolov8seg_predictions.json")


# #####################  python test #####################

################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_yolov8seg_py(device_id):
    cmd = f"""
    python3 ./samples/segmentation/yolov8_seg/yolov8_seg.py \
    --vdsp_params ./data/configs/yolov8seg_bgr888.json \
    --device_id {device_id} \
    --elf_file /opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc \
    --threshold 0.4 \
    --label_file ./data/labels/coco2id.txt \
    --input_file ./data/images/cycling.jpg \
    --output_file yolov8_seg_result.jpg
    """
    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.43

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(
                f"yolov8_seg python:score {score} is smaller than {score_thres}"
            )

    assert (
        len(scores) == 5
    ), f"yolov8_seg python:detected object count={len(scores)} is not 5"
    assert os.path.exists(
        "yolov8_seg_result.jpg"
    ), "yolov8_seg python:can't yolov8_seg_result.jpg"
    assert (
        re.findall(pattern=r"bicycle", string=res) != []
    ), "yolov8_seg python:can't find bicycle in result string"
    assert (
        re.findall(pattern=r"person", string=res) != []
    ), "yolov8_seg python:can't find dog in result string"

    os.system("rm yolov8_seg_result.jpg")


#################  performance test
@pytest.mark.fast
def test_yolov8seg_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/segmentation/yolov8_seg/yolov8_seg_prof.py \
    -m /opt/vastai/vaststreamx/data/models/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolov8seg_bgr888.json \
    --elf_file /opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc \
    --device_ids [{device_id}] \
    --shape "[3,640,640]" \
    --batch_size 1 \
    --instance 6 \
    --iterations 600 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 80
    assert (
        float(qps) > qps_thresh
    ), f"yolov8_seg python:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    python3 ./samples/segmentation/yolov8_seg/yolov8_seg_prof.py \
    -m /opt/vastai/vaststreamx/data/models/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolov8seg_bgr888.json \
    --elf_file /opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc \
    --device_ids [{device_id}] \
    --shape "[3,640,640]" \
    --batch_size 1 \
    --instance 1 \
    --iterations 100 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 12
    assert (
        float(qps) > qps_thresh
    ), f"yolov8_seg python:best prof qps {qps} is smaller than {qps_thresh}"


############################# dataset test
@pytest.mark.slow
def test_yolov8seg_py_precision(device_id):
    os.system("mkdir -p yolov8_seg_out")
    cmd = f"""
    python3 ./samples/segmentation/yolov8_seg/yolov8_seg.py \
    -m /opt/vastai/vaststreamx/data/models/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolov8seg_bgr888.json \
    --device_id {device_id} \
    --elf_file /opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc \
    --threshold 0.01 \
    --label_file ./data/labels/coco2id.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder ./yolov8_seg_out
    """
    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/yolov8_seg/yolov8_seg_eval.py \
    --output_path yolov8_seg_out \
    --gt ./evaluation/yolov8_seg/instances_val2017.json
    """

    res = run_cmd(cmd)

    mAP_strs = re.findall(pattern=r"] = \d+.\d+", string=res)

    mAPs = []
    for map_str in mAP_strs:
        mAP = re.search(pattern=r"\d+.\d+", string=map_str).group()
        mAPs.append(mAP)


    det_mAP_50_95_thres = 0.47
    det_mAP_50_thres = 0.63

    seg_mAP_50_95_thres = 0.37
    seg_mAP_50_thres = 0.59

    assert len(mAPs) == 24, f"len(mAPs)={len(mAPs)} is not equal 24"

    assert (
        float(mAPs[0]) > det_mAP_50_95_thres
    ), f"yolov8_seg python: (mAPs[0] {mAPs[0]} is smaller than {det_mAP_50_95_thres}"

    assert (
        float(mAPs[1]) > det_mAP_50_thres
    ), f"yolov8_seg python: (mAPs[1] {mAPs[1]} is smaller than {det_mAP_50_thres}"

    assert (
        float(mAPs[12]) > seg_mAP_50_95_thres
    ), f"yolov8_seg python: (mAPs[12] {mAPs[12]} is smaller than {seg_mAP_50_95_thres}"

    assert (
        float(mAPs[13]) > seg_mAP_50_thres
    ), f"yolov8_seg python: (mAPs[13] {mAPs[13]} is smaller than {seg_mAP_50_thres}"

    os.system("rm -rf yolov8_seg_out yolov8seg_predictions.json")
