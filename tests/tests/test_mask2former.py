
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
def test_mask2former_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/mask2former  \
    --model_prefix /opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod \
    --vdsp_params ./data/configs/mask2former_rgbplanar.json \
    --device_id {device_id} \
    --threshold 0.6 \
    --label_file ./data/labels/coco2id.txt \
    --input_file ./data/images/cycling.jpg \
    --output_file mask2former_result.jpg
    """
    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.8

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(
                f"mask2former c++:score {score} is smaller than {score_thres}"
            )

    assert len(scores) == 6, f"mask2former c++:detected object count={len(scores)} is not 6"
    assert os.path.exists(
        "mask2former_result.jpg"
    ), "mask2former c++:can't mask2former_result.jpg"
    assert (
        re.findall(pattern=r"bicycle", string=res) != []
    ), "mask2former c++:can't find bicycle in result string"
    assert (
        re.findall(pattern=r"person", string=res) != []
    ), "mask2former c++:can't find dog in result string"

    os.system("rm mask2former_result.jpg")



#################  performance test
@pytest.mark.fast
def test_mask2former_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/mask2former_prof \
    --model_prefix /opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod \
    --vdsp_params ./data/configs/mask2former_rgbplanar.json \
    --device_ids [{device_id}] \
    --shape "[3,1024,1024]" \
    --batch_size 1 \
    --instance 1 \
    --iterations 10 \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 0.34
    assert (
        float(qps) > qps_thresh
    ), f"mask2former c++:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/mask2former_prof \
    --model_prefix /opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod \
    --vdsp_params ./data/configs/mask2former_rgbplanar.json \
    --device_ids [{device_id}] \
    --shape "[3,1024,1024]" \
    --batch_size 1 \
    --instance 1 \
    --iterations 10 \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 0.34
    assert (
        float(qps) > qps_thresh
    ), f"mask2former c++:best latency qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.slow
def test_mask2former_cpp_precision(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/mask2former \
    --model_prefix /opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod \
    --vdsp_params ./data/configs/mask2former_rgbplanar.json \
    --device_id {device_id} \
    --threshold 0.001 \
    --label_file  ./data/labels/coco2id.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_file mask2former_predictions.json
    """

    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/coco_seg/coco_seg_eval.py \
    --prediction_file mask2former_predictions.json \
    --gt ./evaluation/coco_seg/instances_val2017.json
    """

    res = run_cmd(cmd)

    mAP_strs = re.findall(pattern=r"] = \d+.\d+", string=res)

    mAPs = []
    for map_str in mAP_strs:
        mAP = re.search(pattern=r"\d+.\d+", string=map_str).group()
        mAPs.append(mAP)

    det_mAP_50_95_thres = 0.42
    det_mAP_50_thres = 0.625

    seg_mAP_50_95_thres = 0.41
    seg_mAP_50_thres = 0.63

    assert len(mAPs) == 24, f"len(mAPs)={len(mAPs)} is not equal 24"
    assert (
        float(mAPs[0]) > det_mAP_50_95_thres
    ), f"mask2former c++: (mAPs[0] {mAPs[0]} is smaller than {det_mAP_50_95_thres}"

    assert (
        float(mAPs[1]) > det_mAP_50_thres
    ), f"mask2former c++: (mAPs[1] {mAPs[1]} is smaller than {det_mAP_50_thres}"

    assert (
        float(mAPs[12]) > seg_mAP_50_95_thres
    ), f"mask2former c++: (mAPs[12] {mAPs[12]} is smaller than {seg_mAP_50_95_thres}"

    assert (
        float(mAPs[13]) > seg_mAP_50_thres
    ), f"mask2former c++: (mAPs[13] {mAPs[13]} is smaller than {seg_mAP_50_thres}"

    os.system("rm -rf mask2former_predictions.json")


# #####################  python test #####################

################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_mask2former_py(device_id):
    cmd = f"""
    python3 ./samples/segmentation/mask2former/mask2former.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod \
    --vdsp_params ./data/configs/mask2former_rgbplanar.json \
    --device_id {device_id} \
    --threshold 0.6 \
    --label_file ./data/labels/coco2id.txt \
    --input_file ./data/images/cycling.jpg \
    --output_file mask2former_result.jpg
    """
    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.8

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(
                f"mask2former python:score {score} is smaller than {score_thres}"
            )

    assert len(scores) == 6, f"mask2former python:detected object count={len(scores)} is not 6"
    assert os.path.exists(
        "mask2former_result.jpg"
    ), "mask2former python:can't mask2former_result.jpg"
    assert (
        re.findall(pattern=r"bicycle", string=res) != []
    ), "mask2former python:can't find bicycle in result string"
    assert (
        re.findall(pattern=r"person", string=res) != []
    ), "mask2former python:can't find dog in result string"

    os.system("rm mask2former_result.jpg")


#################  performance test
@pytest.mark.fast
def test_mask2former_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/segmentation/mask2former/mask2former_prof.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod \
    --vdsp_params ./data/configs/mask2former_rgbplanar.json \
    --device_ids [{device_id}] \
    --shape "[3,1024,1024]" \
    --batch_size 1 \
    --instance 1 \
    --iterations 10 \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 0.34
    assert (
        float(qps) > qps_thresh
    ), f"mask2former python:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    python3 ./samples/segmentation/mask2former/mask2former_prof.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod \
    --vdsp_params ./data/configs/mask2former_rgbplanar.json \
    --device_ids [{device_id}] \
    --shape "[3,1024,1024]" \
    --batch_size 1 \
    --instance 1 \
    --iterations 10 \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 0.31
    assert (
        float(qps) > qps_thresh
    ), f"mask2former python:best latency qps {qps} is smaller than {qps_thresh}"







############################# dataset test
@pytest.mark.slow
def test_mask2former_py_precision(device_id):
    cmd = f"""
    python3 ./samples/segmentation/mask2former/mask2former.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod \
    --vdsp_params ./data/configs/mask2former_rgbplanar.json \
    --device_id {device_id} \
    --threshold 0.001 \
    --label_file  ./data/labels/coco2id.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_file mask2former_predictions.json
    """

    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/coco_seg/coco_seg_eval.py \
    --prediction_file mask2former_predictions.json \
    --gt ./evaluation/coco_seg/instances_val2017.json
    """

    res = run_cmd(cmd)

    mAP_strs = re.findall(pattern=r"] = \d+.\d+", string=res)

    mAPs = []
    for map_str in mAP_strs:
        mAP = re.search(pattern=r"\d+.\d+", string=map_str).group()
        mAPs.append(mAP)

    det_mAP_50_95_thres = 0.42
    det_mAP_50_thres = 0.625

    seg_mAP_50_95_thres = 0.41
    seg_mAP_50_thres = 0.63

    assert len(mAPs) == 24, f"len(mAPs)={len(mAPs)} is not equal 24"
    assert (
        float(mAPs[0]) > det_mAP_50_95_thres
    ), f"mask2former python: (mAPs[0] {mAPs[0]} is smaller than {det_mAP_50_95_thres}"

    assert (
        float(mAPs[1]) > det_mAP_50_thres
    ), f"mask2former python: (mAPs[1] {mAPs[1]} is smaller than {det_mAP_50_thres}"

    assert (
        float(mAPs[12]) > seg_mAP_50_95_thres
    ), f"mask2former python: (mAPs[12] {mAPs[12]} is smaller than {seg_mAP_50_95_thres}"

    assert (
        float(mAPs[13]) > seg_mAP_50_thres
    ), f"mask2former python: (mAPs[13] {mAPs[13]} is smaller than {seg_mAP_50_thres}"

    os.system("rm -rf mask2former_predictions.json")





