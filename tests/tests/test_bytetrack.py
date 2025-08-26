
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
def test_bytetracker_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/bytetracker \
    -m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
    --vdsp_params ./data/configs/bytetrack_rgbplanar.json \
    --device_id {device_id} \
    --det_threshold 0.001 \
    --label_file ./data/labels/coco2id.txt \
    --input_file /opt/vastai/vaststreamx/data/datasets/mot17/test/MOT17-02-FRCNN/img1/000001.jpg \
    --output_file ./bytetrack_result.jpg 
    """


    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.65

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(
                f"bytetrack c++: score {score} is smaller than {score_thres}"
            )

    assert (
        len(scores) >= 19
    ), f"bytetrack c++: detected object count={len(scores)} is not equal or larger than 19"

    assert os.path.exists(
        "bytetrack_result.jpg"
    ), "bytetrack c++:can't find bytetrack_result.jpg"



    os.system("rm bytetrack_result.jpg")


#################  performance test
@pytest.mark.fast
def test_bytetracker_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/bytetracker_prof \
    -m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
    --vdsp_params ./data/configs/bytetrack_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 3 \
    --shape "[3,800,1440]" \
    --percentiles "[50,90,95,99]" \
    --iterations 800 \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 130
    assert (
        float(qps) > qps_thresh
    ), f"bytetrack c++: best prof qps {qps} is smaller than {qps_thresh}"


    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/bytetracker_prof \
    -m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
    --vdsp_params ./data/configs/bytetrack_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --shape "[3,800,1440]" \
    --percentiles "[50,90,95,99]" \
    --iterations 800 \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 95
    assert (
        float(qps) > qps_thresh
    ), f"bytetrack c++: best latency qps {qps} is smaller than {qps_thresh}"


############################# dataset test
@pytest.mark.fast
def test_bytetracker_cpp_precision(device_id):
    os.system("mkdir -p mot_output")
    cmd = f"""
    ./build/vaststreamx-samples/bin/bytetracker \
    -m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
    --vdsp_params ./data/configs/bytetrack_rgbplanar.json \
    --device_id {device_id} \
    --det_threshold 0.001 \
    --label_file ./data/labels/coco2id.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/mot17/MOT17-02-FRCNN-filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/mot17/ \
    --dataset_result_file ./mot_output/MOT17-02-FRCNN.txt 
    """
    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/mot/mot_eval.py  \
    -gt /opt/vastai/vaststreamx/data/datasets/mot17/test \
    -r ./mot_output
    """

    res = run_cmd(cmd, check_stderr=False)


    prec_str = re.findall(pattern=r"FRCNN \d+.\d+", string=res)


    precisions = []
    for prec in prec_str:
        precision = re.search(pattern=r"\d+.\d+", string=prec).group()
        precisions.append(precision)

    assert (
        len(precisions) == 2
    ), "bytetrack c++: dataset test error, len(precision) is not 2"

    rcll_thresh = 79.7 
    idf1_thres = 58.0 
    assert (
        float(precisions[0]) > rcll_thresh
    ), f"bytetrack c++: dataset test error, Rcall {float(precisions[0])} is smaller than {rcll_thresh}"
    assert (
        float(precisions[1]) > idf1_thres
    ), f"bytetrack c++: dataset test error, IDF1 {float(precisions[1])} is smaller than {idf1_thres}"


    os.system("rm -rf mot_output")


######################  python test #####################

################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_bytetracker_py(device_id):
    cmd = f"""
    python3 ./samples/multi_object_tracking/bytetracker.py \
    -m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
    --vdsp_params ./data/configs/bytetrack_rgbplanar.json \
    --device_id {device_id} \
    --label_file ./data/labels/coco2id.txt \
    --detect_threshold 0.001 \
    --track_thresh 0.6 \
    --track_buffer 30 \
    --match_thresh 0.9 \
    --min_box_area 100 \
    --input_file /opt/vastai/vaststreamx/data/datasets/mot17/test/MOT17-02-FRCNN/img1/000001.jpg \
    --output_file bytetrack_result.jpg
    """

    res = run_cmd(cmd, check_stderr=False)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.65

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(
                f"bytetrack python: score {score} is smaller than {score_thres}"
            )

    assert (
        len(scores) >= 19
    ), f"bytetrack python: detected object count={len(scores)} is not equal or larger than 19"

    assert os.path.exists(
        "bytetrack_result.jpg"
    ), "bytetrack python:can't find bytetrack_result.jpg"



    os.system("rm bytetrack_result.jpg")


#################  performance test
@pytest.mark.fast
def test_bytetracker_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/multi_object_tracking/bytetracker_prof.py \
    -m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
    --vdsp_params ./data/configs/bytetrack_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --shape "[3,800,1440]" \
    --percentiles "[50,90,95,99]" \
    --iterations 800 \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd, check_stderr=False)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 130
    assert (
        float(qps) > qps_thresh
    ), f"bytetrack python: best prof qps {qps} is smaller than {qps_thresh}"


    # 测试最小时延
    cmd = f"""
    python3 ./samples/multi_object_tracking/bytetracker_prof.py \
    -m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
    --vdsp_params ./data/configs/bytetrack_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --shape "[3,800,1440]" \
    --percentiles "[50,90,95,99]" \
    --iterations 800 \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 95
    assert (
        float(qps) > qps_thresh
    ), f"bytetrack python: best prof qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_bytetracker_py_precision(device_id):
    os.system("mkdir -p mot_output")
    cmd = f"""
    python3 ./samples/multi_object_tracking/bytetracker.py \
    -m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
    --vdsp_params ./data/configs/bytetrack_rgbplanar.json \
    --device_id {device_id} \
    --label_file ./data/labels/coco2id.txt \
    --detect_threshold 0.001 \
    --track_thresh 0.6 \
    --track_buffer 30 \
    --match_thresh 0.9 \
    --min_box_area 100 \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/mot17/MOT17-02-FRCNN-filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/mot17/ \
    --dataset_result_file ./mot_output/MOT17-02-FRCNN.txt 
    """
    run_cmd(cmd, check_stderr=False)

    cmd = f"""
    python3 ./evaluation/mot/mot_eval.py  \
    -gt /opt/vastai/vaststreamx/data/datasets/mot17/test \
    -r ./mot_output
    """

    res = run_cmd(cmd, check_stderr=False)

    prec_str = re.findall(pattern=r"FRCNN \d+.\d+", string=res)

    precisions = []
    for prec in prec_str:
        precision = re.search(pattern=r"\d+.\d+", string=prec).group()
        precisions.append(precision)

    assert (
        len(precisions) == 2
    ), "bytetrack python: dataset test error, len(precision) is not 2"

    rcll_thresh = 79.7 
    idf1_thres = 58.0 
    assert (
        float(precisions[0]) > rcll_thresh
    ), f"bytetrack python: dataset test error, Rcall {float(precisions[0])} is smaller than {rcll_thresh}"
    assert (
        float(precisions[1]) > idf1_thres
    ), f"bytetrack python: dataset test error, IDF1 {float(precisions[1])} is smaller than {idf1_thres}"

    os.system("rm -rf mot_output")
