
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
def test_retinaface_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/face_detection \
    -m /opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/retinaface_rgbplanar.json \
    --device_id {device_id} \
    --threshold 0.5 \
    --input_file ./data/images/face.jpg \
    --output_file face_det_result.jpg
    """
    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.98

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(f"retinaface:score {score} is smaller than {score_thres}")

    assert len(scores) == 1, f"retinaface:detected object count={len(scores)} is not 3"
    assert os.path.exists(
        "face_det_result.jpg"
    ), "retinaface:can't find face_det_result.jpg"

    os.system("rm face_det_result.jpg")

#################  performance test

@pytest.mark.fast
def test_retinaface_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/face_detection_prof \
    -m /opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/retinaface_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size  2 \
    --instance  1 \
    --shape [3,640,640] \
    --iterations 1500 \
    --percentiles [50,90,95,99] \
    --threshold 0.01 \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 430
    assert (
        float(qps) > qps_thresh
    ), f"retinaface:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/face_detection_prof \
    -m /opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/retinaface_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance  1 \
    --shape [3,640,640] \
    --iterations 1500 \
    --percentiles [50,90,95,99] \
    --threshold 0.01 \
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
    ), f"retinaface:best latancy qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_retinaface_cpp_precision(device_id):
    os.system("mkdir -p facedet_out")
    cmd = f"""
    ./build/vaststreamx-samples/bin/face_detection \
    -m /opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/retinaface_rgbplanar.json \
    --device_id {device_id} \
    --threshold 0.001 \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/widerface_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder facedet_out
    """
    run_cmd(cmd)

    os.system(
        """
        cd ./evaluation/face_detection
        python3 setup.py build_ext --inplace
        """
    )

    cmd = f"""
    python3 ./evaluation/face_detection/evaluation.py \
    -g ./evaluation/face_detection/ground_truth \
    -p ./facedet_out
    """
    res = run_cmd(cmd, check_stderr=False)

    aps_str = re.findall(pattern=r"AP: \d+.\d+", string=res)
    aps = []
    for ap in aps_str:
        aps.append(re.search(pattern=r"\d+.\d+", string=ap).group())

    assert len(aps) == 3, f"retinaface: len of aps {len(aps)} != 3"
    easy_ap = aps[0]
    medium_ap = aps[1]
    hard_ap = aps[2]

    easy_ap_thres = 0.93
    medium_ap_thres = 0.899
    hard_ap_thres = 0.62

    assert (
        float(easy_ap) > easy_ap_thres
    ), f"retinaface: easy_ap {easy_ap} is smaller than {easy_ap_thres}"
    assert (
        float(medium_ap) > medium_ap_thres
    ), f"retinaface: medium_ap {medium_ap} is smaller than {medium_ap_thres}"
    assert (
        float(hard_ap) > hard_ap_thres
    ), f"retinaface: hard_ap {hard_ap} is smaller than {hard_ap_thres}"

    os.system("rm -rf facedet_out")


######################  python test #####################

################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_retinaface_py(device_id):
    cmd = f"""
    python3 ./samples/face_detection/face_detection.py \
    -m /opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/retinaface_rgbplanar.json \
    --device_id {device_id} \
    --threshold 0.5 \
    --input_file ./data/images/face.jpg \
    --output_file face_det_result.jpg
    """
    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.98

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(f"retinaface:score {score} is smaller than {score_thres}")

    assert len(scores) == 1, f"retinaface: detected object count={len(scores)} is not 3"
    assert os.path.exists(
        "face_det_result.jpg"
    ), "retinaface: can't find face_det_result.jpg"

    os.system("rm face_det_result.jpg")

#################  performance test

@pytest.mark.fast
def test_retinaface_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/face_detection/face_detection_prof.py \
    -m /opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/retinaface_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size  2 \
    --instance  1 \
    --shape [3,640,640] \
    --iterations 1500 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 430
    assert (
        float(qps) > qps_thresh
    ), f"retinaface:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    python3 ./samples/face_detection/face_detection_prof.py \
    -m /opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/retinaface_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size  1 \
    --instance  1 \
    --shape [3,640,640] \
    --iterations 1500 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 255
    assert (
        float(qps) > qps_thresh
    ), f"retinaface:best latancy qps {qps} is smaller than {qps_thresh}"


############################# dataset test
@pytest.mark.fast
def test_retinaface_py_precision(device_id):
    os.system("mkdir -p facedet_out")
    cmd = f"""
    python3 ./samples/face_detection/face_detection.py \
    -m /opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/retinaface_rgbplanar.json \
    --device_id {device_id} \
    --threshold 0.001 \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/widerface_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder facedet_out
    """
    run_cmd(cmd)

    # os.system("""
    # cd ./evaluation/face_detection
    # python3 setup.py build_ext --inplace
    # """)

    cmd = f"""
    python3 ./evaluation/face_detection/evaluation.py \
    -g ./evaluation/face_detection/ground_truth \
    -p ./facedet_out
    """
    res = run_cmd(cmd, check_stderr=False)

    aps_str = re.findall(pattern=r"AP: \d+.\d+", string=res)
    aps = []
    for ap in aps_str:
        aps.append(re.search(pattern=r"\d+.\d+", string=ap).group())

    assert len(aps) == 3, f"retinaface: len of aps {len(aps)} != 3"
    easy_ap = aps[0]
    medium_ap = aps[1]
    hard_ap = aps[2]

    easy_ap_thres = 0.93
    medium_ap_thres = 0.899
    hard_ap_thres = 0.62

    assert (
        float(easy_ap) > easy_ap_thres
    ), f"retinaface: easy_ap {easy_ap} is smaller than {easy_ap_thres}"
    assert (
        float(medium_ap) > medium_ap_thres
    ), f"retinaface: medium_ap {medium_ap} is smaller than {medium_ap_thres}"
    assert (
        float(hard_ap) > hard_ap_thres
    ), f"retinaface: hard_ap {hard_ap} is smaller than {hard_ap_thres}"

    os.system("rm -rf facedet_out")
