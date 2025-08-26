
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
def test_hih_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/hih \
    -m /opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod \
    --vdsp_params ./data/configs/hih_bgr888.json \
    --device_id {device_id} \
    --input_file ./data/images/face.jpg
    """
    res = run_cmd(cmd)
    res = res.split("Face alignment results:")[-1]

    landmarks = re.findall(pattern=r"\d+.\d+", string=res)

    x_gt = 0.0859375
    y_gt = 0.431641

    assert (
        float(landmarks[0]) > x_gt - 0.05
        and float(landmarks[0]) < x_gt + 0.05
        and float(landmarks[1]) > y_gt - 0.05
        and float(landmarks[1]) < y_gt + 0.05
    ), f"hih c++ :first landmark ({landmarks[0]},{landmarks[1]}) is different with ({x_gt},{y_gt})"


#################  performance test
@pytest.mark.fast
def test_hih_cpp_performance(device_id):
    # 最佳性能与时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/hih_prof \
    -m /opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod \
    --vdsp_params ./data/configs/hih_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 10 \
    --instance 1 \
    --iterations 1024 \
    --shape "[3,256,256]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 2
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 800
    assert float(qps) > qps_thresh, f"hih:best prof qps {qps} is smaller than {qps_thresh}"

    # 最佳时延与性能
    cmd = f"""
    ./build/vaststreamx-samples/bin/hih_prof \
    -m /opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod \
    --vdsp_params ./data/configs/hih_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 1024 \
    --shape "[3,256,256]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 185
    assert float(qps) > qps_thresh, f"hih:best prof qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_hih_cpp_precision(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/hih \
    -m /opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod \
    --vdsp_params ./data/configs/hih_bgr888.json \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/WFLW/test_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/WFLW/test/ \
    --dataset_output_file ./hih_output.txt
    """
    res = run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/face_alignment/eval_wflw.py \
    --result hih_output.txt \
    --gt /opt/vastai/vaststreamx/data/datasets/WFLW/test.txt
    """

    res = run_cmd(cmd,False)

    accuracys = re.findall(pattern=r"\d+.\d+", string=res)

    nme = 0.043
    failure_rate = 0.033  # @0.1
    auc = 0.59  # @0.1

    assert float(accuracys[0]) <= nme, f"hih: NMS {accuracys[0]} is larger than {nme}"
    assert (
        float(accuracys[2]) <= failure_rate
    ), f"hih: FR {accuracys[2]} is larger than {failure_rate}"
    assert float(accuracys[-1]) >= auc, f"hih: AUC {accuracys[-1]} is smaller than {auc}"

    os.system("rm hih_output.txt")

#####################  python test #####################


##############################  one image test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_hih_py(device_id):
    cmd = f"""
    python3 ./samples/face_alignment/hih/hih.py \
    -m /opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod \
    --vdsp_params ./data/configs/hih_bgr888.json \
    --device_id {device_id} \
    --input_file  ./data/images/face.jpg
    """
    res = run_cmd(cmd)

    res = res.split("Face alignment results:")[-1]

    landmarks = re.findall(pattern=r"\d+.\d+", string=res)

    x_gt = 0.0859375
    y_gt = 0.431641

    assert (
        float(landmarks[0]) > x_gt - 0.05
        and float(landmarks[0]) < x_gt + 0.05
        and float(landmarks[1]) > y_gt - 0.05
        and float(landmarks[1]) < y_gt + 0.05
    ), f"hih python:first landmark ({landmarks[0]},{landmarks[1]}) is different with ({x_gt},{y_gt})"



#################  performance test
@pytest.mark.fast
def test_hih_py_performance(device_id):
    # 最佳性能与时延
    cmd = f"""
    python3 ./samples/face_alignment/hih/hih_prof.py \
    -m /opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod \
    --vdsp_params ./data/configs/hih_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 10 \
    --instance 2 \
    --iterations 1024 \
    --shape "[3,256,256]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 800
    assert float(qps) > qps_thresh, f"hih:best prof qps {qps} is smaller than {qps_thresh}"

    # 最佳时延与性能
    cmd = f"""
    python3 ./samples/face_alignment/hih/hih_prof.py \
    -m /opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod \
    --vdsp_params ./data/configs/hih_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 1024 \
    --shape "[3,256,256]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 200
    assert float(qps) > qps_thresh, f"hih:best prof qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_hih_py_precision(device_id):
    cmd = f"""
    python3 ./samples/face_alignment/hih/hih.py \
    -m /opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod \
    --vdsp_params ./data/configs/hih_bgr888.json \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/WFLW/test_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/WFLW/test \
    --dataset_output_file hih_output.txt
    """

    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/face_alignment/eval_wflw.py \
    --result hih_output.txt \
    --gt /opt/vastai/vaststreamx/data/datasets/WFLW/test.txt
    """

    res = run_cmd(cmd,False)

    accuracys = re.findall(pattern=r"\d+.\d+", string=res)

    nme = 0.043
    failure_rate = 0.033  # @0.1
    auc = 0.59  # @0.1

    assert float(accuracys[0]) <= nme, f"hih: NMS {accuracys[0]} is larger than {nme}"
    assert (
        float(accuracys[2]) <= failure_rate
    ), f"hih: FR {accuracys[2]} is larger than {failure_rate}"
    assert float(accuracys[-1]) >= auc, f"hih: AUC {accuracys[-1]} is smaller than {auc}"

    os.system("rm hih_output.txt")
