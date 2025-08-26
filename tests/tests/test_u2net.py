
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
def test_u2net_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/u2net \
    -m /opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod \
    --vdsp_params ./data/configs/u2net_bgr888.json \
    --device_id {device_id} \
    --input_file ./data/images/cat.jpg \
    --output_file u2net_result.png
    """
    res = run_cmd(cmd)

    assert os.path.exists("u2net_result.png"), "u2net:can't find u2net_result.png"

    os.system("rm u2net_result.png")

#################  performance test
@pytest.mark.fast
def test_u2net_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/u2net_prof \
    -m /opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod \
    --vdsp_params ./data/configs/u2net_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 1024 \
    --shape "[3,320,320]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 90
    assert float(qps) > qps_thresh, f"u2net:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/u2net_prof \
    -m /opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod \
    --vdsp_params ./data/configs/u2net_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 1024 \
    --shape "[3,320,320]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 65
    assert float(qps) > qps_thresh, f"u2net:best prof qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_u2net_cpp_precision(device_id):
    os.system("mkdir -p u2net_output")
    cmd = f"""
    ./build/vaststreamx-samples/bin/u2net \
    -m /opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod \
    --vdsp_params ./data/configs/u2net_bgr888.json \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/ECSSD/filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ECSSD/image/ \
    --dataset_output_folder ./u2net_output
    """
    res = run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/salient_object_detection/PySODEval/eval.py \
    --dataset-json ./evaluation/salient_object_detection/PySODEval/examples/config_dataset.json \
    --method-json ./evaluation/salient_object_detection/PySODEval/examples/config_method_u2net.json
    """

    res = run_cmd(cmd,False)

    accuracys = re.findall(pattern=r"\d+.\d+", string=res)

    mae = 0.036 # the smaller, the better
    maxF = 0.939 # the larger the better
    wfm = 0.900 # the larger, the better

    assert float(accuracys[-12]) <= mae, f"u2net: MAE {accuracys[-12]} is larger than {mae}"
    assert (float(accuracys[-11]) >= maxF), f"u2net: maxF {accuracys[-11]} is smaller than {maxF}"
    assert float(accuracys[-1]) >= wfm, f"u2net: wfm {accuracys[-1]} is smaller than {wfm}"

    os.system("rm -r u2net_output")

#####################  python test #####################


##############################  one image test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_u2net_py(device_id):
    cmd = f"""
    python3 ./samples/salient_object_detection/u2net/u2net.py \
    -m /opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod \
    --vdsp_params ./data/configs/u2net_bgr888.json \
    --device_id {device_id} \
    --input_file  ./data/images/cat.jpg \
    --output_file u2net_result.png
    """
    res = run_cmd(cmd)

    assert os.path.exists("u2net_result.png"), "u2net:can't find u2net_result.png"

    os.system("rm u2net_result.png")

#################  performance test
@pytest.mark.fast
def test_u2net_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/salient_object_detection/u2net/u2net_prof.py \
    -m /opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod \
    --vdsp_params ./data/configs/u2net_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 2 \
    --instance 1 \
    --iterations 1024 \
    --shape "[3,320,320]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 90
    assert float(qps) > qps_thresh, f"u2net:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    python3 ./samples/salient_object_detection/u2net/u2net_prof.py \
    -m /opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod \
    --vdsp_params ./data/configs/u2net_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 1024 \
    --shape "[3,320,320]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 70
    assert float(qps) > qps_thresh, f"u2net:best prof qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_u2net_py_precision(device_id):
    os.system("mkdir -p u2net_output")
    cmd = f"""
    python3 ./samples/salient_object_detection/u2net/u2net.py \
    -m /opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod \
    --vdsp_params ./data/configs/u2net_bgr888.json \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/ECSSD/filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ECSSD/image/ \
    --dataset_output_folder ./u2net_output
    """
    res = run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/salient_object_detection/PySODEval/eval.py \
    --dataset-json ./evaluation/salient_object_detection/PySODEval/examples/config_dataset.json \
    --method-json ./evaluation/salient_object_detection/PySODEval/examples/config_method_u2net.json
    """
    res = run_cmd(cmd,False)

    accuracys = re.findall(pattern=r"\d+.\d+", string=res)

    mae = 0.036 # the smaller, the better
    maxF = 0.939 # the larger the better
    wfm = 0.900 # the larger, the better

    assert float(accuracys[-12]) <= mae, f"u2net: MAE {accuracys[-12]} is larger than {mae}"
    assert (float(accuracys[-11]) >= maxF), f"u2net: maxF {accuracys[-11]} is smaller than {maxF}"
    assert float(accuracys[-1]) >= wfm, f"u2net: wfm {accuracys[-1]} is smaller than {wfm}"

    os.system("rm -r u2net_output")

