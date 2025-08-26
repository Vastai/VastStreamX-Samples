
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
def test_crnn_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/crnn \
    -m /opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --label_file ./data/labels/key_37.txt \
    --device_id {device_id} \
    --input_file ./data/images/word_336.png 
    """

    res = run_cmd(cmd)

    scores = re.findall(pattern=r"\d+.\d+", string=res)

    score_gt = 0.98
    assert (
        float(scores[0]) > score_gt
    ), f"crnn: score {scores[0]} is smaller than {score_gt}"


    text = re.findall(pattern=r"super", string=res)

    assert text != [], f'crnn:can\'t find "super" in result string'


#################  performance test

@pytest.mark.fast
def test_crnn_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/crnn_prof \
    -m /opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_ids [{device_id}] \
    --label_file ./data/labels/key_37.txt \
    --batch_size 8 \
    --instance 1 \
    --shape [3,32,100] \
    --iterations 200 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 260
    assert float(qps) > qps_thresh, f"crnn:best prof qps {qps} is smaller than {qps_thresh}"


    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/crnn_prof \
    -m /opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_ids [{device_id}] \
    --label_file ./data/labels/key_37.txt \
    --batch_size 1 \
    --instance 1 \
    --shape [3,32,100] \
    --iterations 500 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 110
    assert (
        float(qps) > qps_thresh
    ), f"crnn:best latancy qps {qps} is smaller than {qps_thresh}"




############################# dataset test

@pytest.mark.fast
def test_crnn_cpp_precision(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/crnn \
    -m /opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_id {device_id} \
    --label_file ./data/labels/key_37.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/CUTE80/CUTE80_img_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/CUTE80 \
    --dataset_output_file cute80_pred.txt
    """
    run_cmd(cmd)


    cmd = f"""
    python3 ./evaluation/crnn/crnn_eval.py \
    --gt_file /opt/vastai/vaststreamx/data/datasets/CUTE80/CUTE80_gt.txt \
    --output_file cute80_pred.txt
    """

    res = run_cmd(cmd)

    acc = re.search(
        pattern=r"\d+.\d+", string=(re.search(pattern=r"acc = \d+.\d+", string=res).group())
    ).group()

    acc_thresh = 0.65

    assert float(acc) > acc_thresh, f"crnn:acc {acc} is smaller than {acc_thresh}"


    os.system("rm  cute80_pred.txt")

#####################  python test #####################


##############################  one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_crnn_py(device_id):
    cmd = f"""
    python3 ./samples/ocr/crnn.py \
    -m /opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --label_file ./data/labels/key_37.txt \
    --device_id {device_id} \
    --input_file ./data/images/word_336.png 
    """
    res = run_cmd(cmd)

    scores = re.findall(pattern=r"\d+.\d+", string=res)

    score_gt = 0.98
    assert (
        float(scores[0]) > score_gt
    ), f"crnn: score {scores[0]} is smaller than {score_gt}"


    text = re.findall(pattern=r"super", string=res)

    assert text != [], f'crnn:can\'t find "super" in result string'



#################  performance test

@pytest.mark.fast
def test_crnn_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/ocr/crnn_prof.py \
    -m /opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_ids [{device_id}]  \
    --label_file ./data/labels/key_37.txt \
    --batch_size 8 \
    --instance 1 \
    --shape [3,32,100] \
    --iterations 200 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 260
    assert float(qps) > qps_thresh, f"crnn:best prof qps {qps} is smaller than {qps_thresh}"


    # 测试最小时延
    cmd = f"""
    python3 ./samples/ocr/crnn_prof.py \
    -m /opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_ids [{device_id}]  \
    --label_file ./data/labels/key_37.txt \
    --batch_size 1 \
    --instance 1 \
    --shape [3,32,100] \
    --iterations 500 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 110
    assert (
        float(qps) > qps_thresh
    ), f"crnn:best latancy qps {qps} is smaller than {qps_thresh}"




############################# dataset test

@pytest.mark.fast
def test_crnn_py_precision(device_id):
    cmd = f"""
    python3 ./samples/ocr/crnn.py \
    -m /opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_id {device_id} \
    --label_file ./data/labels/key_37.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/CUTE80/CUTE80_img_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/CUTE80 \
    --dataset_output_file cute80_pred.txt
    """
    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/crnn/crnn_eval.py \
    --gt_file /opt/vastai/vaststreamx/data/datasets/CUTE80/CUTE80_gt.txt \
    --output_file cute80_pred.txt
    """

    res = run_cmd(cmd)

    acc = re.search(
        pattern=r"\d+.\d+", string=(re.search(pattern=r"acc = \d+.\d+", string=res).group())
    ).group()

    acc_thresh = 0.65

    assert float(acc) > acc_thresh, f"crnn:acc {acc} is smaller than {acc_thresh}"


    os.system("rm  cute80_pred.txt")
