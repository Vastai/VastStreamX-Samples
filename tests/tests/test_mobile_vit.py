
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
def test_mobilevit_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/mobilevit \
    -m /opt/vastai/vaststreamx/data/models/mobilevit_s-fp16-none-1_3_224_224-vacc/mod \
    --vdsp_params ./data/configs/mobilevit_rgbplanar.json \
    --device_id {device_id} \
    --label_file ./data/labels/imagenet.txt \
    --input_file ./data/images/cat.jpg 
    """

    res = run_cmd(cmd)

    scores = re.findall(pattern=r"\d+.\d+", string=res)

    score_gt = 2.0
    assert (
        float(scores[0]) > score_gt
    ), f"mobile-vit c++:max score {scores[0]} is smaller than {score_gt}"


#################  performance test

@pytest.mark.fast
def test_mobilevit_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/mobilevit_prof \
    -m /opt/vastai/vaststreamx/data/models/mobilevit_s-fp16-none-1_3_224_224-vacc/mod \
    --vdsp_params ./data/configs/mobilevit_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 300 \
    --shape "[3,224,224]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 40
    assert (
        float(qps) > qps_thresh
    ), f"mobile-vit c++:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/mobilevit_prof \
    -m /opt/vastai/vaststreamx/data/models/mobilevit_s-fp16-none-1_3_224_224-vacc/mod \
    --vdsp_params ./data/configs/mobilevit_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 300 \
    --shape "[3,224,224]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 40
    assert (
        float(qps) > qps_thresh
    ), f"mobile-vit c++:best latancy qps {qps} is smaller than {qps_thresh}"



############################# dataset test

@pytest.mark.slow
def test_mobilevit_cpp_precision(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/mobilevit \
    -m /opt/vastai/vaststreamx/data/models/mobilevit_s-fp16-none-1_3_224_224-vacc/mod \
    --vdsp_params ./data/configs/mobilevit_rgbplanar.json \
    --device_id {device_id} \
    --label_file ./data/labels/imagenet.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/ILSVRC2012_img_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_file mobilevit_result.txt
    """

    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/classification/eval_topk.py mobilevit_result.txt  
    """

    res = run_cmd(cmd)

    accuracys = re.findall(pattern=r"\d+.\d+", string=res)

    top1_rate = 63.0
    top5_rate = 85.0

    assert (
        float(accuracys[0]) > top1_rate
    ), f"mobile-vit c++:top1_rate {accuracys[0]} is smaller than {top1_rate}"
    assert (
        float(accuracys[1]) > top5_rate
    ), f"mobile-vit c++:top5_rate {accuracys[1]} is smaller than {top5_rate}"

    os.system("rm  mobilevit_result.txt")


#####################  python test #####################


##############################  one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_mobilevit_py(device_id):
    cmd = f"""
    python3 ./samples/classification/mobilevit/mobilevit.py \
    -m /opt/vastai/vaststreamx/data/models/mobilevit_s-fp16-none-1_3_224_224-vacc/mod \
    --vdsp_params ./data/configs/mobilevit_rgbplanar.json \
    --device_id {device_id} \
    --label_file ./data/labels/imagenet.txt \
    --input_file ./data/images/cat.jpg 
    """
    res = run_cmd(cmd)

    scores = re.findall(pattern=r"\d+.\d+", string=res)

    score_gt = 2.0
    assert (
        float(scores[0]) > score_gt
    ), f"mobile-vit python:max score {scores[0]} is smaller than {score_gt}"


#################  performance test
@pytest.mark.fast
def test_mobilevit_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/classification/mobilevit/mobilevit_prof.py \
    -m /opt/vastai/vaststreamx/data/models/mobilevit_s-fp16-none-1_3_224_224-vacc/mod \
    --vdsp_params ./data/configs/mobilevit_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 300 \
    --shape "[3,224,224]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 40
    assert (
        float(qps) > qps_thresh
    ), f"mobile-vit python:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    python3 ./samples/classification/mobilevit/mobilevit_prof.py \
    -m /opt/vastai/vaststreamx/data/models/mobilevit_s-fp16-none-1_3_224_224-vacc/mod \
    --vdsp_params ./data/configs/mobilevit_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 300 \
    --shape "[3,224,224]" \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 40
    assert (
        float(qps) > qps_thresh
    ), f"mobile-vit python:best latancy qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.slow
def test_mobilevit_py_precision(device_id):
    cmd = f"""
    python3 ./samples/classification/mobilevit/mobilevit.py \
    -m /opt/vastai/vaststreamx/data/models/mobilevit_s-fp16-none-1_3_224_224-vacc/mod \
    --vdsp_params ./data/configs/mobilevit_rgbplanar.json \
    --device_id {device_id} \
    --label_file ./data/labels/imagenet.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/ILSVRC2012_img_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_file mobilevit_result.txt
    """

    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/classification/eval_topk.py mobilevit_result.txt  
    """

    res = run_cmd(cmd)

    accuracys = re.findall(pattern=r"\d+.\d+", string=res)

    top1_rate = 63.0
    top5_rate = 85.0

    assert (
        float(accuracys[0]) > top1_rate
    ), f"mobile-vit python:top1_rate {accuracys[0]} is smaller than {top1_rate}"
    assert (
        float(accuracys[1]) > top5_rate
    ), f"mobile-vit python:top5_rate {accuracys[1]} is smaller than {top5_rate}"

    os.system("rm  mobilevit_result.txt")
