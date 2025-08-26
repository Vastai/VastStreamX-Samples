
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
def test_classification_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/classification \
    -m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
    --vdsp_params ./data/configs/resnet_rgbplanar.json \
    --device_id {device_id} \
    --label_file ./data/labels/imagenet.txt \
    --input_file ./data/images/cat.jpg
    """
    res = run_cmd(cmd)

    scores = re.findall(pattern=r"\d+.\d+", string=res)

    score_gt = 0.7
    assert (
        float(scores[0]) > score_gt
    ), f"classification  c++:max score {scores[0]} is smaller than {score_gt}"


#################  performance test

@pytest.mark.fast
def test_classification_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/cls_prof \
    -m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
    --vdsp_params ./data/configs/resnet_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 8 \
    --instance 1 \
    --iterations 2000 \
    --shape [3,256,256] \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 3100
    assert (
        float(qps) > qps_thresh
    ), f"classification c++:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/cls_prof \
    -m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
    --vdsp_params ./data/configs/resnet_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 4000 \
    --shape [3,256,256] \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 830
    assert (
        float(qps) > qps_thresh
    ), f"classification c++:best latancy qps {qps} is smaller than {qps_thresh}"



############################# dataset test

@pytest.mark.fast
def test_classification_cpp_precision(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/classification \
    -m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
    --vdsp_params ./data/configs/resnet_rgbplanar.json \
    --device_id {device_id} \
    --label_file ./data/labels/imagenet.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/ILSVRC2012_img_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_file cls_result.txt
    """

    run_cmd(cmd)


    cmd = f"""
    python3 ./evaluation/classification/eval_topk.py  cls_result.txt 
    """

    res = run_cmd(cmd)

    accuracys = re.findall(pattern=r"\d+.\d+", string=res)

    top1_rate = 75.6
    top5_rate = 92.5

    assert (
        float(accuracys[0]) > top1_rate
    ), f"classification c++:top1_rate {accuracys[0]} is smaller than {top1_rate}"
    assert (
        float(accuracys[1]) > top5_rate
    ), f"classification c++:top5_rate {accuracys[1]} is smaller than {top5_rate}"


    os.system("rm  cls_result.txt")


#####################  python test #####################


##############################  one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_classification_py(device_id):
    cmd = f"""
    python3 ./samples/classification/resnet/classification.py \
    -m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
    --vdsp_params ./data/configs/resnet_rgbplanar.json \
    --device_id {device_id} \
    --label_file ./data/labels/imagenet.txt \
    --input_file ./data/images/cat.jpg
    """
    res = run_cmd(cmd)

    scores = re.findall(pattern=r"\d+.\d+", string=res)

    score_gt = 0.7
    assert (
        float(scores[0]) > score_gt
    ), f"classification:max score {scores[0]} is smaller than {score_gt}"



#################  performance test

@pytest.mark.fast
def test_classification_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/classification/resnet/cls_prof.py \
    -m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
    --vdsp_params ./data/configs/resnet_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 8 \
    --instance 1 \
    --iterations 2000 \
    --shape [3,256,256] \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 3050
    assert (
        float(qps) > qps_thresh
    ), f"classification:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    python3 ./samples/classification/resnet/cls_prof.py \
    -m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
    --vdsp_params ./data/configs/resnet_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 4000 \
    --shape [3,256,256] \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 830
    assert (
        float(qps) > qps_thresh
    ), f"classification:best latancy qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_classification_py_precision(device_id):
    cmd = f"""
    python3 ./samples/classification/resnet/classification.py \
    -m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
    --vdsp_params ./data/configs/resnet_rgbplanar.json \
    --device_id {device_id} \
    --label_file ./data/labels/imagenet.txt \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/ILSVRC2012_img_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_file cls_result.txt
    """

    run_cmd(cmd)


    cmd = f"""
    python3 ./evaluation/classification/eval_topk.py  cls_result.txt 
    """

    res = run_cmd(cmd)

    accuracys = re.findall(pattern=r"\d+.\d+", string=res)

    top1_rate = 75.6
    top5_rate = 92.5

    assert (
        float(accuracys[0]) > top1_rate
    ), f"classification:top1_rate {accuracys[0]} is smaller than {top1_rate}"
    assert (
        float(accuracys[1]) > top5_rate
    ), f"classification:top5_rate {accuracys[1]} is smaller than {top5_rate}"


    os.system("rm  cls_result.txt")
