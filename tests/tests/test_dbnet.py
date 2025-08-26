
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
def test_dbnet_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/dbnet \
    -m /opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod \
    --vdsp_params ./data/configs/dbnet_rgbplanar.json \
    --device_id {device_id} \
    --threshold 0.3 \
    --box_unclip_ratio 1.5 \
    --use_polygon_score 0 \
    --elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
    --input_file ./data/images/detect.jpg \
    --output_file dbnet_result.jpg
    """

    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score:\d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.65

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(f"dbnet:score {score} is smaller than {score_thres}")

    assert len(scores) == 14, f"dbnet:detected object count={len(scores)} is not 14"
    assert os.path.exists("dbnet_result.jpg"), "dbnet:can't find dbnet_result.jpg"

    os.system("rm dbnet_result.jpg")


#################  performance test

@pytest.mark.fast
def test_dbnet_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/dbnet_prof \
    -m /opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod \
    --vdsp_params ./data/configs/dbnet_rgbplanar.json \
    --device_ids [{device_id}] \
    --iterations 1024 \
    --instance 1 \
    --batch_size 1 \
    --shape [3,736,1280] \
    --elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 90
    assert (
        float(qps) > qps_thresh
    ), f"dbnet:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/dbnet_prof \
    -m /opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod \
    --vdsp_params ./data/configs/dbnet_rgbplanar.json \
    --device_ids [{device_id}] \
    --iterations 1024 \
    --instance 1 \
    --batch_size 1 \
    --shape [3,736,1280] \
    --elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 50
    assert (
        float(qps) > qps_thresh
    ), f"dbnet:best latancy qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_dbnet_cpp_precision(device_id):
    os.system("mkdir -p dbnet_output")
    cmd = f"""
    ./build/vaststreamx-samples/bin/dbnet \
    -m /opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod \
    --vdsp_params ./data/configs/dbnet_rgbplanar.json \
    --device_id {device_id} \
    --threshold 0.3 \
    --box_unclip_ratio 1.5 \
    --use_polygon_score 0 \
    --elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder dbnet_output
    """

    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/text_detection/eval.py \
    --test_image_path  /opt/vastai/vaststreamx/data/datasets/ch4_test_images \
    --boxes_npz_dir ./dbnet_output \
    --label_file ./data/labels/test_icdar2015_label.txt 
    """

    res = run_cmd(cmd)

    precision = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"'precision': \d+.\d+", string=res).group(),
    ).group()
    recall = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"'recall': \d+.\d+", string=res).group()
    ).group()
    hmean = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"'hmean': \d+.\d+", string=res).group()
    ).group()

    precision_thresh = 0.82
    recall_thresh = 0.79
    hmean_thresh = 0.80

    assert (
        float(precision) > precision_thresh
    ), f"dbnet:precision ${precision} is smaller than {precision_thresh}"
    assert (
        float(recall) > recall_thresh
    ), f"dbnet:recall ${recall} is smaller than {recall_thresh}"
    assert (
        float(hmean) > hmean_thresh
    ), f"dbnet:hmean ${hmean} is smaller than {hmean_thresh}"

    os.system("rm -rf dbnet_output")


#####################  python test #####################


##############################  one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_dbnet_py(device_id):
    cmd = f"""
    python3 ./samples/text_detection/dbnet.py \
    -m /opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod \
    --vdsp_params ./data/configs/dbnet_rgbplanar.json \
    --device_id {device_id} \
    --input_file ./data/images/detect.jpg \
    --output_file dbnet_result.jpg
    """
    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score:\d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.65

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(f"dbnet:score {score} is smaller than {score_thres}")

    assert len(scores) == 14, f"dbnet:detected object count={len(scores)} is not 14"
    assert os.path.exists("dbnet_result.jpg"), "dbnet:can't find dbnet_result.jpg"

    os.system("rm dbnet_result.jpg")



#################  performance test

@pytest.mark.fast
def test_dbnet_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/text_detection/dbnet_prof.py \
    -m /opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod \
    --vdsp_params ./data/configs/dbnet_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --shape [3,736,1280] \
    --iterations 600 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 90
    assert (
        float(qps) > qps_thresh
    ), f"dbnet:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    python3 ./samples/text_detection/dbnet_prof.py \
    -m /opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod \
    --vdsp_params ./data/configs/dbnet_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --shape [3,736,1280] \
    --iterations 300 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 50
    assert (
        float(qps) > qps_thresh
    ), f"dbnet:best latancy qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_dbnet_py_precision(device_id):
    os.system("mkdir -p dbnet_output")
    cmd = f"""
    python3 ./samples/text_detection/dbnet.py \
    -m /opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod \
    --vdsp_params ./data/configs/dbnet_rgbplanar.json \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder dbnet_output
    """

    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/text_detection/eval.py \
    --test_image_path  /opt/vastai/vaststreamx/data/datasets/ch4_test_images \
    --boxes_npz_dir ./dbnet_output \
    --label_file ./data/labels/test_icdar2015_label.txt 
    """

    res = run_cmd(cmd)

    precision = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"'precision': \d+.\d+", string=res).group(),
    ).group()
    recall = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"'recall': \d+.\d+", string=res).group()
    ).group()
    hmean = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"'hmean': \d+.\d+", string=res).group()
    ).group()

    precision_thresh = 0.82
    recall_thresh = 0.79
    hmean_thresh = 0.80

    assert (
        float(precision) > precision_thresh
    ), f"dbnet:precision ${precision} is smaller than {precision_thresh}"
    assert (
        float(recall) > recall_thresh
    ), f"dbnet:recall ${recall} is smaller than {recall_thresh}"
    assert (
        float(hmean) > hmean_thresh
    ), f"dbnet:hmean ${hmean} is smaller than {hmean_thresh}"

    os.system("rm -rf dbnet_output")
