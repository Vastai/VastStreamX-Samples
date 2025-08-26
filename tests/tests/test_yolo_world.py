
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
################ prepare tokens
@pytest.mark.fast
@pytest.mark.ai_integration
def test_yoloworld_cpp_prepare_tokens(device_id):
    os.system("mkdir -p tokens")
    cmd = f"""
    python3 ./samples/yolo_world/make_tokens.py \
    --class_text ./data/labels/lvis_v1_class_texts.json \
    --tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/clip-vit-base-patch32 \
    --save_path tokens
    """
    run_cmd(cmd)
    assert os.path.exists("tokens")
    
################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_yoloworld_cpp(device_id):
    assert os.path.exists("tokens")
    cmd = f"""
    ./build/vaststreamx-samples/bin/yolo_world \
    --imgmod_prefix  /opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod \
    --imgmod_vdsp_params ./data/configs/yolo_world_1280_1280_bgr888.json \
    --txtmod_prefix /opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod \
    --txtmod_vdsp_params  ./data/configs/clip_txt_vdsp.json \
    --device_id {device_id} \
    --label_file ./data/labels/lvis_v1_class_texts.json \
    --npz_files_path tokens \
    --input_file ./data/images/dog.jpg \
    --output_file yolo_world_result.jpg
    """
    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.5

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(f"yolo_world c++:score {score} is smaller than {score_thres}")

    assert len(scores) == 7, f"yolo_world c++:detected object count={len(scores)} is not 7"
    assert os.path.exists("yolo_world_result.jpg"), "yolo_world c++:can't find yolo_world_result.jpg"
    assert (
        re.findall(pattern=r"bicycle", string=res) != []
    ), "yolo_world c++:can't find bicycle in result string"
    assert (
        re.findall(pattern=r"dog", string=res) != []
    ), "yolo_world c++:can't find dog in result string"
    assert (
        re.findall(pattern=r"truck", string=res) != []
    ), "yolo_world c++:can't find truck in result string"

    os.system("rm yolo_world_result.jpg")

#################  performance test

@pytest.mark.fast
def test_yoloworld_cpp_performance_text(device_id):
    assert os.path.exists("tokens")
    # 测试 yolo_world_text 最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/yolo_world_text_prof \
    --model_prefix /opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod \
    --vdsp_params  ./data/configs/clip_txt_vdsp.json \
    --test_npz_file ./tokens/Bible.npz  \
    --device_ids [{device_id}] \
    --batch_size  1 \
    --instance 1 \
    --iterations 2000 \
    --percentiles "[50,90,95,99]" \
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
    ), f"yolo_world_text c++:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试 yolo_world_text 模型最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/yolo_world_text_prof \
    --model_prefix /opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod \
    --vdsp_params  ./data/configs/clip_txt_vdsp.json \
    --test_npz_file ./tokens/Bible.npz  \
    --device_ids [{device_id}] \
    --batch_size  1 \
    --instance 1 \
    --iterations 2000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 330
    assert (
        float(qps) > qps_thresh
    ), f"yolo_world_text c++:best latancy qps {qps} is smaller than {qps_thresh}"

@pytest.mark.fast
def test_yoloworld_cpp_performance_image(device_id):
    # 测试 yolo_world_image 最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/yolo_world_image_prof \
    --model_prefix /opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod  \
    --vdsp_params ./data/configs/yolo_world_1280_1280_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size  1 \
    --instance 1 \
    --iterations 20 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 2.9
    assert (
        float(qps) > qps_thresh
    ), f"yolo_world_image c++:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试 yolo_world_image 模型最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/yolo_world_image_prof \
    --model_prefix /opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod  \
    --vdsp_params ./data/configs/yolo_world_1280_1280_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size  1 \
    --instance 1 \
    --iterations 20 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 2.2
    assert (
        float(qps) > qps_thresh
    ), f"yolo_world_image c++:best latancy qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.slow
def test_yoloworld_cpp_precision(device_id):
    assert os.path.exists("tokens")
    cmd = f"""
    ./build/vaststreamx-samples/bin/yolo_world \
    --imgmod_prefix  /opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod \
    --imgmod_vdsp_params ./data/configs/yolo_world_1280_1280_bgr888.json \
    --txtmod_prefix /opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod \
    --txtmod_vdsp_params  ./data/configs/clip_txt_vdsp.json \
    --device_id {device_id} \
    --label_file ./data/labels/lvis_v1_class_texts.json \
    --npz_files_path tokens \
    --max_per_image 300 \
    --score_thresh  0.001 \
    --iou_thresh  0.7 \
    --nms_pre  30000 \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_file yoloworld_dataset_result.json
    """
    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/yolo_world/eval_lvis.py  \
    --path_res yoloworld_dataset_result.json \
    --path_ann_file ./evaluation/yolo_world/lvis_v1_minival_inserted_image_name.json
    """

    res = run_cmd(cmd,False)

    precision_strs = re.findall(pattern=r"catIds=all] = \d+.\d+",string=res)

    assert len(precision_strs) == 10 
    bbox_mAP = re.search(pattern=r"\d+.\d+", string=precision_strs[0]).group()
    bbox_mAP_50 =  re.search(pattern=r"\d+.\d+", string=precision_strs[1]).group()

    bbox_mAP_thres = 0.34
    bbox_mAP_50_thres = 0.45

    assert (
        float(bbox_mAP) > bbox_mAP_thres
    ), f"yolo_world c++: bbox_mAP {bbox_mAP} is smaller than {bbox_mAP_thres}"
    assert (
        float(bbox_mAP_50) > bbox_mAP_50_thres
    ), f"yolo_world c++: bbox_mAP_50 {bbox_mAP_50} is smaller than {bbox_mAP_50_thres}"

    os.system("rm -rf tokens yoloworld_dataset_result.json")


#####################  python test #####################

################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_yoloworld_py(device_id):
    cmd = f"""
    python3 ./samples/yolo_world/yolo_world.py \
    --imgmod_prefix  /opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod \
    --imgmod_vdsp_params ./data/configs/yolo_world_1280_1280_bgr888.json \
    --txtmod_prefix /opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod \
    --txtmod_vdsp_params  ./data/configs/clip_txt_vdsp.json \
    --tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/clip-vit-base-patch32 \
    --device_id {device_id} \
    --max_per_image 300 \
    --score_thres  0.5 \
    --iou_thres  0.7 \
    --nms_pre  30000 \
    --label_file ./data/labels/lvis_v1_class_texts.json \
    --input_file ./data/images/dog.jpg \
    --output_file yolo_world_result.jpg
    """
    res = run_cmd(cmd,False)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.5

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(f"yolo_world python:score {score} is smaller than {score_thres}")

    assert len(scores) == 7, f"yolo_world python:detected object count={len(scores)} is not 7"
    assert os.path.exists("yolo_world_result.jpg"), "yolo_world python:can't find yolo_world_result.jpg"
    assert (
        re.findall(pattern=r"bicycle", string=res) != []
    ), "yolo_world python:can't find bicycle in result string"
    assert (
        re.findall(pattern=r"dog", string=res) != []
    ), "yolo_world python:can't find dog in result string"
    assert (
        re.findall(pattern=r"truck", string=res) != []
    ), "yolo_world python:can't find truck in result string"

    os.system("rm yolo_world_result.jpg")

#################  performance test
@pytest.mark.fast
def test_yoloworld_py_performance_text(device_id):
    # 测试 yolo_world_text 最大吞吐
    cmd = f"""
    python3 ./samples/yolo_world/yolo_world_text_prof.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod \
    --vdsp_params  ./data/configs/clip_txt_vdsp.json \
    --tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/clip-vit-base-patch32 \
    --device_ids [{device_id}] \
    --batch_size  1 \
    --instance 1 \
    --iterations 2000 \
    --percentiles "[50,90,95,99]" \
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
    ), f"yolo_world_text python:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试 yolo_world_text 模型最小时延
    cmd = f"""
    python3 ./samples/yolo_world/yolo_world_text_prof.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod \
    --vdsp_params  ./data/configs/clip_txt_vdsp.json \
    --tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/clip-vit-base-patch32 \
    --device_ids [{device_id}] \
    --batch_size  1 \
    --instance 1 \
    --iterations 2000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 330
    assert (
        float(qps) > qps_thresh
    ), f"yolo_world_text python:best latancy qps {qps} is smaller than {qps_thresh}"

@pytest.mark.fast
def test_yoloworld_py_performance_image(device_id):
    # 测试 yolo_world_image 最大吞吐
    cmd = f"""
    python3 ./samples/yolo_world/yolo_world_image_prof.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod  \
    --vdsp_params ./data/configs/yolo_world_1280_1280_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size  1 \
    --instance 1 \
    --iterations 20 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 2.9
    assert (
        float(qps) > qps_thresh
    ), f"yolo_world_image python:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试 yolo_world_image 模型最小时延
    cmd = f"""
    python3 ./samples/yolo_world/yolo_world_image_prof.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod  \
    --vdsp_params ./data/configs/yolo_world_1280_1280_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size  1 \
    --instance 1 \
    --iterations 20 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 2.2
    assert (
        float(qps) > qps_thresh
    ), f"yolo_world_image python:best latancy qps {qps} is smaller than {qps_thresh}"


############################# dataset test
@pytest.mark.slow
def test_yoloworld_py_precision(device_id):
    cmd = f"""
    python3 ./samples/yolo_world/yolo_world.py \
    --imgmod_prefix  /opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod \
    --imgmod_vdsp_params ./data/configs/yolo_world_1280_1280_bgr888.json \
    --txtmod_prefix /opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod \
    --txtmod_vdsp_params  ./data/configs/clip_txt_vdsp.json \
    --tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/clip-vit-base-patch32 \
    --device_id {device_id} \
    --max_per_image 300 \
    --score_thres  0.001 \
    --iou_thres  0.7 \
    --nms_pre  30000 \
    --label_file ./data/labels/lvis_v1_class_texts.json \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_file yoloworld_dataset_result.json
    """
    run_cmd(cmd,False)

    cmd = f"""
    python3 ./evaluation/yolo_world/eval_lvis.py  \
    --path_res yoloworld_dataset_result.json \
    --path_ann_file ./evaluation/yolo_world/lvis_v1_minival_inserted_image_name.json
    """
    res = run_cmd(cmd,False)

    precision_strs = re.findall(pattern=r"catIds=all] = \d+.\d+",string=res)

    assert len(precision_strs) == 10 
    bbox_mAP = re.search(pattern=r"\d+.\d+", string=precision_strs[0]).group()
    bbox_mAP_50 =  re.search(pattern=r"\d+.\d+", string=precision_strs[1]).group()

    bbox_mAP_thres = 0.34
    bbox_mAP_50_thres = 0.45

    assert (
        float(bbox_mAP) > bbox_mAP_thres
    ), f"yolo_world python: bbox_mAP {bbox_mAP} is smaller than {bbox_mAP_thres}"
    assert (
        float(bbox_mAP_50) > bbox_mAP_50_thres
    ), f"yolo_world python: bbox_mAP_50 {bbox_mAP_50} is smaller than {bbox_mAP_50_thres}"

    os.system("rm -rf  yoloworld_dataset_result.json")


