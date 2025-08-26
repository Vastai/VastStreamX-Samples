
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

################ data prepare
@pytest.mark.fast
@pytest.mark.ai_integration
@pytest.mark.slow
def test_grounding_dino_cpp_generate_tokens_for_cpp(device_id):
    cmd = f"""
    python3 ./samples/detection/grounding_dino/generate_tokens_for_cpp.py \
    --tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/bert-base-uncased \
    --label_file ./data/labels/coco2id.txt \
    --output_file input_tokens.npz
    """
    run_cmd(cmd)
    assert os.path.exists("input_tokens.npz"), "grounding_dino c++:can't find input_tokens.npz"

################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_grounding_dino_cpp(device_id):
    assert os.path.exists("input_tokens.npz")
    cmd = f"""
    ./build/vaststreamx-samples/bin/grounding_dino \
    --txtmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod \
    --txtmod_vdsp_params  ./data/configs/clip_txt_vdsp.json \
    --imgmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod \
    --imgmod_vdsp_params ./data/configs/groundingdino_bgr888.json \
    --decmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/decoder-fp16-none-1_22223_256_1_195_256_1_195_1_195_1_195_195-vacc/mod \
    --npz_file input_tokens.npz \
    --label_file ./data/labels/coco2id.txt \
    --positive_map_file ./data/bin/positive_map.bin \
    --device_id {device_id} \
    --threshold 0.2 \
    --input_file /opt/vastai/vaststreamx/data/datasets/det_coco_val/000000000139.jpg  \
    --output_file grounding_dino_result.jpg
    """

    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.2

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(f"grounding_dino c++:score {score} is smaller than {score_thres}")

    assert len(scores) == 53, f"grounding_dino c++:detected object count={len(scores)} is not 53"
    assert os.path.exists("grounding_dino_result.jpg"), "grounding_dino c++:can't find grounding_dino_result.jpg"
    assert (
        re.findall(pattern=r"tv", string=res) != []
    ), "grounding_dino c++:can't find tv in result string"
    assert (
        re.findall(pattern=r"person", string=res) != []
    ), "grounding_dino c++:can't find person in result string"
    assert (
        re.findall(pattern=r"clock", string=res) != []
    ), "grounding_dino c++:can't find clock in result string"

    os.system("rm grounding_dino_result.jpg")


#################  performance test
@pytest.mark.fast
def test_grounding_dino_cpp_performance(device_id):
    assert os.path.exists("input_tokens.npz")
    # 测试 text_encoder 最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/grounding_dino_text_enc_prof \
    -m /opt/vastai/vaststreamx/data/models/groundingdino/text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod \
    --vdsp_params ./data/configs/clip_txt_vdsp.json \
    --test_npz_file input_tokens.npz \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 1000 \
    --percentiles [50,90,95,99] \
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
    ), f"grounding_dino c++:text_encoder best prof qps {qps} is smaller than {qps_thresh}"

    # 测试 text_encoder 最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/grounding_dino_text_enc_prof \
    -m /opt/vastai/vaststreamx/data/models/groundingdino/text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod \
    --vdsp_params ./data/configs/clip_txt_vdsp.json \
    --test_npz_file input_tokens.npz \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 1000 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 105
    assert (
        float(qps) > qps_thresh
    ), f"grounding_dino c++: text_encoder best latancy qps {qps} is smaller than {qps_thresh}"

    # 测试 image_encoder 最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/grounding_dino_image_enc_prof \
    -m /opt/vastai/vaststreamx/data/models/groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod \
    --vdsp_params ./data/configs/groundingdino_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 10 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1  
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 1.5
    assert (
        float(qps) > qps_thresh
    ), f"grounding_dino c++: image_encoder best prof qps {qps} is smaller than {qps_thresh}"

    # 测试 image_encoder 最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/grounding_dino_image_enc_prof \
    -m /opt/vastai/vaststreamx/data/models/groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod \
    --vdsp_params ./data/configs/groundingdino_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 10 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 1.5 
    assert (
        float(qps) > qps_thresh
    ), f"grounding_dino c++: image_encoder best latancy qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.slow
def test_grounding_dino_cpp_precision(device_id):
    assert os.path.exists("input_tokens.npz")
    os.system("mkdir -p grounding_dino_out")
    cmd = f"""
    ./build/vaststreamx-samples/bin/grounding_dino \
    --txtmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod \
    --txtmod_vdsp_params  ./data/configs/clip_txt_vdsp.json \
    --imgmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod \
    --imgmod_vdsp_params ./data/configs/groundingdino_bgr888.json \
    --decmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/decoder-fp16-none-1_22223_256_1_195_256_1_195_1_195_1_195_195-vacc/mod \
    --npz_file input_tokens.npz \
    --label_file ./data/labels/coco2id.txt \
    --positive_map_file ./data/bin/positive_map.bin \
    --device_id {device_id} \
    --threshold 0.01 \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder ./grounding_dino_out
    """

    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/detection/eval_map.py \
    --gt ./evaluation/detection/instances_val2017.json \
    --txt ./grounding_dino_out
    """

    res = run_cmd(cmd)

    bbox_mAP = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"'bbox_mAP': \d+.\d+", string=res).group(),
    ).group()

    bbox_mAP_50 = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"'bbox_mAP_50': \d+.\d+", string=res).group(),
    ).group()

    bbox_mAP_thres = 0.45
    bbox_mAP_50_thres = 0.60

    assert (
        float(bbox_mAP) > bbox_mAP_thres
    ), f"grounding_dino c++: bbox_mAP {bbox_mAP} is smaller than {bbox_mAP_thres}"
    assert (
        float(bbox_mAP_50) > bbox_mAP_50_thres
    ), f"grounding_dino c++: bbox_mAP_50 {bbox_mAP_50} is smaller than {bbox_mAP_50_thres}"

    os.system("rm -rf grounding_dino_out input_tokens.npz")

# #####################  python test #####################

################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_grounding_dino_py(device_id):
    cmd = f"""
    python3 ./samples/detection/grounding_dino/grounding_dino.py \
    --txtmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod \
    --txtmod_vdsp_params  ./data/configs/clip_txt_vdsp.json \
    --imgmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod \
    --imgmod_vdsp_params ./data/configs/groundingdino_bgr888.json \
    --decmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/decoder-fp16-none-1_22223_256_1_195_256_1_195_1_195_1_195_195-vacc/mod \
    --tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/bert-base-uncased \
    --label_file ./data/labels/coco2id.txt \
    --device_id {device_id} \
    --threshold 0.2 \
    --input_file /opt/vastai/vaststreamx/data/datasets/det_coco_val/000000000139.jpg  \
    --output_file grounding_dino_result.jpg
    """

    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.2

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(f"grounding_dino python:score {score} is smaller than {score_thres}")

    assert len(scores) == 53, f"grounding_dino python:detected object count={len(scores)} is not 53"
    assert os.path.exists("grounding_dino_result.jpg"), "grounding_dino python:can't find grounding_dino_result.jpg"
    assert (
        re.findall(pattern=r"tv", string=res) != []
    ), "grounding_dino python:can't find tv in result string"
    assert (
        re.findall(pattern=r"person", string=res) != []
    ), "grounding_dino python:can't find person in result string"
    assert (
        re.findall(pattern=r"clock", string=res) != []
    ), "grounding_dino python:can't find clock in result string"

    os.system("rm grounding_dino_result.jpg")


#################  performance test
@pytest.mark.fast
def test_grounding_dino_py_performance(device_id):
    # 测试 text_encoder 最大吞吐
    cmd = f"""
    python3 ./samples/detection/grounding_dino/grounding_dino_text_enc_prof.py \
    -m /opt/vastai/vaststreamx/data/models/groundingdino/text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod \
    --vdsp_params ./data/configs/clip_txt_vdsp.json \
    --tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/bert-base-uncased \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 1000 \
    --percentiles [50,90,95,99] \
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
    ), f"grounding_dino python:text_encoder best prof qps {qps} is smaller than {qps_thresh}"

    # 测试 text_encoder 最小时延
    cmd = f"""
    python3 ./samples/detection/grounding_dino/grounding_dino_text_enc_prof.py \
    -m /opt/vastai/vaststreamx/data/models/groundingdino/text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod \
    --vdsp_params ./data/configs/clip_txt_vdsp.json \
    --tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/bert-base-uncased \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 1000 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 105
    assert (
        float(qps) > qps_thresh
    ), f"grounding_dino python: text_encoder best latancy qps {qps} is smaller than {qps_thresh}"

    # 测试 image_encoder 最大吞吐
    cmd = f"""
    python3 ./samples/detection/grounding_dino/grounding_dino_image_enc_prof.py \
    -m /opt/vastai/vaststreamx/data/models/groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod \
    --vdsp_params ./data/configs/groundingdino_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 10 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1  
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 1.5
    assert (
        float(qps) > qps_thresh
    ), f"grounding_dino python: image_encoder best prof qps {qps} is smaller than {qps_thresh}"


    # 测试 image_encoder 最小时延
    cmd = f"""
    python3 ./samples/detection/grounding_dino/grounding_dino_image_enc_prof.py \
    -m /opt/vastai/vaststreamx/data/models/groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod \
    --vdsp_params ./data/configs/groundingdino_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 10 \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 1.5 
    assert (
        float(qps) > qps_thresh
    ), f"grounding_dino python: image_encoder best latancy qps {qps} is smaller than {qps_thresh}"


############################# dataset test
@pytest.mark.slow
def test_grounding_dino_py_precision(device_id):
    os.system("mkdir -p grounding_dino_out")
    cmd = f"""
    python3 ./samples/detection/grounding_dino/grounding_dino.py \
    --txtmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod \
    --txtmod_vdsp_params  ./data/configs/clip_txt_vdsp.json \
    --imgmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod \
    --imgmod_vdsp_params ./data/configs/groundingdino_bgr888.json \
    --decmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/decoder-fp16-none-1_22223_256_1_195_256_1_195_1_195_1_195_195-vacc/mod \
    --tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/bert-base-uncased \
    --label_file ./data/labels/coco2id.txt \
    --device_id {device_id} \
    --threshold 0.01 \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder ./grounding_dino_out
    """

    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/detection/eval_map.py \
    --gt ./evaluation/detection/instances_val2017.json \
    --txt ./grounding_dino_out
    """

    res = run_cmd(cmd)

    bbox_mAP = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"'bbox_mAP': \d+.\d+", string=res).group(),
    ).group()

    bbox_mAP_50 = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"'bbox_mAP_50': \d+.\d+", string=res).group(),
    ).group()

    bbox_mAP_thres = 0.45
    bbox_mAP_50_thres = 0.60

    assert (
        float(bbox_mAP) > bbox_mAP_thres
    ), f"grounding_dino python: bbox_mAP {bbox_mAP} is smaller than {bbox_mAP_thres}"
    assert (
        float(bbox_mAP_50) > bbox_mAP_50_thres
    ), f"grounding_dino python: bbox_mAP_50 {bbox_mAP_50} is smaller than {bbox_mAP_50_thres}"

    os.system("rm -rf grounding_dino_out ")

