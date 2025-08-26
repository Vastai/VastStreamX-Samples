
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
def test_clip_cpp(device_id):
    cmd = f"""
    python3 ./samples/clip/make_input_npz.py \
    --label_file ./samples/clip/test_label.txt \
    --npz_files_path npz_files
    """

    res = run_cmd(cmd)


    cmd = f"""
    ./build/vaststreamx-samples/bin/clip_sample \
    --imgmod_prefix /opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod \
    --norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
    --space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
    --txtmod_prefix /opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod \
    --txtmod_vdsp_params ./data/configs/clip_txt_vdsp.json \
    --device_id {device_id} \
    --label_file ./samples/clip/test_label.txt \
    --input_file ./data/images/CLIP.png \
    --npz_files_path npz_files
    """

    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_gt = 0.9
    assert (
        float(scores[0]) > score_gt
    ), f"clip  c++:max score {scores[0]} is smaller than {score_gt}"


    os.system("rm -rf npz_files")

############# preprocess
@pytest.mark.fast
@pytest.mark.slow
def test_clip_cpp_preprocess(device_id):

    cmd = f"""
    python3 ./samples/clip/make_input_npz.py \
    --label_file ./data/labels/imagenet.txt \
    --npz_files_path imagenet_label_npz_files
    """
    res = run_cmd(cmd)

    assert os.path.exists("imagenet_label_npz_files")

#################  performance test
@pytest.mark.fast
def test_clip_cpp_performance(device_id):
    assert os.path.exists("imagenet_label_npz_files")

    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/clip_image_prof \
    -m /opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod \
    --norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
    --space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
    --device_ids [{device_id}] \
    --batch_size  1 \
    --instance 1 \
    --iterations 1000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 180
    assert (
        float(qps) > qps_thresh
    ), f"clip image c++:best prof qps {qps} is smaller than {qps_thresh}"


    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/clip_image_prof \
    -m /opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod \
    --norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
    --space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
    --device_ids [{device_id}] \
    --batch_size  1 \
    --instance 1 \
    --iterations 8000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 130
    assert (
        float(qps) > qps_thresh
    ), f"clip image c++:best latency qps {qps} is smaller than {qps_thresh}"


    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/clip_text_prof  \
    -m /opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod \
    --vdsp_params ./data/configs/clip_txt_vdsp.json \
    --test_input_npz "./imagenet_label_npz_files/Afghan hound, Afghan.npz" \
    --device_ids [{device_id}] \
    --batch_size  1 \
    --instance 1 \
    --iterations 1500 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 330
    assert (
        float(qps) > qps_thresh
    ), f"clip text c++:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/clip_text_prof  \
    -m /opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod \
    --vdsp_params ./data/configs/clip_txt_vdsp.json \
    --test_input_npz "./imagenet_label_npz_files/Afghan hound, Afghan.npz" \
    --device_ids [{device_id}] \
    --batch_size  1 \
    --instance 1 \
    --iterations 1000 \
    --percentiles "[50,90,95,99]" \
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
    ), f"clip text c++:best latency qps {qps} is smaller than {qps_thresh}"




################# dataset test
@pytest.mark.slow
def test_clip_cpp_precision(device_id):
    assert os.path.exists("imagenet_label_npz_files")
    cmd = f"""
    ./build/vaststreamx-samples/bin/clip_sample \
    --imgmod_prefix /opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod \
    --norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
    --space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
    --txtmod_prefix /opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod \
    --txtmod_vdsp_params ./data/configs/clip_txt_vdsp.json \
    --device_id {device_id} \
    --label_file ./data/labels/imagenet.txt \
    --npz_files_path imagenet_label_npz_files \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/ILSVRC2012_img_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_file clip_result.txt
    """
    res = run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/classification/eval_topk.py  clip_result.txt  
    """

    res = run_cmd(cmd)

    top1_rate = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"top1_rate: \d+.\d+", string=res).group(),
    ).group()


    top5_rate = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"top5_rate: \d+.\d+", string=res).group(),
    ).group()


    top1_rate_thresh = 55.2
    top5_rate_thresh = 82.2

    assert (
        float(top1_rate) > top1_rate_thresh
    ), f"clip c++:top1_rate {top1_rate} is smaller than {top1_rate_thresh}"
    assert (
        float(top5_rate) > top5_rate_thresh
    ), f"clip c++:top5_rate {top5_rate} is smaller than {top5_rate_thresh}"


    os.system("rm -rf clip_result.txt imagenet_label_npz_files")


#####################  python test #####################


##############################  one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_clip_py(device_id):
    cmd = f"""
    python3 ./samples/clip/clip_sample.py \
    --imgmod_prefix /opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod \
    --norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
    --space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
    --txtmod_prefix /opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod \
    --txtmod_vdsp_params ./data/configs/clip_txt_vdsp.json \
    --device_id {device_id} \
    --input_file ./data/images/CLIP.png \
    --strings "[a diagram,a dog,a cat]"
    """
    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_gt = 0.9
    assert (
        float(scores[0]) > score_gt
    ), f"clip python:max score {scores[0]} is smaller than {score_gt}"



#################  performance test
@pytest.mark.fast
def test_clip_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/clip/clip_image_prof.py \
    -m /opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod \
    --norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
    --space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
    --device_ids [{device_id}] \
    --batch_size  1 \
    --instance 1 \
    --iterations 1000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 180
    assert (
        float(qps) > qps_thresh
    ), f"clip image python:best prof qps {qps} is smaller than {qps_thresh}"


    # 测试最小时延
    cmd = f"""
    python3 ./samples/clip/clip_image_prof.py \
    -m /opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod \
    --norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
    --space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
    --device_ids [{device_id}] \
    --batch_size  1 \
    --instance 1 \
    --iterations 800 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 130
    assert (
        float(qps) > qps_thresh
    ), f"clip image python:best latency qps {qps} is smaller than {qps_thresh}"


    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/clip/clip_text_prof.py \
    -m /opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod \
    --vdsp_params ./data/configs/clip_txt_vdsp.json \
    --device_ids [{device_id}] \
    --batch_size  1 \
    --instance 1 \
    --iterations 1500 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 330
    assert (
        float(qps) > qps_thresh
    ), f"clip text python:best prof qps {qps} is smaller than {qps_thresh}"


    # 测试最小时延
    cmd = f"""
    python3 ./samples/clip/clip_text_prof.py \
    -m /opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod \
    --vdsp_params ./data/configs/clip_txt_vdsp.json \
    --device_ids [{device_id}] \
    --batch_size  1 \
    --instance 1 \
    --iterations 1200 \
    --percentiles "[50,90,95,99]" \
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
    ), f"clip text python:best latency qps {qps} is smaller than {qps_thresh}"




################# dataset test
@pytest.mark.slow
def test_clip_py_precision(device_id):
    cmd = f"""
    python3 ./samples/clip/clip_sample.py \
    --imgmod_prefix /opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod \
    --norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
    --space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
    --txtmod_prefix /opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod \
    --txtmod_vdsp_params ./data/configs/clip_txt_vdsp.json \
    --label_file ./data/labels/imagenet.txt \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/ILSVRC2012_img_val_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_file clip_result.txt
    """

    res = run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/classification/eval_topk.py  clip_result.txt  
    """

    res = run_cmd(cmd)

    top1_rate = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"top1_rate: \d+.\d+", string=res).group(),
    ).group()


    top5_rate = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"top5_rate: \d+.\d+", string=res).group(),
    ).group()


    top1_rate_thresh = 55.2
    top5_rate_thresh = 82.2

    assert (
        float(top1_rate) > top1_rate_thresh
    ), f"clip python:top1_rate {top1_rate} is smaller than {top1_rate_thresh}"
    assert (
        float(top5_rate) > top5_rate_thresh
    ), f"clip python:top5_rate {top5_rate} is smaller than {top5_rate_thresh}"

    os.system("rm -rf clip_result.txt ")
