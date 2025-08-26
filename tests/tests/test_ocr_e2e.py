
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
import glob

################# c++ test  #######################

#################  e2e test ###########################
@pytest.mark.fast
@pytest.mark.ai_integration
def test_ocr_e2e_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/ocr_e2e \
    --det_model /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
    --det_config ./data/configs/dbnet_rgbplanar.json \
    --cls_model  /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod  \
    --cls_config ./data/configs/crnn_rgbplanar.json \
    --rec_model /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
    --rec_config ./data/configs/crnn_rgbplanar.json \
    --det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
    --rec_label_file ./data/labels/ocr_rec_dict.txt \
    --rec_drop_score 0.5 \
    --use_angle_cls 1 \
    --device_ids [{device_id}] \
    --input_file ./data/images/detect.jpg \
    --output_file ocr_res.jpg
    """

    res = run_cmd(cmd)

    res_file = glob.glob("*ocr_res.jpg")
    assert len(res_file) > 0, "ocr_e2e c++: can't find *ocr_res.jpg"
    os.remove(res_file[0])

    dict_strs = re.findall(pattern=r'score: \d+.\d+', string=res)
    score_strs = []
    for dstr in dict_strs:
        score = re.findall(pattern=r'\d+.\d+', string=dstr)[0]
        score_strs.append(score)

    assert len(score_strs) >= 8, f"ocr_e2e c++: len(score_strs) = {len(score_strs)} is not equal to 8"

    score_thresh = 0.7
    for score in score_strs:
        assert(float(score) > score_thresh), f"ocr_e2e c++: score={float(score)} is smaller than {score_thresh}, res={res}"



#################  performance test #######################
@pytest.mark.fast
def test_ocr_e2e_cpp_text_det_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/text_det_prof \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
    --vdsp_params ./data/configs/dbnet_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 600 \
    --shape "[3,736,1280]" \
    --elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 80
    assert (
        float(qps) > qps_thresh
    ), f"ocr_e2e det python:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/text_det_prof \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
    --vdsp_params ./data/configs/dbnet_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 600 \
    --shape "[3,736,1280]" \
    --elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
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
    ), f"ocr_e2e det c++:best latency qps {qps} is smaller than {qps_thresh}"



@pytest.mark.fast
def test_ocr_e2e_cpp_text_cls_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/text_cls_prof \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 32 \
    --instance 1 \
    --iterations 600 \
    --shape "[3,48,192]" \
    --queue_size 1
    """ 

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 1800
    assert (
        float(qps) > qps_thresh
    ), f"ocr_e2e cls c++:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/text_cls_prof \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_ids [{device_id}] \
    --batch_size 32 \
    --instance 1 \
    --iterations 600 \
    --shape "[3,48,192]" \
    --queue_size 0
    """ 

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 1500
    assert (
        float(qps) > qps_thresh
    ), f"ocr_e2e cls c++:best latency qps {qps} is smaller than {qps_thresh}"




@pytest.mark.fast
def test_ocr_e2e_cpp_text_rec_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/text_rec_prof \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_ids [{device_id}] \
    --label_file ./data/labels/ocr_rec_dict.txt \
    --batch_size 1 \
    --instance 4 \
    --shape "[3,48,320]" \
    --iterations 1000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """ 

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 245
    assert (
        float(qps) > qps_thresh
    ), f"ocr_e2e rec c++:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/text_rec_prof \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_ids [{device_id}] \
    --label_file ./data/labels/ocr_rec_dict.txt \
    --batch_size 1 \
    --instance 1 \
    --shape "[3,48,320]" \
    --iterations 1000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """ 

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 70
    assert (
        float(qps) > qps_thresh
    ), f"ocr_e2e rec c++:best latency qps {qps} is smaller than {qps_thresh}"



#################  text detection test #######################
@pytest.mark.fast
@pytest.mark.ai_integration
def test_ocr_e2e_cpp_text_det(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/text_det \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
    --vdsp_params ./data/configs/dbnet_rgbplanar.json \
    --device_id {device_id} \
    --threshold 0.3 \
    --box_unclip_ratio 1.5 \
    --use_polygon_score 0 \
    --elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
    --input_file ./data/images/detect.jpg \
    --output_file text_det_result.jpg
    """

    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score:\d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.6

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(f"ocr_e2e text detection c++:score {score} is smaller than {score_thres}")

    assert os.path.exists("text_det_result.jpg"), "ocr_e2e text detection c++:can't find text_det_result.jpg"

    os.system("rm text_det_result.jpg")

#################  detection dataset test
@pytest.mark.fast
def test_ocr_e2e_cpp_text_det_precision(device_id):
    os.system("mkdir -p text_det_output")

    cmd = f"""
    ./build/vaststreamx-samples/bin/text_det \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
    --vdsp_params ./data/configs/dbnet_rgbplanar.json \
    --device_id {device_id} \
    --threshold 0.3 \
    --box_unclip_ratio 1.5 \
    --use_polygon_score 0 \
    --elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder text_det_output
    """

    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/text_detection/eval.py \
    --test_image_path  /opt/vastai/vaststreamx/data/datasets/ch4_test_images \
    --boxes_npz_dir ./text_det_output \
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

    precision_thresh = 0.54
    recall_thresh = 0.41
    hmean_thresh = 0.47

    assert (
        float(precision) > precision_thresh
    ), f"ocr_e2e text detection c++:precision ${precision} is smaller than {precision_thresh}"
    assert (
        float(recall) > recall_thresh
    ), f"ocr_e2e text detection  c++:recall ${recall} is smaller than {recall_thresh}"
    assert (
        float(hmean) > hmean_thresh
    ), f"ocr_e2e text detection  c++ :hmean ${hmean} is smaller than {hmean_thresh}"

    os.system("rm -rf text_det_output")




#################  recognition one image test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_ocr_e2e_cpp_text_rec(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/text_rec \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_id {device_id} \
    --label_file ./data/labels/ocr_rec_dict.txt \
    --input_file ./data/images/word_336.png 
    """

    res = run_cmd(cmd)

    scores = re.findall(pattern=r"\d+.\d+", string=res)

    score_gt = 0.97
    assert (
        float(scores[0]) > score_gt
    ), f"ocr_e2e recognition c++: score {scores[0]} is smaller than {score_gt}"

    text = re.findall(pattern=r"SUPER", string=res)

    assert text != [], f'ocr_e2e recognition c++:can\'t find "SUPER" in result string'



############################# dataset test

@pytest.mark.fast
def test_ocr_e2e_cpp_text_rec_precision(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/text_rec \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_id {device_id} \
    --label_file ./data/labels/ocr_rec_dict.txt \
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

    acc_thresh = 0.80

    assert float(acc) > acc_thresh, f"ocr_e2e recognition c++:acc {acc} is smaller than {acc_thresh}"

    os.system("rm  cute80_pred.txt")

################# python test  #######################


#################  e2e test ############################
@pytest.mark.fast
@pytest.mark.ai_integration
def test_ocr_e2e_py(device_id):
    cmd = f"""
    python3 ./samples/ocr_e2e/ocr_e2e.py \
    --det_model /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
    --det_vdsp_params ./data/configs/dbnet_rgbplanar.json \
    --cls_model /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod \
    --cls_vdsp_params ./data/configs/crnn_rgbplanar.json \
    --rec_model /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
    --rec_vdsp_params ./data/configs/crnn_rgbplanar.json \
    --det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
    --device_ids [{device_id}] \
    --rec_label_file ./data/labels/ocr_rec_dict.txt \
    --input_file ./data/images/detect.jpg \
    --det_box_type quad \
    --output_file ocr_res.jpg
    """

    res = run_cmd(cmd)

    res_file = glob.glob("*ocr_res.jpg")
    assert len(res_file) > 0, "ocr_e2e python: can't find *ocr_res.jpg"
    os.remove(res_file[0])

    dict_strs = re.findall(pattern=r'\((.*?)\)', string=res)
    score_strs = []
    for dstr in dict_strs:
        score_strs.append(dstr.split(' ')[-1])

    assert len(score_strs) >= 7, f"ocr_e2e python: len(score_strs) = {len(score_strs)} is not equal to 7"

    score_thresh = 0.75
    for score in score_strs:
        assert(float(score) > score_thresh), f"ocr_e2e python: score={float(score)} is smaller than {score_thresh}, res={res}"




#################  performance test
@pytest.mark.fast
def test_ocr_e2e_py_text_det_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/ocr_e2e/text_det_prof.py \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
    --vdsp_params ./data/configs/dbnet_rgbplanar.json \
    --elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
    --device_ids [{device_id}]  \
    --batch_size 1 \
    --instance 1 \
    --shape "[3,736,1280]" \
    --iterations 500 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """ 

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 75
    assert (
        float(qps) > qps_thresh
    ), f"ocr_e2e det python:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    python3 ./samples/ocr_e2e/text_det_prof.py \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
    --vdsp_params ./data/configs/dbnet_rgbplanar.json \
    --elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
    --device_ids [{device_id}]  \
    --batch_size 1 \
    --instance 1 \
    --shape "[3,736,1280]" \
    --iterations 500 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """ 

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 36
    assert (
        float(qps) > qps_thresh
    ), f"ocr_e2e det python:best latency qps {qps} is smaller than {qps_thresh}"




@pytest.mark.fast
def test_ocr_e2e_py_text_cls_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/ocr_e2e/text_cls_prof.py \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_ids [{device_id}]  \
    --batch_size 32 \
    --instance 1 \
    --shape "[3,48,192]" \
    --iterations 500 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """ 

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 1800
    assert (
        float(qps) > qps_thresh
    ), f"ocr_e2e cls python:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    python3 ./samples/ocr_e2e/text_cls_prof.py \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_ids [{device_id}]  \
    --batch_size 32 \
    --instance 1 \
    --shape "[3,48,192]" \
    --iterations 500 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """ 

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 1200
    assert (
        float(qps) > qps_thresh
    ), f"ocr_e2e cls python:best latency qps {qps} is smaller than {qps_thresh}"




@pytest.mark.fast
def test_ocr_e2e_py_text_rec_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/ocr_e2e/text_rec_prof.py \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_ids [{device_id}]  \
    --batch_size 6 \
    --instance 1 \
    --label_file ./data/labels/ocr_rec_dict.txt \
    --shape "[3,48,320]" \
    --iterations 500 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 1
    """ 

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 245
    assert (
        float(qps) > qps_thresh
    ), f"ocr_e2e rec python:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    python3 ./samples/ocr_e2e/text_rec_prof.py \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_ids [{device_id}]  \
    --batch_size 6 \
    --instance 1 \
    --label_file ./data/labels/ocr_rec_dict.txt \
    --shape "[3,48,320]" \
    --iterations 500 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 \
    --queue_size 0
    """ 

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 85
    assert (
        float(qps) > qps_thresh
    ), f"ocr_e2e rec python:best latency qps {qps} is smaller than {qps_thresh}"



#################  text detection test

@pytest.mark.fast
@pytest.mark.ai_integration
def test_ocr_e2e_py_text_det(device_id):
    cmd = f"""
    python3 ./samples/ocr_e2e/text_det.py \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
    --vdsp_params ./data/configs/dbnet_rgbplanar.json \
    --elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
    --device_id {device_id} \
    --input_file ./data/images/detect.jpg \
    --output_file text_det_result.jpg
    """

    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score:\d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    score_thres = 0.6

    for score in scores:
        if float(score) < score_thres:
            raise RuntimeError(f"ocr_e2e text detection python:score {score} is smaller than {score_thres}")

    assert os.path.exists("text_det_result.jpg"), "ocr_e2e text detection python:can't find text_det_result.jpg"

    os.system("rm text_det_result.jpg")




#################  detection dataset test
@pytest.mark.fast
def test_ocr_e2e_py_text_det_precision(device_id):
    os.system("mkdir -p text_det_output")

    cmd = f"""
    python3 ./samples/ocr_e2e/text_det.py  \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
    --vdsp_params ./data/configs/dbnet_rgbplanar.json \
    --elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder text_det_output
    """

    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/text_detection/eval.py \
    --test_image_path  /opt/vastai/vaststreamx/data/datasets/ch4_test_images \
    --boxes_npz_dir ./text_det_output \
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

    precision_thresh = 0.54
    recall_thresh = 0.41
    hmean_thresh = 0.47

    assert (
        float(precision) > precision_thresh
    ), f"ocr_e2e text detection python:precision ${precision} is smaller than {precision_thresh}"
    assert (
        float(recall) > recall_thresh
    ), f"ocr_e2e text detection python:recall ${recall} is smaller than {recall_thresh}"
    assert (
        float(hmean) > hmean_thresh
    ), f"ocr_e2e text detection python :hmean ${hmean} is smaller than {hmean_thresh}"

    os.system("rm -rf text_det_output")



#################  recognition one image test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_ocr_e2e_py_text_rec(device_id):
    cmd = f"""
    python3 ./samples/ocr_e2e/text_rec.py \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_id {device_id} \
    --label_file ./data/labels/ocr_rec_dict.txt \
    --input_file ./data/images/word_336.png 
    """

    res = run_cmd(cmd)

    scores = re.findall(pattern=r"\d+.\d+", string=res)

    score_gt = 0.97
    assert (
        float(scores[0]) > score_gt
    ), f"ocr_e2e recognition python: score {scores[0]} is smaller than {score_gt}"


    text = re.findall(pattern=r"SUPER", string=res)

    assert text != [], f'ocr_e2e recognition python:can\'t find "SUPER" in result string'




############################# dataset test

@pytest.mark.fast
def test_ocr_e2e_py_text_rec_precision(device_id):
    cmd = f"""
    python3 ./samples/ocr_e2e/text_rec.py \
    -m /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
    --vdsp_params ./data/configs/crnn_rgbplanar.json \
    --device_id {device_id} \
    --label_file ./data/labels/ocr_rec_dict.txt \
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

    acc_thresh = 0.80

    assert float(acc) > acc_thresh, f"ocr_e2e recognition python:acc {acc} is smaller than {acc_thresh}"

    os.system("rm  cute80_pred.txt")


