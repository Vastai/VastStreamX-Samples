
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


#####################  python test #####################


##############################  one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_elic_inference_py(device_id):
    cmd = f"""
    python3 ./samples/elic/elic/elic_inference.py \
    --gaha_model_prefix  /opt/vastai/vaststreamx/data/models/elic/gaha-fp16-none-1_3_512_512-vacc/mod \
    --gaha_vdsp_params  ./data/configs/elic_compress_gaha_rgbplanar.json \
    --hs_model_prefix  /opt/vastai/vaststreamx/data/models/elic/hs_chunk-fp16-none-1_192_8_8/mod \
    --gs_model_prefix   /opt/vastai/vaststreamx/data/models/elic/gs-fp16-none-1_320_32_32/mod \
    --torch_model  /opt/vastai/vaststreamx/data/pre-trained/ELIC_0450_ft_3980_Plateau.pth.tar \
    --tensorize_elf_path /opt/vastai/vaststreamx/data/elf/tensorize_ext_op \
    --device_id {device_id} \
    --input_file  /opt/vastai/vaststreamx/data/datasets/Kodak-512/kodim01.png \
    --output_file  elic_result.png
    """

    run_cmd(cmd,False)

    assert os.path.exists(
        "elic_result.png"
    ), "elic python:can't find elic_result.png"

    os.system("rm elic_result.png")

############################## dataset  test

@pytest.mark.fast
def test_elic_inference_py_precision(device_id):
    cmd = f"""
    python3 ./samples/elic/elic/elic_inference.py \
    --gaha_model_prefix  /opt/vastai/vaststreamx/data/models/elic/gaha-fp16-none-1_3_512_512-vacc/mod \
    --gaha_vdsp_params  ./data/configs/elic_compress_gaha_rgbplanar.json \
    --hs_model_prefix  /opt/vastai/vaststreamx/data/models/elic/hs_chunk-fp16-none-1_192_8_8/mod \
    --gs_model_prefix /opt/vastai/vaststreamx/data/models/elic/gs-fp16-none-1_320_32_32/mod \
    --torch_model  /opt/vastai/vaststreamx/data/pre-trained/ELIC_0450_ft_3980_Plateau.pth.tar \
    --tensorize_elf_path /opt/vastai/vaststreamx/data/elf/tensorize_ext_op \
    --device_id {device_id} \
    --dataset_path /opt/vastai/vaststreamx/data/datasets/Kodak-512/ \
    --dataset_output_path dataset_outputs
    """ 
    res =  run_cmd(cmd,False)


    compress_time =re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"Compress time:\d+.\d+", string=res).group()
    ).group()
    decompress_time = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"Decompress time:\d+.\d+", string=res).group()
    ).group()
    pnsr = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"PNSR:\d+.\d+", string=res).group()
    ).group()

    
    compress_time_thresh = 800
    decompress_time_thresh = 500
    pnsr_thresh = 37.6 


    assert (
        float(compress_time) < compress_time_thresh
    ), f"elic python :compress_time {compress_time} is larger than {compress_time_thresh}"

    assert (
        float(decompress_time) < decompress_time_thresh
    ), f"elic python :decompress_time {decompress_time} is larger than {decompress_time_thresh}"

    assert (
        float(pnsr) > pnsr_thresh
    ), f"elic python :pnsr {pnsr} is smaller than {pnsr_thresh}"

    os.system("rm -rf dataset_outputs")




############################## dynamic one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_dynamic_elic_inference_py(device_id):
    cmd = f"""
    python3 ./samples/elic/elic/dynamic_elic_inference.py \
    --gaha_model_info /opt/vastai/vaststreamx/data/models/elic/gaha-dynamic/gaha-dynamic_module_info.json \
    --gaha_vdsp_params  ./data/configs/elic_compress_gaha_rgbplanar.json \
    --hs_model_info /opt/vastai/vaststreamx/data/models/elic/hs_chunk-dynamic/hs_chunk-dynamic_module_info.json \
    --gs0_model_info /opt/vastai/vaststreamx/data/models/elic/gs0-dynamic/gs0-dynamic_module_info.json \
    --gs_model_info /opt/vastai/vaststreamx/data/models/elic/gs-dynamic/gs-dynamic_module_info.json \
    --torch_model  /opt/vastai/vaststreamx/data/pre-trained/ELIC_0450_ft_3980_Plateau.pth.tar \
    --tensorize_elf_path /opt/vastai/vaststreamx/data/elf/tensorize_ext_op \
    --device_id {device_id} \
    --input_file  /opt/vastai/vaststreamx/data/datasets/Kodak/kodim01.png \
    --output_file  elic_dynamic_result.png
    """

    run_cmd(cmd,False)

    assert os.path.exists(
        "elic_dynamic_result.png"
    ), "dynamic elic python:can't find elic_dynamic_result.png"

    os.system("rm elic_dynamic_result.png")




############################## dataset  test

@pytest.mark.fast
def test_dynamic_elic_inference_py_precision(device_id):
    cmd = f"""
    python3 ./samples/elic/elic/dynamic_elic_inference.py \
    --gaha_model_info /opt/vastai/vaststreamx/data/models/elic/gaha-dynamic/gaha-dynamic_module_info.json \
    --gaha_vdsp_params  ./data/configs/elic_compress_gaha_rgbplanar.json \
    --hs_model_info /opt/vastai/vaststreamx/data/models/elic/hs_chunk-dynamic/hs_chunk-dynamic_module_info.json \
    --gs0_model_info /opt/vastai/vaststreamx/data/models/elic/gs0-dynamic/gs0-dynamic_module_info.json \
    --gs_model_info /opt/vastai/vaststreamx/data/models/elic/gs-dynamic/gs-dynamic_module_info.json \
    --torch_model  /opt/vastai/vaststreamx/data/pre-trained/ELIC_0450_ft_3980_Plateau.pth.tar \
    --tensorize_elf_path /opt/vastai/vaststreamx/data/elf/tensorize_ext_op \
    --device_id {device_id} \
    --dataset_path /opt/vastai/vaststreamx/data/datasets/Kodak/ \
    --dataset_output_path dataset_outputs
    """ 
    res =  run_cmd(cmd,False)

    compress_time =re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"Compress time:\d+.\d+", string=res).group()
    ).group()
    decompress_time = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"Decompress time:\d+.\d+", string=res).group()
    ).group()
    pnsr = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"PNSR:\d+.\d+", string=res).group()
    ).group()

    compress_time_thresh = 600
    decompress_time_thresh = 500
    pnsr_thresh = 37.4 

    assert (
        float(compress_time) < compress_time_thresh
    ), f"dynamic elic python :compress_time {compress_time} is larger than {compress_time_thresh}"

    assert (
        float(decompress_time) < decompress_time_thresh
    ), f"dynamic elic python :decompress_time {decompress_time} is larger than {decompress_time_thresh}"

    assert (
        float(pnsr) > pnsr_thresh
    ), f"dynamic elic python :pnsr {pnsr} is smaller than {pnsr_thresh}"

    os.system("rm -rf dataset_outputs")



 

############################## no_entropy 512x512 one image  test

@pytest.mark.fast
@pytest.mark.ai_integration
def test_elic_no_entropy_inference_py_512_512(device_id):
    cmd = f"""
    python3 ./samples/elic/elic/elic_no_entropy_inference.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/elic/elic_no_entropy-512_512/mod \
    --vdsp_params  ./data/configs/elic_compress_gaha_rgbplanar.json \
    --device_id {device_id} \
    --input_file  /opt/vastai/vaststreamx/data/datasets/Kodak-512/kodim01.png \
    --output_file  elic_no_entropy_512x512.png
    """

    run_cmd(cmd,False)

    assert os.path.exists(
        "elic_no_entropy_512x512.png"
    ), "no_entropy elic python:can't find elic_no_entropy_512x512.png"

    os.system("rm elic_no_entropy_512x512.png")


############################## no_entropy 1280x2048 one image  test

@pytest.mark.fast
@pytest.mark.ai_integration
def test_elic_no_entropy_inference_py_1280_2048(device_id):
    cmd = f"""
    python3 ./samples/elic/elic/elic_no_entropy_inference.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/elic/elic_no_entropy-1280_2048/mod \
    --vdsp_params  ./data/configs/elic_compress_gaha_rgbplanar.json \
    --device_id {device_id} \
    --input_file  /opt/vastai/vaststreamx/data/datasets/Kodak_1280_2048/kodim01.png \
    --output_file  elic_no_entropy_1280x2048.png
    """

    run_cmd(cmd,False)

    assert os.path.exists(
        "elic_no_entropy_1280x2048.png"
    ), "no_entropy elic python:can't find elic_no_entropy_1280x2048.png"

    os.system("rm elic_no_entropy_1280x2048.png")




############################## 512x512 dataset  test

@pytest.mark.fast
def test_elic_no_entropy_inference_py_512_512_precision(device_id):
    cmd = f"""
    python3 ./samples/elic/elic/elic_no_entropy_inference.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/elic/elic_no_entropy-512_512/mod \
    --vdsp_params  ./data/configs/elic_compress_gaha_rgbplanar.json \
    --device_id {device_id} \
    --dataset_path /opt/vastai/vaststreamx/data/datasets/Kodak-512/ \
    --dataset_output_path dataset_outputs_512x512
    """ 
    res =  run_cmd(cmd,False)

    compress_time =re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"Compress time:\d+.\d+", string=res).group()
    ).group()
    bbp = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"bbp:\d+.\d+", string=res).group()
    ).group()
    pnsr = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"PNSR:\d+.\d+", string=res).group()
    ).group()

    compress_time_thresh = 200
    bbp_thresh = 0.84
    pnsr_thresh = 37.65 

    assert (
        float(compress_time) < compress_time_thresh
    ), f"no_entropy 512x512 elic python :compress_time {compress_time} is larger than {compress_time_thresh}"

    assert (
        float(bbp) > bbp_thresh
    ), f"no_entropy 512x512 elic python :bbp {bbp} is smaller than {bbp_thresh}"

    assert (
        float(pnsr) > pnsr_thresh
    ), f"no_entropy 512x512 elic python :pnsr {pnsr} is smaller than {pnsr_thresh}"

    os.system("rm -rf dataset_outputs_512x512")





############################## 1280x2048 dataset  test

@pytest.mark.fast
def test_elic_no_entropy_inference_py_1280_2048_precision(device_id):
    cmd = f"""
    python3 ./samples/elic/elic/elic_no_entropy_inference.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/elic/elic_no_entropy-1280_2048/mod \
    --vdsp_params  ./data/configs/elic_compress_gaha_rgbplanar.json \
    --device_id {device_id} \
    --dataset_path /opt/vastai/vaststreamx/data/datasets/Kodak_1280_2048/ \
    --dataset_output_path dataset_outputs_1280x2048
    """ 
    res =  run_cmd(cmd,False)

    compress_time =re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"Compress time:\d+.\d+", string=res).group()
    ).group()
    bbp = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"bbp:\d+.\d+", string=res).group()
    ).group()
    pnsr = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"PNSR:\d+.\d+", string=res).group()
    ).group()

    compress_time_thresh = 2000
    bbp_thresh = 0.26
    pnsr_thresh = 42.99 

    assert (
        float(compress_time) < compress_time_thresh
    ), f"no_entropy 1280x2048 elic python :compress_time {compress_time} is larger than {compress_time_thresh}"

    assert (
        float(bbp) > bbp_thresh
    ), f"no_entropy 1280x2048 elic python :bbp {bbp} is smaller than {bbp_thresh}"

    assert (
        float(pnsr) > pnsr_thresh
    ), f"no_entropy 1280x2048 elic python :pnsr {pnsr} is smaller than {pnsr_thresh}"

    os.system("rm -rf dataset_outputs_1280x2048")



#################  performance test

@pytest.mark.fast
def test_elic_no_entropy_inference_py_512_512_performance(device_id):
    # 测试最大吞吐 512x512 模型
    cmd = f"""
    python ./samples/elic/elic/elic_noentropy_prof.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/elic/elic_no_entropy-512_512/mod \
    --vdsp_params  ./data/configs/elic_compress_gaha_rgbplanar.json \
    --device_ids [{device_id}] \
    --instance 1 \
    --iterations 100 \
    --queue_size 1 \
    --batch_size 1
    """ 
    res = run_cmd(cmd,False)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 9
    assert (
        float(qps) > qps_thresh
    ), f"elic_no_entropy 512x512 python:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延 512x512 模型
    cmd = f"""
    python ./samples/elic/elic/elic_noentropy_prof.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/elic/elic_no_entropy-512_512/mod \
    --vdsp_params  ./data/configs/elic_compress_gaha_rgbplanar.json \
    --device_ids [{device_id}] \
    --instance 1 \
    --iterations 100 \
    --batch_size 1 \
    --queue_size 0
    """ 
    res = run_cmd(cmd,False)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 9
    assert (
        float(qps) > qps_thresh
    ), f"elic_no_entropy 512x512 python:best latancy qps {qps} is smaller than {qps_thresh}"



@pytest.mark.fast
def test_elic_no_entropy_inference_py_1280_2048_performance(device_id):
    # 测试最大吞吐 1280x2048 模型
    cmd = f"""
    python ./samples/elic/elic/elic_noentropy_prof.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/elic/elic_no_entropy-1280_2048/mod \
    --vdsp_params  ./data/configs/elic_compress_gaha_rgbplanar.json \
    --device_ids [{device_id}] \
    --instance 1 \
    --iterations 10 \
    --batch_size 1  \
    --queue_size 1
    """ 
    res = run_cmd(cmd,False)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 0.795
    assert (
        float(qps) > qps_thresh
    ), f"elic_no_entropy 1280x2048 python:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延 1280x2048 模型
    cmd = f"""
    python ./samples/elic/elic/elic_noentropy_prof.py \
    --model_prefix /opt/vastai/vaststreamx/data/models/elic/elic_no_entropy-1280_2048/mod \
    --vdsp_params  ./data/configs/elic_compress_gaha_rgbplanar.json \
    --device_ids [{device_id}] \
    --instance 1 \
    --iterations 10 \
    --batch_size 1  \
    --queue_size 0
    """ 
    res = run_cmd(cmd,False)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 0.745
    assert (
        float(qps) > qps_thresh
    ), f"elic_no_entropy 1280x2048 python:best latancy qps {qps} is smaller than {qps_thresh}"
