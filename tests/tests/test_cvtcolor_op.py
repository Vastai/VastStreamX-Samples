
# 
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 
from run_cmd import run_cmd
import os
import pytest

################# c++ test  #######################
@pytest.mark.fast
@pytest.mark.ai_integration
def test_cvtcolor_cpp_yuv2rgb_nv12(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/cvtcolor \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code YUV2RGB_NV12 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor c++ YUV2RGB_NV12 :can't find cvtcolor_result.jpg"
    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_cpp_yuv2bgr_nv12(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/cvtcolor \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code YUV2BGR_NV12 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor c++ YUV2BGR_NV12 :can't find cvtcolor_result.jpg"
    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_cpp_bgr2rgb(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/cvtcolor \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code BGR2RGB 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor c++ BGR2RGB :can't find cvtcolor_result.jpg"

    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_cpp_rgb2bgr(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/cvtcolor \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code RGB2BGR 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor c++ RGB2BGR :can't find cvtcolor_result.jpg"

    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_cpp_bgr2rgb_interleave2planar(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/cvtcolor \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code BGR2RGB_INTERLEAVE2PLANAR 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor c++ BGR2RGB_INTERLEAVE2PLANAR :can't find cvtcolor_result.jpg"
    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_cpp_rgb2bgr_interleave2planar(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/cvtcolor \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code RGB2BGR_INTERLEAVE2PLANAR 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor c++ RGB2BGR_INTERLEAVE2PLANAR :can't find cvtcolor_result.jpg"

    os.system("rm cvtcolor_result.jpg")
    
@pytest.mark.fast
def test_cvtcolor_cpp_bgr2rgb_interleave2planar(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/cvtcolor \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code BGR2BGR_INTERLEAVE2PLANAR 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor c++ BGR2BGR_INTERLEAVE2PLANAR :can't find cvtcolor_result.jpg"

    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_cpp_rgb2rgb_interleave2planar(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/cvtcolor \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code RGB2RGB_INTERLEAVE2PLANAR 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor c++ RGB2RGB_INTERLEAVE2PLANAR :can't find cvtcolor_result.jpg"
    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_cpp_yuv2gray_nv12(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/cvtcolor \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code YUV2GRAY_NV12 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor c++ YUV2GRAY_NV12 :can't find cvtcolor_result.jpg"
    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_cpp_bgr2gray_interleave(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/cvtcolor \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code BGR2GRAY_INTERLEAVE 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor c++ BGR2GRAY_INTERLEAVE :can't find cvtcolor_result.jpg"
    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_cpp_bgr2gray_planar(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/cvtcolor \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code BGR2GRAY_PLANAR 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor c++ BGR2GRAY_PLANAR :can't find cvtcolor_result.jpg"
    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_cpp_rgb2gray_interleave(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/cvtcolor \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code  RGB2GRAY_INTERLEAVE
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor c++ RGB2GRAY_INTERLEAVE :can't find cvtcolor_result.jpg"
    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_cpp_rgb2gray_planar(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/cvtcolor \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code  RGB2GRAY_PLANAR
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor c++ RGB2GRAY_PLANAR :can't find cvtcolor_result.jpg"

    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_cpp_rgb2yuv_nv12_planar(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/cvtcolor \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code  RGB2YUV_NV12_PLANAR
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor c++ RGB2YUV_NV12_PLANAR :can't find cvtcolor_result.jpg"

    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_cpp_bgr2yuv_nv12_planar(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/cvtcolor \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code  BGR2YUV_NV12_PLANAR
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor c++ BGR2YUV_NV12_PLANAR :can't find cvtcolor_result.jpg"

    os.system("rm cvtcolor_result.jpg")


########################### Python test ###########################
@pytest.mark.fast
@pytest.mark.ai_integration
def test_cvtcolor_py_yuv2rgb_nv12(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/cvtcolor/cvtcolor.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code YUV2RGB_NV12 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor python YUV2RGB_NV12 :can't find cvtcolor_result.jpg"
    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_py_yuv2bgr_nv12(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/cvtcolor/cvtcolor.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code YUV2BGR_NV12 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor python YUV2BGR_NV12 :can't find cvtcolor_result.jpg"
    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_py_bgr2rgb(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/cvtcolor/cvtcolor.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code BGR2RGB 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor python BGR2RGB :can't find cvtcolor_result.jpg"

    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_py_rgb2bgr(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/cvtcolor/cvtcolor.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code RGB2BGR 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor python RGB2BGR :can't find cvtcolor_result.jpg"

    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_py_bgr2rgb_interleave2planar(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/cvtcolor/cvtcolor.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code BGR2RGB_INTERLEAVE2PLANAR 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor python BGR2RGB_INTERLEAVE2PLANAR :can't find cvtcolor_result.jpg"
    os.system("rm cvtcolor_result.jpg")


@pytest.mark.fast
def test_cvtcolor_py_rgb2bgr_interleave2planar(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/cvtcolor/cvtcolor.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code RGB2BGR_INTERLEAVE2PLANAR 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor python RGB2BGR_INTERLEAVE2PLANAR :can't find cvtcolor_result.jpg"

    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_py_bgr2rgb_interleave2planar(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/cvtcolor/cvtcolor.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code BGR2BGR_INTERLEAVE2PLANAR 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor python BGR2BGR_INTERLEAVE2PLANAR :can't find cvtcolor_result.jpg"

    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_py_rgb2rgb_interleave2planar(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/cvtcolor/cvtcolor.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code RGB2RGB_INTERLEAVE2PLANAR 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor python RGB2RGB_INTERLEAVE2PLANAR :can't find cvtcolor_result.jpg"
    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_py_yuv2gray_nv12(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/cvtcolor/cvtcolor.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code YUV2GRAY_NV12 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor python YUV2GRAY_NV12 :can't find cvtcolor_result.jpg"
    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_py_bgr2gray_interleave(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/cvtcolor/cvtcolor.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code BGR2GRAY_INTERLEAVE 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor python BGR2GRAY_INTERLEAVE :can't find cvtcolor_result.jpg"
    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_py_bgr2gray_planar(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/cvtcolor/cvtcolor.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code BGR2GRAY_PLANAR 
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor python BGR2GRAY_PLANAR :can't find cvtcolor_result.jpg"
    os.system("rm cvtcolor_result.jpg")


@pytest.mark.fast
def test_cvtcolor_py_rgb2gray_interleave(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/cvtcolor/cvtcolor.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code  RGB2GRAY_INTERLEAVE
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor python RGB2GRAY_INTERLEAVE :can't find cvtcolor_result.jpg"
    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_py_rgb2gray_planar(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/cvtcolor/cvtcolor.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code  RGB2GRAY_PLANAR
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor python RGB2GRAY_PLANAR :can't find cvtcolor_result.jpg"

    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_py_rgb2yuv_nv12_planar(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/cvtcolor/cvtcolor.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code  RGB2YUV_NV12_PLANAR
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor python RGB2YUV_NV12_PLANAR :can't find cvtcolor_result.jpg"

    os.system("rm cvtcolor_result.jpg")

@pytest.mark.fast
def test_cvtcolor_py_bgr2yuv_nv12_planar(device_id):
    cmd = f"""
    python3 ./samples/vdsp_op/cvtcolor/cvtcolor.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file cvtcolor_result.jpg \
    --cvtcolor_code  BGR2YUV_NV12_PLANAR
    """
    run_cmd(cmd)

    assert os.path.exists(
        "cvtcolor_result.jpg"
    ), "cvtcolor python BGR2YUV_NV12_PLANAR :can't find cvtcolor_result.jpg"

    os.system("rm cvtcolor_result.jpg")

