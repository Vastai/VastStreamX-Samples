
# 
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 
from run_cmd import run_cmd
import re
import pytest

################# c++ test  #######################
@pytest.mark.fast
def test_buildin_op_prof_cpp_resize(device_id):
    ### SINGLE_OP_RESIZE
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/resize_op.json \
    --device_ids [{device_id}] \
    --instance 6 \
    --iterations 25000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 5900
    assert (
        float(qps) > qps_thresh
    ), f"resize_op c++: performance qps {qps} is smaller than {qps_thresh}"




@pytest.mark.fast
def test_buildin_op_prof_cpp_crop(device_id):
    ### SINGLE_OP_CROP
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/crop_op.json \
    --device_ids [{device_id}] \
    --instance 12 \
    --iterations 150000 \
    --percentiles "[50,90,95,99]" \
    --input_host 0 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 47500
    assert (
        float(qps) > qps_thresh
    ), f"crop_op c++: performance qps {qps} is smaller than {qps_thresh}"




### SINGLE_OP_CVT_COLOR
@pytest.mark.fast
def test_buildin_op_prof_cpp_cvtcolor(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/cvtcolor_op.json \
    --device_ids [{device_id}] \
    --instance 8 \
    --iterations 100000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 13000
    assert (
        float(qps) > qps_thresh
    ), f"cvtcolor_op c++: performance qps {qps} is smaller than {qps_thresh}"




### SINGLE_OP_BATCH_CROP_RESIZE
@pytest.mark.fast
def test_buildin_op_prof_cpp_batch_crop_resize(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/batch_crop_resize_op.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 10000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 2100
    assert (
        float(qps) > qps_thresh
    ), f"batch_crop_resize_op c++: performance qps {qps} is smaller than {qps_thresh}"




### SINGLE_OP_WARP_AFFINE
@pytest.mark.fast
@pytest.mark.ai_integration
def test_buildin_op_prof_cpp_warpaffine(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/warpaffine_op.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 5000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 950
    assert (
        float(qps) > qps_thresh
    ), f"warpaffine_op c++: performance qps {qps} is smaller than {qps_thresh}"




### SINGLE_OP_FLIP
@pytest.mark.fast
def test_buildin_op_prof_cpp_flip(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/flip_op.json \
    --device_ids [{device_id}] \
    --instance 6 \
    --iterations 20000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 3900
    assert (
        float(qps) > qps_thresh
    ), f"flip_op c++: performance qps {qps} is smaller than {qps_thresh}"




### SINGLE_OP_SCALE
@pytest.mark.fast
def test_buildin_op_prof_cpp_scale(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/scale_op.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 15000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 3700
    assert (
        float(qps) > qps_thresh
    ), f"scale_op c++: performance qps {qps} is smaller than {qps_thresh}"




### SINGLE_OP_COPY_MAKE_BORDER
@pytest.mark.fast
def test_buildin_op_prof_cpp_copy_make_boarder(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/copy_make_boarder_op.json \
    --device_ids [{device_id}] \
    --instance 6 \
    --iterations 60000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 10000
    assert (
        float(qps) > qps_thresh
    ), f"copy_make_boarder_op c++: performance qps {qps} is smaller than {qps_thresh}"



### FUSION_OP_YUV_NV12_RESIZE_2RGB_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_cpp_nv12_resize_2rgb(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/nv12_resize_2rgb.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 15000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 2900
    assert (
        float(qps) > qps_thresh
    ), f"nv12_resize_2rgb c++: performance qps {qps} is smaller than {qps_thresh}"



### FUSION_OP_YUV_NV12_CVTCOLOR_RESIZE_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_cpp_nv12_cvtcolor_resize(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/nv12_cvtcolor_resize.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 35000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 6900
    assert (
        float(qps) > qps_thresh
    ), f"nv12_cvtcolor_resize c++: performance qps {qps} is smaller than {qps_thresh}"



### FUSION_OP_YUV_NV12_RESIZE_CVTCOLOR_CROP_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_cpp_nv12_resize_cvtcolor_crop(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/nv12_resize_cvtcolor_crop.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 10000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 1500
    assert (
        float(qps) > qps_thresh
    ), f"nv12_resize_cvtcolor_crop c++: performance qps {qps} is smaller than {qps_thresh}"



### FUSION_OP_YUV_NV12_CROP_CVTCOLOR_RESIZE_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_cpp_nv12_crop_cvtcolor_resize(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/nv12_crop_cvtcolor_resize.json \
    --device_ids [{device_id}] \
    --instance 6 \
    --iterations 30000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 5000
    assert (
        float(qps) > qps_thresh
    ), f"nv12_crop_cvtcolor_resize c++: performance qps {qps} is smaller than {qps_thresh}"



### FUSION_OP_YUV_NV12_CVTCOLOR_RESIZE_CROP_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_cpp_nv12_cvtcolor_resize_crop(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/nv12_cvtcolor_resize_crop.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 30000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 4800
    assert (
        float(qps) > qps_thresh
    ), f"nv12_cvtcolor_resize_crop c++: performance qps {qps} is smaller than {qps_thresh}"




### FUSION_OP_YUV_NV12_CVTCOLOR_LETTERBOX_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_cpp_nv12_cvtcolor_letterbox(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/nv12_cvtcolor_letterbox.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 30000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 7200
    assert (
        float(qps) > qps_thresh
    ), f"nv12_cvtcolor_letterbox c++: performance qps {qps} is smaller than {qps_thresh}"




### FUSION_OP_YUV_NV12_LETTERBOX_2RGB_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_cpp_nv12_letterbox_2rgb(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/nv12_letterbox_2rgb.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 10000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 2100
    assert (
        float(qps) > qps_thresh
    ), f"nv12_letterbox_2rgb c++: performance qps {qps} is smaller than {qps_thresh}"



### FUSION_OP_RGB_CVTCOLOR_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_cpp_rgb_cvtcolor(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/rgb_cvtcolor.json \
    --device_ids [{device_id}] \
    --instance 8 \
    --iterations 100000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 20000
    assert (
        float(qps) > qps_thresh
    ), f"rgb_cvtcolor c++: performance qps {qps} is smaller than {qps_thresh}"



### FUSION_OP_RGB_RESIZE_CVTCOLOR_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_cpp_rgb_resize_cvtcolor(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/rgb_resize_cvtcolor.json \
    --device_ids [{device_id}] \
    --instance 6 \
    --iterations 30000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 7200
    assert (
        float(qps) > qps_thresh
    ), f"rgb_resize_cvtcolor c++: performance qps {qps} is smaller than {qps_thresh}"




### FUSION_OP_RGB_RESIZE_CVTCOLOR_CROP_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_cpp_rgb_resize_cvtcolor_crop(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/rgb_resize_cvtcolor_crop.json \
    --device_ids [{device_id}] \
    --instance 6 \
    --iterations 30000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 5000
    assert (
        float(qps) > qps_thresh
    ), f"rgb_resize_cvtcolor_crop c++: performance qps {qps} is smaller than {qps_thresh}"




### FUSION_OP_RGB_CROP_RESIZE_CVTCOLOR_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_cpp_rgb_crop_resize_cvtcolor(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/rgb_crop_resize_cvtcolor.json \
    --device_ids [{device_id}] \
    --instance 6 \
    --iterations 30000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 5000
    assert (
        float(qps) > qps_thresh
    ), f"rgb_crop_resize_cvtcolor c++: performance qps {qps} is smaller than {qps_thresh}"



### FUSION_OP_RGB_LETTERBOX_CVTCOLOR_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_cpp_rgb_letterbox_cvtcolor(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/rgb_letterbox_cvtcolor.json \
    --device_ids [{device_id}] \
    --instance 6 \
    --iterations 12000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 2100
    assert (
        float(qps) > qps_thresh
    ), f"rgb_letterbox_cvtcolor c++: performance qps {qps} is smaller than {qps_thresh}"



### FUSION_OP_RGB_LETTERBOX_CVTCOLOR_NORM_TENSOR_EXT
@pytest.mark.fast
def test_buildin_op_prof_cpp_rgb_letterbox_cvtcolor_ext(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/buildin_op_prof \
    --op_config ./samples/vdsp_op/buildin_op_prof/rgb_letterbox_cvtcolor_ext.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 8000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 1400
    assert (
        float(qps) > qps_thresh
    ), f"rgb_letterbox_cvtcolor_ext c++: performance qps {qps} is smaller than {qps_thresh}"




############################# Python ##########################

### SINGLE_OP_RESIZE
@pytest.mark.fast
def test_buildin_op_prof_py_resize_op(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/resize_op.json \
    --device_ids [{device_id}] \
    --instance 6 \
    --iterations 50000 \
    --percentiles "[50,90,95,99]" \
    --input_host 0 
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 6300
    assert (
        float(qps) > qps_thresh
    ), f"resize_op python: performance qps {qps} is smaller than {qps_thresh}"



### SINGLE_OP_CROP
@pytest.mark.fast
def test_buildin_op_prof_py_crop_op(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/crop_op.json \
    --device_ids [{device_id}] \
    --instance 12 \
    --iterations 150000 \
    --percentiles "[50,90,95,99]" \
    --input_host 0 
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 46000
    assert (
        float(qps) > qps_thresh
    ), f"crop_op python: performance qps {qps} is smaller than {qps_thresh}"



### SINGLE_OP_CVT_COLOR
@pytest.mark.fast
def test_buildin_op_prof_py_cvtcolor_op(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/cvtcolor_op.json \
    --device_ids [{device_id}] \
    --instance 8 \
    --iterations 100000 \
    --percentiles "[50,90,95,99]" \
    --input_host 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 14000
    assert (
        float(qps) > qps_thresh
    ), f"cvtcolor_op python: performance qps {qps} is smaller than {qps_thresh}"



### SINGLE_OP_BATCH_CROP_RESIZE
@pytest.mark.fast
def test_buildin_op_prof_py_batch_crop_resize_op(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/batch_crop_resize_op.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 10000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 2100
    assert (
        float(qps) > qps_thresh
    ), f"batch_crop_resize_op python: performance qps {qps} is smaller than {qps_thresh}"




### SINGLE_OP_WARP_AFFINE
@pytest.mark.fast
@pytest.mark.ai_integration
def test_buildin_op_prof_py_warpaffine_op(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/warpaffine_op.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 5000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 900
    assert (
        float(qps) > qps_thresh
    ), f"warpaffine_op python: performance qps {qps} is smaller than {qps_thresh}"



### SINGLE_OP_FLIP
@pytest.mark.fast
def test_buildin_op_prof_py_flip_op(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/flip_op.json \
    --device_ids [{device_id}] \
    --instance 6 \
    --iterations 20000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1  
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 3900
    assert (
        float(qps) > qps_thresh
    ), f"flip_op python: performance qps {qps} is smaller than {qps_thresh}"


### SINGLE_OP_SCALE
@pytest.mark.fast
def test_buildin_op_prof_py_scale_op(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/scale_op.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 15000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 3700
    assert (
        float(qps) > qps_thresh
    ), f"scale_op python: performance qps {qps} is smaller than {qps_thresh}"


### SINGLE_OP_COPY_MAKE_BORDER
@pytest.mark.fast
def test_buildin_op_prof_py_copy_make_boarder_op(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/copy_make_boarder_op.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 60000 \
    --percentiles "[50,90,95,99]" \
    --input_host 0  
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 1100
    assert (
        float(qps) > qps_thresh
    ), f"copy_make_boarder_op python: performance qps {qps} is smaller than {qps_thresh}"



### FUSION_OP_YUV_NV12_RESIZE_2RGB_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_py_nv12_resize_2rgb(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/nv12_resize_2rgb.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 15000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 2900
    assert (
        float(qps) > qps_thresh
    ), f"nv12_resize_2rgb python: performance qps {qps} is smaller than {qps_thresh}"



### FUSION_OP_YUV_NV12_CVTCOLOR_RESIZE_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_py_nv12_cvtcolor_resize(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/nv12_cvtcolor_resize.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 35000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 7000
    assert (
        float(qps) > qps_thresh
    ), f"nv12_cvtcolor_resize python: performance qps {qps} is smaller than {qps_thresh}"


### FUSION_OP_YUV_NV12_RESIZE_CVTCOLOR_CROP_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_py_nv12_resize_cvtcolor_crop(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/nv12_resize_cvtcolor_crop.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 10000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 1500
    assert (
        float(qps) > qps_thresh
    ), f"nv12_resize_cvtcolor_crop python: performance qps {qps} is smaller than {qps_thresh}"


### FUSION_OP_YUV_NV12_CROP_CVTCOLOR_RESIZE_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_py_nv12_crop_cvtcolor_resize(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/nv12_crop_cvtcolor_resize.json \
    --device_ids [{device_id}] \
    --instance 6 \
    --iterations 30000 \
    --percentiles "[50,90,95,99]" \
    --input_host 0 
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 5100
    assert (
        float(qps) > qps_thresh
    ), f"nv12_crop_cvtcolor_resize python: performance qps {qps} is smaller than {qps_thresh}"


### FUSION_OP_YUV_NV12_CVTCOLOR_RESIZE_CROP_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_py_nv12_cvtcolor_resize_crop(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/nv12_cvtcolor_resize_crop.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 30000 \
    --percentiles "[50,90,95,99]" \
    --input_host 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 4800
    assert (
        float(qps) > qps_thresh
    ), f"nv12_cvtcolor_resize_crop python: performance qps {qps} is smaller than {qps_thresh}"


### FUSION_OP_YUV_NV12_CVTCOLOR_LETTERBOX_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_py_nv12_cvtcolor_letterbox(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/nv12_cvtcolor_letterbox.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 30000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 7100
    assert (
        float(qps) > qps_thresh
    ), f"nv12_cvtcolor_letterbox python: performance qps {qps} is smaller than {qps_thresh}"


### FUSION_OP_YUV_NV12_LETTERBOX_2RGB_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_py_nv12_letterbox_2rgb(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/nv12_letterbox_2rgb.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 10000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 2100
    assert (
        float(qps) > qps_thresh
    ), f"nv12_letterbox_2rgb python: performance qps {qps} is smaller than {qps_thresh}"


### FUSION_OP_RGB_CVTCOLOR_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_py_rgb_cvtcolor(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/rgb_cvtcolor.json \
    --device_ids [{device_id}] \
    --instance 8 \
    --iterations 100000 \
    --percentiles "[50,90,95,99]" \
    --input_host 0 
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 20000
    assert (
        float(qps) > qps_thresh
    ), f"rgb_cvtcolor python: performance qps {qps} is smaller than {qps_thresh}"


### FUSION_OP_RGB_RESIZE_CVTCOLOR_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_py_rgb_resize_cvtcolor(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/rgb_resize_cvtcolor.json \
    --device_ids [{device_id}] \
    --instance 6 \
    --iterations 30000 \
    --percentiles "[50,90,95,99]" \
    --input_host 1 
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 7400
    assert (
        float(qps) > qps_thresh
    ), f"rgb_resize_cvtcolor python: performance qps {qps} is smaller than {qps_thresh}"


### FUSION_OP_RGB_RESIZE_CVTCOLOR_CROP_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_py_rgb_resize_cvtcolor_crop(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/rgb_resize_cvtcolor_crop.json \
    --device_ids [{device_id}] \
    --instance 6 \
    --iterations 30000 \
    --percentiles "[50,90,95,99]" \
    --input_host 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 5100
    assert (
        float(qps) > qps_thresh
    ), f"rgb_resize_cvtcolor_crop python: performance qps {qps} is smaller than {qps_thresh}"



### FUSION_OP_RGB_CROP_RESIZE_CVTCOLOR_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_py_rgb_crop_resize_cvtcolor(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/rgb_crop_resize_cvtcolor.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 30000 \
    --percentiles "[50,90,95,99]" \
    --input_host 0 
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 5200
    assert (
        float(qps) > qps_thresh
    ), f"rgb_crop_resize_cvtcolor python: performance qps {qps} is smaller than {qps_thresh}"


### FUSION_OP_RGB_LETTERBOX_CVTCOLOR_NORM_TENSOR
@pytest.mark.fast
def test_buildin_op_prof_py_rgb_letterbox_cvtcolor(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/rgb_letterbox_cvtcolor.json \
    --device_ids [{device_id}] \
    --instance 6 \
    --iterations 12000 \
    --percentiles "[50,90,95,99]" \
    --input_host 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 2300
    assert (
        float(qps) > qps_thresh
    ), f"rgb_letterbox_cvtcolor python: performance qps {qps} is smaller than {qps_thresh}"


### FUSION_OP_RGB_LETTERBOX_CVTCOLOR_NORM_TENSOR_EXT
@pytest.mark.fast
def test_buildin_op_prof_py_rgb_letterbox_cvtcolor_ext(device_id):
    cmd = f"""
    python3 samples/vdsp_op/buildin_op_prof/buildin_op_prof.py \
    --op_config ./samples/vdsp_op/buildin_op_prof/rgb_letterbox_cvtcolor_ext.json \
    --device_ids [{device_id}] \
    --instance 5 \
    --iterations 8000 \
    --percentiles "[50,90,95,99]" \
    --input_host 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 1400
    assert (
        float(qps) > qps_thresh
    ), f"rgb_letterbox_cvtcolor_ext python: performance qps {qps} is smaller than {qps_thresh}"

