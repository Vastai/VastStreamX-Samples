
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
# h264 encode
@pytest.mark.codec
@pytest.mark.codec_integration
def test_video_encode_cpp_h264(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/video_encode \
    --device_id {device_id} \
    --input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
    --width 1920 \
    --height 1080 \
    --frame_rate 30 \
    --codec_type h264 \
    --output_file output.h264 
    """

    run_cmd(cmd)

    assert os.path.exists("output.h264"), "video_encode c++:can't find output.h264"

    os.system("rm output.h264")


@pytest.mark.codec
@pytest.mark.codec_integration
def test_video_encode_cpp_h265(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/video_encode \
    --device_id {device_id} \
    --input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
    --width 1920 \
    --height 1080 \
    --frame_rate 30 \
    --codec_type h265 \
    --output_file output.h265 
    """

    run_cmd(cmd)

    assert os.path.exists("output.h265"), "video_encode c++:can't find output.h265"

    os.system("rm output.h265")


#################  performance test

@pytest.mark.codec
def test_video_encode_cpp_performance_h264(device_id):
    # h264 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/video_encode_prof \
    --device_ids [{device_id}] \
    --codec_type H264  \
    --percentiles "[50,90,95,99]" \
    --input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
    --width 1920 \
    --height 1080 \
    --frame_rate 30 \
    --instance 4 \
    --iterations 1000
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 170
    assert (
        float(qps) > qps_thresh
    ), f"video_encode h264 c++:best prof qps {qps} is smaller than {qps_thresh}"

    # H264测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/video_encode_prof \
    --device_ids [{device_id}] \
    --codec_type H264  \
    --percentiles "[50,90,95,99]" \
    --input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
    --width 1920 \
    --height 1080 \
    --frame_rate 30 \
    --instance 1 \
    --iterations 500
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 45
    assert (
        float(qps) > qps_thresh
    ), f"video_encode h264 c++:best latency qps {qps} is smaller than {qps_thresh}"


@pytest.mark.codec
def test_video_encode_cpp_performance_h264(device_id):
    # h265 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/video_encode_prof \
    --device_ids [{device_id}] \
    --codec_type H265  \
    --percentiles "[50,90,95,99]" \
    --input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
    --width 1920 \
    --height 1080 \
    --frame_rate 30 \
    --instance 4 \
    --iterations 1000
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 75
    assert (
        float(qps) > qps_thresh
    ), f"video_encode h265 c++:best prof qps {qps} is smaller than {qps_thresh}"

    # H265测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/video_encode_prof \
    --device_ids [{device_id}] \
    --codec_type H265  \
    --percentiles "[50,90,95,99]" \
    --input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
    --width 1920 \
    --height 1080 \
    --frame_rate 30 \
    --instance 1 \
    --iterations 500
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 18
    assert (
        float(qps) > qps_thresh
    ), f"video_encode h265 c++:best latency qps {qps} is smaller than {qps_thresh}"


################# python test  #######################
# h264 encode
@pytest.mark.codec
@pytest.mark.codec_integration
def test_video_encode_py_h264(device_id):
    cmd = f"""
    python3 samples/video_encode/video_encode.py \
    --device_id {device_id} \
    --input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
    --width 1920 \
    --height 1080 \
    --frame_rate 30 \
    --codec_type h264 \
    --output_file output.h264 
    """
    run_cmd(cmd)

    assert os.path.exists("output.h264"), "video_encode python:can't find output.h264"

    os.system("rm output.h264")


@pytest.mark.codec
@pytest.mark.codec_integration
def test_video_encode_py_h265(device_id):
    cmd = f"""
    python3 samples/video_encode/video_encode.py \
    --device_id {device_id} \
    --input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
    --width 1920 \
    --height 1080 \
    --frame_rate 30 \
    --codec_type h265 \
    --output_file output.h265 
    """
    run_cmd(cmd)

    assert os.path.exists("output.h265"), "video_encode python:can't find output.h265"

    os.system("rm output.h265")


#################  performance test

@pytest.mark.codec
def test_video_encode_py_performance_h264(device_id):
    # h264 测试最大吞吐
    cmd = f"""
    python3 samples/video_encode/video_encode_prof.py \
    --device_ids [{device_id}] \
    --codec_type H264  \
    --percentiles "[50,90,95,99]" \
    --input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
    --width 1920 \
    --height 1080 \
    --frame_rate 30 \
    --instance 4 \
    --iterations 1000
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 170
    assert (
        float(qps) > qps_thresh
    ), f"video_encode h264 python:best prof qps {qps} is smaller than {qps_thresh}"

    # H264测试最小时延
    cmd = f"""
    python3 samples/video_encode/video_encode_prof.py \
    --device_ids [{device_id}] \
    --codec_type H264  \
    --percentiles "[50,90,95,99]" \
    --input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
    --width 1920 \
    --height 1080 \
    --frame_rate 30 \
    --instance 1 \
    --iterations 500
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 45
    assert (
        float(qps) > qps_thresh
    ), f"video_encode h264 python:best latency qps {qps} is smaller than {qps_thresh}"


@pytest.mark.codec
def test_video_encode_py_performance_h265(device_id):
    # h265 测试最大吞吐
    cmd = f"""
    python3 samples/video_encode/video_encode_prof.py \
    --device_ids [{device_id}] \
    --codec_type H265  \
    --percentiles "[50,90,95,99]" \
    --input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
    --width 1920 \
    --height 1080 \
    --frame_rate 30 \
    --instance 4 \
    --iterations 1000
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 75
    assert (
        float(qps) > qps_thresh
    ), f"video_encode h265 python:best prof qps {qps} is smaller than {qps_thresh}"

    # H265测试最小时延
    cmd = f"""
    python3 samples/video_encode/video_encode_prof.py \
    --device_ids [{device_id}] \
    --codec_type H265  \
    --percentiles "[50,90,95,99]" \
    --input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
    --width 1920 \
    --height 1080 \
    --frame_rate 30 \
    --instance 1 \
    --iterations 500
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 18
    assert (
        float(qps) > qps_thresh
    ), f"video_encode h265 python:best latency qps {qps} is smaller than {qps_thresh}"

