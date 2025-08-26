
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
# h264 decode
@pytest.mark.codec
@pytest.mark.codec_integration
def test_video_decode_cpp_h264(device_id):
    os.system("mkdir -p output_h264")
    cmd = f"""
    ./build/vaststreamx-samples/bin/video_decode \
    --device_id {device_id} \
    --codec_type h264 \
    --input_file /opt/vastai/vaststreamx/data/videos/h264/1920x1080.h264 \
    --output_folder output_h264
    """
    run_cmd(cmd,False)

    cmd = f"""
    cd output_h264 && ls -all | wc -l 
    """

    res = run_cmd(cmd)

    file_count = int(res)

    assert (
        file_count >= 127
    ), f"video_decode h264 c++: yuv file count {file_count} is smaller than 127"

    os.system("rm -rf output_h264")


# h265 decode
@pytest.mark.codec
@pytest.mark.codec_integration
def test_video_decode_cpp_h265(device_id):
    os.system("mkdir -p output_h265")
    cmd = f"""
    ./build/vaststreamx-samples/bin/video_decode \
    --device_id {device_id} \
    --codec_type h265 \
    --input_file /opt/vastai/vaststreamx/data/videos/hevc/1920x1080.hevc \
    --output_folder output_h265
    """
    run_cmd(cmd,False)

    cmd = f"""
    cd output_h265 && ls -all | wc -l 
    """

    res = run_cmd(cmd)

    file_count = int(res)

    assert (
        file_count >= 127
    ), f"video_decode h265 c++: yuv file count {file_count} is smaller than 127"

    os.system("rm -rf output_h265")



#################  performance test

@pytest.mark.codec
def test_video_decode_cpp_performance_h264(device_id):
    # h264 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/video_decode_prof \
    --device_ids [{device_id}] \
    --input_file /opt/vastai/vaststreamx/data/videos/h264/1920x1080.h264 \
    --codec_type H264  \
    --instance 10 \
    --iterations 10000 
    """
    res = run_cmd(cmd,False)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 1500
    assert (
        float(qps) > qps_thresh
    ), f"video_decode h264 c++:best prof qps {qps} is smaller than {qps_thresh}"

    # H264测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/video_decode_prof \
    --device_ids [{device_id}] \
    --input_file /opt/vastai/vaststreamx/data/videos/h264/1920x1080.h264 \
    --codec_type H264  \
    --instance 1 \
    --iterations 3000 
    """
    res = run_cmd(cmd,False)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 450
    assert (
        float(qps) > qps_thresh
    ), f"video_decode h264 c++:best latency qps {qps} is smaller than {qps_thresh}"

@pytest.mark.codec
def test_video_decode_cpp_performance_h265(device_id):
    # H265测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/video_decode_prof \
    --input_file /opt/vastai/vaststreamx/data/videos/hevc/1920x1080.hevc \
    --device_ids [{device_id}] \
    --codec_type H265 \
    --instance 10 \
    --iterations 10000 
    """
    res = run_cmd(cmd,False)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 1800
    assert (
        float(qps) > qps_thresh
    ), f"video_decode h265 c++:best prof qps {qps} is smaller than {qps_thresh}"

    # H265测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/video_decode_prof \
    --input_file /opt/vastai/vaststreamx/data/videos/hevc/1920x1080.hevc \
    --device_ids [{device_id}] \
    --codec_type H265 \
    --instance 1 \
    --iterations 5000 
    """
    res = run_cmd(cmd,False)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 550
    assert (
        float(qps) > qps_thresh
    ), f"video_decode h265 c++:best latency qps {qps} is smaller than {qps_thresh}"



################# python test  #######################
# h264 decode
@pytest.mark.codec
@pytest.mark.codec_integration
def test_video_decode_py_h264(device_id):
    os.system("mkdir -p output_h264")
    cmd = f"""
    python3 samples/video_decode/video_decode.py \
    --device_id {device_id} \
    --codec_type h264 \
    --input_file /opt/vastai/vaststreamx/data/videos/h264/1920x1080.h264 \
    --output_folder output_h264
    """
    run_cmd(cmd,False)

    cmd = f"""
    cd output_h264 && ls -all | wc -l 
    """

    res = run_cmd(cmd,False)

    file_count = int(res)

    assert (
        file_count >= 127
    ), f"video_decode h264 python: yuv file count {file_count} is smaller than 127"

    os.system("rm -rf output_h264")


# h265 decode
@pytest.mark.codec
@pytest.mark.codec_integration
def test_video_decode_py_h265(device_id):
    os.system("mkdir -p output_h265")
    cmd = f"""
    python3 samples/video_decode/video_decode.py \
    --device_id {device_id} \
    --codec_type h265 \
    --input_file /opt/vastai/vaststreamx/data/videos/hevc/1920x1080.hevc \
    --output_folder output_h265
    """
    run_cmd(cmd)

    cmd = f"""
    cd output_h265 && ls -all | wc -l 
    """

    res = run_cmd(cmd,False)

    file_count = int(res)

    assert (
        file_count >= 127
    ), f"video_decode h265 python: yuv file count {file_count} is smaller than 127"

    os.system("rm -rf output_h265")



#################  performance test

@pytest.mark.codec
def test_video_decode_py_performance_h264(device_id):
    # h264 测试最大吞吐
    cmd = f"""
    python3 samples/video_decode/video_decode_prof.py \
    --device_ids [{device_id}] \
    --input_file /opt/vastai/vaststreamx/data/videos/h264/1920x1080.h264 \
    --codec_type H264  \
    --instance 10 \
    --iterations 10000 
    """
    res = run_cmd(cmd,False)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 1500
    assert (
        float(qps) > qps_thresh
    ), f"video_decode h264 python:best prof qps {qps} is smaller than {qps_thresh}"

    # H264测试最小时延
    cmd = f"""
    python3 samples/video_decode/video_decode_prof.py \
    --device_ids [{device_id}] \
    --input_file /opt/vastai/vaststreamx/data/videos/h264/1920x1080.h264 \
    --codec_type H264  \
    --instance 1 \
    --iterations 3000 
    """
    res = run_cmd(cmd,False)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 450
    assert (
        float(qps) > qps_thresh
    ), f"video_decode h264 python:best latency qps {qps} is smaller than {qps_thresh}"

@pytest.mark.codec
def test_video_decode_py_performance_h265(device_id):
    # H265测试最大吞吐
    cmd = f"""
    python3 samples/video_decode/video_decode_prof.py \
    --input_file /opt/vastai/vaststreamx/data/videos/hevc/1920x1080.hevc \
    --device_ids [{device_id}] \
    --codec_type H265 \
    --instance 10 \
    --iterations 10000 
    """
    res = run_cmd(cmd,False)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 1800
    assert (
        float(qps) > qps_thresh
    ), f"video_decode h265 python:best prof qps {qps} is smaller than {qps_thresh}"

    # H265测试最小时延
    cmd = f"""
    python3 samples/video_decode/video_decode_prof.py \
    --input_file /opt/vastai/vaststreamx/data/videos/hevc/1920x1080.hevc \
    --device_ids [{device_id}] \
    --codec_type H265 \
    --instance 1 \
    --iterations 5000 
    """
    res = run_cmd(cmd,False)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 550
    assert (
        float(qps) > qps_thresh
    ), f"video_decode h265 python:best latency qps {qps} is smaller than {qps_thresh}"

