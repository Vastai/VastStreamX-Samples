
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
# h264 encode 
@pytest.mark.codec
@pytest.mark.codec_integration
def test_video_writer_cpp_h264(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/video_writer \
    --device_id {device_id} \
    --input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
    --width 1920 \
    --height 1080 \
    --frame_rate 30 \
    --codec_type h264 \
    --output_uri ./test_h264.ts
    """
    res = run_cmd(cmd,False)

    assert os.path.exists("test_h264.ts"), "video_writer c++:can't find test_h264.ts"

    os.system("rm test_h264.ts")


# h265 encode 
@pytest.mark.codec
@pytest.mark.codec_integration
def test_video_writer_cpp_h265(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/video_writer \
    --device_id {device_id} \
    --input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
    --width 1920 \
    --height 1080 \
    --frame_rate 30 \
    --codec_type h265 \
    --output_uri ./test_h265.ts
    """
    res = run_cmd(cmd,False)

    assert os.path.exists("test_h265.ts"), "video_writer c++:can't find test_h265.ts"

    os.system("rm test_h265.ts")




################# python test  #######################

# h264 encode 
@pytest.mark.codec
@pytest.mark.codec_integration
def test_video_writer_py_h264(device_id):
    cmd = f"""
    python3 ./samples/video_writer/video_writer.py \
    --device_id {device_id} \
    --input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
    --width 1920 \
    --height 1080 \
    --frame_rate 30 \
    --codec_type h264 \
    --output_uri ./test_h264.ts
    """
    res = run_cmd(cmd,False)

    assert os.path.exists("test_h264.ts"), "video_writer python:can't find test_h264.ts"

    os.system("rm test_h264.ts")


# h265 encode 
@pytest.mark.codec
@pytest.mark.codec_integration
def test_video_writer_py_h265(device_id):
    cmd = f"""
    python3 ./samples/video_writer/video_writer.py \
    --device_id {device_id} \
    --input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
    --width 1920 \
    --height 1080 \
    --frame_rate 30 \
    --codec_type h265 \
    --output_uri ./test_h265.ts
    """
    res = run_cmd(cmd,False)

    assert os.path.exists("test_h265.ts"), "video_writer python:can't find test_h265.ts"

    os.system("rm test_h265.ts")


