
# 
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 
from run_cmd import run_cmd
import pytest
import os

################# c++ test  #######################
@pytest.mark.codec
def test_video_capture_cpp(device_id):
    os.system("mkdir -p video_capture_output")
    cmd = f"""
    ./build/vaststreamx-samples/bin/video_capture \
    --device_id {device_id} \
    --input_uri ./data/videos/test.mp4 \
    --frame_count 10 \
    --output_folder video_capture_output
    """
    run_cmd(cmd, False)

    cmd = f"""
    ls -all video_capture_output | wc -l
    """

    res = run_cmd(cmd)

    num = int(res) - 3

    assert num >= 10, f"video_capture c++: files count {num} is smaller than 10,res={res}"

    os.system("rm -rf video_capture_output")


################# python test  #######################
@pytest.mark.codec
def test_video_capture_py(device_id):
    os.system("mkdir -p video_capture_output")
    cmd = f"""
    python3 samples/video_capture/video_capture.py  \
    --device_id {device_id} \
    --input_uri ./data/videos/test.mp4 \
    --frame_count 10 \
    --output_folder video_capture_output
    """
    run_cmd(cmd, False)

    cmd = f"""
    ls -all video_capture_output | wc -l
    """

    res = run_cmd(cmd)

    num = int(res) - 3

    assert (
        num >= 10
    ), f"video_capture python: files count {num} is smaller than 10,res={res}"

    os.system("rm -rf video_capture_output")

