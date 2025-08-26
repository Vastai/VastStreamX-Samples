
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
@pytest.mark.codec
def test_decode_and_detection_cpp(device_id):
    os.system("mkdir -p dec_det_out")

    cmd = f"""
    ./build/vaststreamx-samples/bin/decode_and_detection \
    -m /opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolo_div255_yuv_nv12.json \
    --device_id {device_id} \
    --threshold 0.5 \
    --uri ./data/videos/test.mp4 \
    --output_path dec_det_out \
    --num_channels 1
    """

    run_cmd(cmd, False)

    cmd = f"""
    ls -all dec_det_out | wc -l
    """

    res = run_cmd(cmd)

    num = int(res) - 3

    actual_frames = 299

    assert num >= actual_frames, f"decode_and_detection c++: files count {num} is smaller than {actual_frames},res={res}"

    os.system("rm -rf dec_det_out")


################# python test  #######################
@pytest.mark.codec
def test_decode_and_detection_py(device_id):
    os.system("mkdir -p dec_det_out")

    cmd = f"""
    python3 ./samples/decode_and_detection/decode_and_detection.py  \
    -m /opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolo_div255_yuv_nv12.json \
    --device_id {device_id} \
    --threshold 0.5 \
    --uri ./data/videos/test.mp4 \
    --output_path dec_det_out \
    --num_channels 1
    """


    run_cmd(cmd, False)

    cmd = f"""
    ls -all dec_det_out | wc -l
    """

    res = run_cmd(cmd)

    num = int(res) - 3

    actual_frames = 299

    assert num >= actual_frames, f"decode_and_detection python: files count {num} is smaller than {actual_frames},res={res}"

    os.system("rm -rf dec_det_out")

