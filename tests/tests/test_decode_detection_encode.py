
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
def test_decode_detection_encode_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/decode_detection_encode \
    -m /opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolo_div255_yuv_nv12.json \
    --device_id {device_id} \
    --threshold  0.5 \
    --label_file ./data/labels/coco2id.txt \
    --input_uri  ./data/videos/test.mp4 \
    --output_file  out.mp4
    """


    res = run_cmd(cmd,False)

    assert os.path.exists("out.mp4"), "decode_detection_encode c++:can't find out.mp4"

    os.system("rm out.mp4")


################# python test  #######################

@pytest.mark.codec
def test_decode_detection_encode_py(device_id):
    cmd = f"""
    python3 ./samples/decode_detection_encode/decode_detection_encode.py \
    -m /opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
    --vdsp_params ./data/configs/yolo_div255_yuv_nv12.json \
    --device_id {device_id} \
    --threshold  0.5 \
    --label_file ./data/labels/coco2id.txt \
    --input_uri  ./data/videos/test.mp4 \
    --output_file  out.mp4
    """

    res = run_cmd(cmd,False)

    assert os.path.exists("out.mp4"), "decode_detection_encode python:can't find out.mp4"

    os.system("rm out.mp4")
