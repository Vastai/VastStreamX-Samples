
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

################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_face_recognition_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/face_recognition \
    -m /opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod \
    --vdsp_params ./data/configs/facenet_bgr888.json \
    --device_id {device_id} \
    --input_file ./data/images/face.jpg
    """

    res = run_cmd(cmd)

    feature_str = res.split("feature:")[-1]
    # feature = re.findall(pattern=r"-?\d+.\d+", string=feature_str)
    feature = feature_str.split(',')

    assert len(feature) == 512, f"facenet: feature len {len(feature)} is not 512"


#################  performance test
@pytest.mark.fast
def test_face_recognition_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/face_recognition_prof \
    -m /opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod \
    --vdsp_params ./data/configs/facenet_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 64 \
    --instance 2 \
    --iterations 500 \
    --shape [3,160,160] \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 2700
    assert (
        float(qps) > qps_thresh
    ), f"facenet:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/face_recognition_prof \
    -m /opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod \
    --vdsp_params ./data/configs/facenet_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 1 \
    --instance 1 \
    --iterations 10240 \
    --shape [3,160,160] \
    --percentiles [50,90,95,99] \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 810
    assert (
        float(qps) > qps_thresh
    ), f"facenet:best latancy qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_face_recognition_cpp_precision(device_id):
    os.system("mkdir -p facenet_output")
    cmd = f"""
    ./build/vaststreamx-samples/bin/face_recognition \
    -m /opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod \
    --vdsp_params ./data/configs/facenet_bgr888.json \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/lfw_mtcnnpy_160_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder facenet_output
    """
    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/face_recognition/facenet_eval.py \
    --gt_dir /opt/vastai/vaststreamx/data/datasets/lfw_mtcnnpy_160 \
    --gt_pairs_path /opt/vastai/vaststreamx/data/datasets/lfw_mtcnnpy_160_pairs.txt \
    --input_npz_path /opt/vastai/vaststreamx/data/datasets/lfw_mtcnnpy_160_filelist.txt \
    --out_npz_dir ./facenet_output
    """
    res = run_cmd(cmd, check_stderr=False)

    accuracy = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"Accuracy: \d+.\d+", string=res).group()
    ).group()

    accuracy_thres = 0.99
    assert (
        float(accuracy) > accuracy_thres
    ), f"facenet: accuracy {accuracy} is smaller than {accuracy_thres}"

    os.system("rm -rf facenet_output")


######################  python test #####################

################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_face_recognition_py(device_id):
    cmd = f"""
    python3 ./samples/face_recognition/face_recognition.py \
    -m /opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod \
    --vdsp_params ./data/configs/facenet_bgr888.json \
    --device_id {device_id} \
    --input_file ./data/images/face.jpg
    """
    res = run_cmd(cmd)

    feature_str = feature_str = res.split("[[")[-1].split("]]")[0]

    feature = re.findall(pattern=r"-?\d+.\d+", string=feature_str)

    assert len(feature) == 512, f"facenet: feature len {len(feature)} is not 512"


#################  performance test

@pytest.mark.fast
def test_face_recognition_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/face_recognition/face_recognition_prof.py \
    -m /opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod \
    --vdsp_params ./data/configs/facenet_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 64 \
    --instance 2 \
    --iterations 500 \
    --shape [3,160,160] \
    --percentiles [50,90,95,99] \
    --input_host true \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 2700
    assert (
        float(qps) > qps_thresh
    ), f"facenet:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    python3 ./samples/face_recognition/face_recognition_prof.py \
    -m /opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod \
    --vdsp_params ./data/configs/facenet_bgr888.json \
    --device_ids [{device_id}] \
    --batch_size 64 \
    --instance 2 \
    --iterations 500 \
    --shape [3,160,160] \
    --percentiles [50,90,95,99] \
    --input_host true \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 810
    assert (
        float(qps) > qps_thresh
    ), f"facenet:best latancy qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.fast
def test_face_recognition_py_precision(device_id):
    os.system("mkdir -p facenet_output")
    cmd = f"""
    python3 ./samples/face_recognition/face_recognition.py \
    -m /opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod \
    --vdsp_params ./data/configs/facenet_bgr888.json \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/lfw_mtcnnpy_160_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/ \
    --dataset_output_folder facenet_output
    """
    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/face_recognition/facenet_eval.py \
    --gt_dir /opt/vastai/vaststreamx/data/datasets/lfw_mtcnnpy_160 \
    --gt_pairs_path /opt/vastai/vaststreamx/data/datasets/lfw_mtcnnpy_160_pairs.txt \
    --input_npz_path /opt/vastai/vaststreamx/data/datasets/lfw_mtcnnpy_160_filelist.txt \
    --out_npz_dir ./facenet_output
    """
    res = run_cmd(cmd, check_stderr=False)

    accuracy = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"Accuracy: \d+.\d+", string=res).group()
    ).group()

    accuracy_thres = 0.99
    assert (
        float(accuracy) > accuracy_thres
    ), f"facenet: accuracy {accuracy} is smaller than {accuracy_thres}"

    os.system("rm -rf facenet_output")
