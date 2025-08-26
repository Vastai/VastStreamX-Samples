
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
def test_pointpillar_cpp(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/point_pillar \
    -m "[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]" \
    --elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
    --max_voxel_num [16000] \
    --voxel_size [0.16,0.16,4] \
    --coors_range [0,-39.68,-3,69.12,39.68,1] \
    --max_points_num 12000000 \
    --shuffle_enabled 0 \
    --normalize_enabled 0 \
    --feat_size [864,496,480,480] \
    --device_id {device_id} \
    --input_file /opt/vastai/vaststreamx/data/datasets/kitti_val/fov_pointcloud_float16/000001.bin  
    """

    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    assert (
        len(scores) >= 12
    ), f"point_pillar:detected object count={len(scores)} is smaller 12"

    os.system("rm 000001.*")

#################  performance test
@pytest.mark.fast
def test_pointpillar_cpp_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    ./build/vaststreamx-samples/bin/point_pillar_prof \
    -m "[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]" \
    --elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
    --max_voxel_num [16000] \
    --voxel_size [0.16,0.16,4] \
    --coors_range [0,-39.68,-3,69.12,39.68,1] \
    --shuffle_enabled 0 \
    --normalize_enabled 0 \
    --max_points_num 12000000 \
    --feat_size [864,496,480,480] \
    --device_ids [{device_id}] \
    --shape [40000] \
    --batch_size 1 \
    --instance 1 \
    --iterations 1500 \
    --input_host 1 \
    --queue_size 1
    """

    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 230
    assert (
        float(qps) > qps_thresh
    ), f"point_pillar c++:best prof qps {qps} is smaller than {qps_thresh}"


    # 测试最小时延
    cmd = f"""
    ./build/vaststreamx-samples/bin/point_pillar_prof \
    -m "[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]" \
    --elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
    --max_voxel_num [16000] \
    --max_points_num 12000000 \
    --voxel_size [0.16,0.16,4] \
    --coors_range [0,-39.68,-3,69.12,39.68,1] \
    --shuffle_enabled 0 \
    --normalize_enabled 0 \
    --feat_size [864,496,480,480] \
    --device_ids [{device_id}] \
    --shape [40000] \
    --batch_size 1 \
    --instance 1 \
    --iterations 1000 \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()


    qps_thresh = 160
    assert (
        float(qps) > qps_thresh
    ), f"point_pillar c++:best prof qps {qps} is smaller than {qps_thresh}"


############################# dataset test
@pytest.mark.slow
def test_pointpillar_cpp_precision(device_id):
    os.system("mkdir -p pointpillar_out")
    cmd = f"""
    ./build/vaststreamx-samples/bin/point_pillar \
    -m "[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]" \
    --elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
    --max_voxel_num [16000] \
    --voxel_size [0.16,0.16,4] \
    --coors_range [0,-39.68,-3,69.12,39.68,1] \
    --max_points_num 12000000 \
    --shuffle_enabled 0 \
    --normalize_enabled 0 \
    --feat_size [864,496,480,480] \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/kitti_val/fov_pointcloud_float16_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/kitti_val/ \
    --dataset_output_folder pointpillar_out
    """

    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/point_pillar/evaluation.py \
    --out_dir pointpillar_out
    """

    res = run_cmd(cmd, check_stderr=False)

    car_ap = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"car AP: \d+.\d+", string=res).group()
    ).group()

    pedestrian_ap = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"pedestrian AP: \d+.\d+", string=res).group(),
    ).group()

    cyclist_ap = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"cyclist AP: \d+.\d+", string=res).group(),
    ).group()

    car_ap_thresh = 90.0
    pedestrian_ap_thresh = 60.0
    cyclist_ap_thresh = 80.0

    assert (
        float(car_ap) > car_ap_thresh
    ), f"point_pillar c++: car_ap {car_ap} is smaller than {car_ap_thresh}"
    assert (
        float(pedestrian_ap) > pedestrian_ap_thresh
    ), f"point_pillar c++: pedestrian_ap {pedestrian_ap} is smaller than {pedestrian_ap_thresh}"
    assert (
        float(cyclist_ap) > cyclist_ap_thresh
    ), f"point_pillar c++: cyclist_ap {cyclist_ap} is smaller than {cyclist_ap_thresh}"

    os.system("rm -rf pointpillar_out")


######################  python test #####################

################ one image  test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_pointpillar_py(device_id):
    cmd = f"""
    python3 ./samples/detection3d/point_pillar/point_pillar.py \
    -m "[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]" \
    --elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
    --max_voxel_num [16000] \
    --voxel_size [0.16,0.16,4] \
    --coors_range [0,-39.68,-3,69.12,39.68,1] \
    --shuffle_enabled 0 \
    --normalize_enabled 0 \
    --max_points_num 12000000 \
    --feat_size [864,496,480,480] \
    --device_id {device_id} \
    --input_file /opt/vastai/vaststreamx/data/datasets/kitti_val/fov_pointcloud_float16/000001.bin  
    """
    res = run_cmd(cmd)

    scores_str = re.findall(pattern=r"score: \d+.\d+", string=res)

    scores = []
    for score_str in scores_str:
        score = re.search(pattern=r"\d+.\d+", string=score_str).group()
        scores.append(score)

    assert (
        len(scores) >= 12
    ), f"point_pillar python :detected object count={len(scores)} is smaller than 12"


#################  performance test
@pytest.mark.fast
def test_pointpillar_py_performance(device_id):
    # 测试最大吞吐
    cmd = f"""
    python3 ./samples/detection3d/point_pillar/point_pillar_prof.py \
    -m "[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]" \
    --elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
    --max_voxel_num [16000] \
    --max_points_num 12000000 \
    --voxel_size [0.16,0.16,4] \
    --coors_range [0,-39.68,-3,69.12,39.68,1] \
    --shuffle_enabled 0 \
    --normalize_enabled 0 \
    --feat_size [864,496,480,480] \
    --device_ids [{device_id}] \
    --shape [40000] \
    --batch_size 1 \
    --instance 1 \
    --iterations 1500 \
    --input_host 1 \
    --queue_size 1
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 230
    assert (
        float(qps) > qps_thresh
    ), f"point_pillar python:best prof qps {qps} is smaller than {qps_thresh}"

    # 测试最小时延
    cmd = f"""
    python3 ./samples/detection3d/point_pillar/point_pillar_prof.py \
    -m "[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]" \
    --elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
    --max_voxel_num [16000] \
    --max_points_num 12000000 \
    --voxel_size [0.16,0.16,4] \
    --coors_range [0,-39.68,-3,69.12,39.68,1] \
    --shuffle_enabled 0 \
    --normalize_enabled 0 \
    --feat_size [864,496,480,480] \
    --device_ids [{device_id}] \
    --shape [40000] \
    --batch_size 1 \
    --instance 1 \
    --iterations 1000 \
    --input_host 1 \
    --queue_size 0
    """
    res = run_cmd(cmd)

    qps = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"\(qps\): \d+.\d+", string=res).group()
    ).group()

    qps_thresh = 160
    assert (
        float(qps) > qps_thresh
    ), f"point_pillar python:best prof qps {qps} is smaller than {qps_thresh}"



############################# dataset test
@pytest.mark.slow
def test_pointpillar_py_precision(device_id):
    os.system("mkdir -p pointpillar_out")
    cmd = f"""
    python3 ./samples/detection3d/point_pillar/point_pillar.py \
    -m "[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]" \
    --elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
    --max_voxel_num [16000] \
    --voxel_size [0.16,0.16,4] \
    --coors_range [0,-39.68,-3,69.12,39.68,1] \
    --shuffle_enabled 0 \
    --normalize_enabled 0 \
    --max_points_num 12000000 \
    --feat_size [864,496,480,480] \
    --device_id {device_id} \
    --dataset_filelist /opt/vastai/vaststreamx/data/datasets/kitti_val/fov_pointcloud_float16_filelist.txt \
    --dataset_root /opt/vastai/vaststreamx/data/datasets/kitti_val/ \
    --dataset_output_folder pointpillar_out
    """
    run_cmd(cmd)

    cmd = f"""
    python3 ./evaluation/point_pillar/evaluation.py \
    --out_dir pointpillar_out
    """

    res = run_cmd(cmd, check_stderr=False)

    car_ap = re.search(
        pattern=r"\d+.\d+", string=re.search(pattern=r"car AP: \d+.\d+", string=res).group()
    ).group()

    pedestrian_ap = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"pedestrian AP: \d+.\d+", string=res).group(),
    ).group()

    cyclist_ap = re.search(
        pattern=r"\d+.\d+",
        string=re.search(pattern=r"cyclist AP: \d+.\d+", string=res).group(),
    ).group()

    car_ap_thresh = 90.0
    pedestrian_ap_thresh = 60.0
    cyclist_ap_thresh = 80.0

    assert (
        float(car_ap) > car_ap_thresh
    ), f"point_pillar python: car_ap {car_ap} is smaller than {car_ap_thresh}"
    assert (
        float(pedestrian_ap) > pedestrian_ap_thresh
    ), f"point_pillar python: pedestrian_ap {pedestrian_ap} is smaller than {pedestrian_ap_thresh}"
    assert (
        float(cyclist_ap) > cyclist_ap_thresh
    ), f"point_pillar python: cyclist_ap {cyclist_ap} is smaller than {cyclist_ap_thresh}"

    os.system("rm -rf pointpillar_out")
