
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

################### image test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_datatype_cpp_image(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/image \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file image_out.jpg
    """

    res = run_cmd(cmd)

    width = re.search(
        pattern=r"\d+", string=re.search(pattern=r"width: \d+", string=res).group()
    ).group()
    height = re.search(
        pattern=r"\d+", string=re.search(pattern=r"height: \d+", string=res).group()
    ).group()

    assert width and int(width) == 768, f"datatype: output width {width} is not 768"
    assert height and int(height) == 576, f"datatype: output height {height} is not 576"

    assert os.path.exists("image_out.jpg"), "datatype: can't find output file:image_out.jpg"

    os.system("rm image_out.jpg")


################### tensor test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_datatype_cpp_tensor(device_id):
    cmd = f"""
    ./build/vaststreamx-samples/bin/tensor \
    --device_id {device_id} \
    --input_npz /opt/vastai/vaststreamx/data/datasets/SQuAD_1.1/val_npz_6inputs/test_0.npz \
    --output_npz tensor_out.npz
    """

    res = run_cmd(cmd, False)

    tensors = re.search(
        pattern=r"\d+", string=re.search(pattern=r"There are \d+ tensors", string=res).group()
    ).group()

    assert int(tensors) == 6, f"datatype: tensor count {int(tensors)} is not 6 in npz file"

    print("datatype: tensor c++ test succeed.")
    os.system("rm tensor_out.npz")

################# python test  #######################


################### image test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_datatype_py_image(device_id):
    cmd = f"""
    python3 ./samples/datatype/image.py \
    --device_id {device_id} \
    --input_file ./data/images/dog.jpg \
    --output_file image_out.jpg
    """

    res = run_cmd(cmd)

    width = re.search(
        pattern=r"\d+", string=re.search(pattern=r"width: \d+", string=res).group()
    ).group()
    height = re.search(
        pattern=r"\d+", string=re.search(pattern=r"height: \d+", string=res).group()
    ).group()

    assert width and int(width) == 768, f"datatype: output width {width} is not 768"
    assert height and int(height) == 576, f"datatype: output height {height} is not 576"

    assert os.path.exists("image_out.jpg"), "datatype: can't find output file:image_out.jpg"

    os.system("rm image_out.jpg")


################### tensor test
@pytest.mark.fast
@pytest.mark.ai_integration
def test_datatype_py_tensor(device_id):
    cmd = f"""
    python3 ./samples/datatype/tensor.py \
    --device_id {device_id} \
    --input_npz /opt/vastai/vaststreamx/data/datasets/SQuAD_1.1/val_npz_6inputs/test_0.npz \
    --output_npz tensor_out.npz
    """

    res = run_cmd(cmd)

    tensors = re.findall(pattern=r"key:", string=res)

    assert len(tensors) == 6, f"datatype: tensor count {len(tensors)} is not 6 in npz file"

    os.system("rm tensor_out.npz")
