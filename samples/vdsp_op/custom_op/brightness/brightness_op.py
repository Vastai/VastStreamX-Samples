#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
from typing import Union, List

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../../../")
sys.path.append(common_path)

from common.custom_op_base import CustomOpBase, vsx
import numpy as np
from typing import Union, List
import ctypes


class yuv_nv12_shape_t(ctypes.Structure):
    _fields_ = [
        ("height", ctypes.c_int),
        ("width", ctypes.c_int),
        ("h_pitch", ctypes.c_int),
        ("w_pitch", ctypes.c_int),
    ]


class brightness_param_t(ctypes.Structure):
    _fields_ = [
        ("iimage_shape", yuv_nv12_shape_t),
        ("oimage_shape", yuv_nv12_shape_t),
        ("scale", ctypes.c_float),
    ]


def cv_nv12_to_vastai(image_cv, device_id):
    assert len(image_cv.shape) == 2
    h = image_cv.shape[0] * 2 // 3
    w = image_cv.shape[1]
    if len(image_cv.shape) == 2:
        return vsx.create_image(image_cv, vsx.ImageFormat.YUV_NV12, w, h, device_id)
    else:
        raise Exception("unsupported ndarray shape", image_cv.shape)


class BrightnessOp(CustomOpBase):
    def __init__(self, op_name, elf_file, device_id=0, scale=2.2) -> None:
        super().__init__(op_name=op_name, elf_file=elf_file, device_id=device_id)
        self.scale_ = scale
        self.custom_op_.set_callback_info(
            [(1, int(376 * 1.5), 500)], [(1, int(376 * 1.5), 500)]
        )

    def process(
        self, input: Union[List[np.ndarray], List[vsx.Image], np.ndarray, vsx.Image]
    ):
        if isinstance(input, list):
            if isinstance(input[0], np.ndarray):
                return self.process(
                    [cv_nv12_to_vastai(x, self.device_id_) for x in input]
                )
            else:
                return self.process_impl(input)
        return self.process([input])[0]

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        assert len(input_shape) >= 2
        height = input_shape[-2]
        width = input_shape[-1]
        dummy = np.zeros((height * 3 // 2, width), dtype=dtype)
        if ctx == "CPU":
            return [dummy] * batch_size
        else:
            vacc_dummy = vsx.create_image(
                dummy, vsx.ImageFormat.YUV_NV12, width, height, self.device_id_
            )
            return [vacc_dummy] * batch_size

    def process_impl(self, inputs):
        outputs = []
        for input in inputs:
            width, height = input.width, input.height
            op_param = brightness_param_t()
            in_shape = yuv_nv12_shape_t()

            in_shape.height = height
            in_shape.width = width
            in_shape.h_pitch = height
            in_shape.w_pitch = width
            out_shape = yuv_nv12_shape_t()
            out_shape.height = height
            out_shape.width = width
            out_shape.h_pitch = height
            out_shape.w_pitch = width

            op_param.iimage_shape = in_shape
            op_param.oimage_shape = out_shape
            op_param.scale = self.scale_

            op_conf_size = ctypes.sizeof(brightness_param_t)

            outs = self.custom_op_.run_sync(
                images=[input],
                config=ctypes.string_at(ctypes.byref(op_param), op_conf_size),
                output_info=[([width, height], vsx.ImageFormat.YUV_NV12)],
            )
            outputs.append(outs[0])
        return outputs
