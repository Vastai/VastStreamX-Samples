#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../../")
sys.path.append(common_path)

from common.custom_op_base import CustomOpBase, vsx
import ctypes
from enum import Enum
import numpy as np

class BorderType(Enum):
    BORDER_CONSTANT = 0

class InterpolationType(Enum):
    INTER_LINEAR = 1


class Flags(Enum):
    FORWARD_MAP = 0
    INVERSE_MAP = 16

class ImageType(Enum):
    PLANAR = 0


class image_shape_t(ctypes.Structure):
    _fields_ = [
        ("channel", ctypes.c_int),
        ("height", ctypes.c_int),
        ("width", ctypes.c_int),
        ("c_pitch", ctypes.c_int),
        ("h_pitch", ctypes.c_int),
        ("w_pitch", ctypes.c_int),
    ]


class roi_t(ctypes.Structure):
    _fields_ = [
        ("c_start", ctypes.c_int),
        ("h_start", ctypes.c_int),
        ("w_start", ctypes.c_int),
        ("c_len", ctypes.c_int),
        ("h_len", ctypes.c_int),
        ("w_len", ctypes.c_int),
    ]


class warp_perspective_param_t(ctypes.Structure):
    _fields_ = [
        ("in_image_shape", image_shape_t),
        ("out_image_shape", image_shape_t),
        ("image_type", ctypes.c_int),
        ("M", ctypes.c_double*9),
        ("inter_type", ctypes.c_int),
        ("border_type", ctypes.c_int),
        ("border_value", ctypes.c_int),
        ("flags", ctypes.c_int),
        ("roi", roi_t),
    ]


class WarpPerspectiveOp(CustomOpBase):
    def __init__(self, op_name="warp_perspective_u8_op", elf_file="/opt/vastai/vaststream/lib/op/vdsp_op", device_id=0) -> None:
        super().__init__(op_name, elf_file, device_id)
        self.custom_op_.set_callback_info(
            [(1, int(376 * 1.5), 500)], [(1, int(376 * 1.5), 500)]
        )
    
    def Process(self, input: vsx.Image, mat: np.ndarray, crop_width, crop_height):
        in_width, in_height = input.width, input.height
        out_width, out_height = crop_width, crop_height

        op_parmas = warp_perspective_param_t()

        op_parmas.in_image_shape.channel = 3 # RGB_PLANAR
        op_parmas.in_image_shape.height = in_height
        op_parmas.in_image_shape.width = in_width
        op_parmas.in_image_shape.c_pitch = 3
        op_parmas.in_image_shape.h_pitch = in_height
        op_parmas.in_image_shape.w_pitch = in_width

        op_parmas.out_image_shape.channel = 3 # RGB_PLANAR
        op_parmas.out_image_shape.height = out_height
        op_parmas.out_image_shape.width = out_width
        op_parmas.out_image_shape.c_pitch = 3
        op_parmas.out_image_shape.h_pitch = out_height
        op_parmas.out_image_shape.w_pitch = out_width

        op_parmas.image_type = ImageType.PLANAR.value
        for i, m in enumerate(mat):
            op_parmas.M[i] = m
        op_parmas.inter_type = InterpolationType.INTER_LINEAR.value
        op_parmas.border_type = BorderType.BORDER_CONSTANT.value
        op_parmas.border_value = 0
        op_parmas.flags = Flags.FORWARD_MAP.value
        op_parmas.roi.c_start = 0
        op_parmas.roi.h_start = 0
        op_parmas.roi.w_start = 0
        op_parmas.roi.c_len = 3
        op_parmas.roi.h_len = out_height
        op_parmas.roi.w_len = out_width

        op_conf_size = ctypes.sizeof(warp_perspective_param_t)

        outputs = self.custom_op_.run_sync(
            images=[input],
            config=ctypes.string_at(ctypes.byref(op_parmas), op_conf_size),
            output_info=[([out_width, out_height], vsx.ImageFormat.RGB_PLANAR)],
        )
        return outputs[0]