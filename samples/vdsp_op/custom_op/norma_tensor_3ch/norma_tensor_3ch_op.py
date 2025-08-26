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
from enum import Enum

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../../../")
sys.path.append(common_path)

from common.custom_op_base import CustomOpBase, vsx
import numpy as np
from typing import Union, List
import ctypes


class ColorSpace(Enum):
    COLOR_SPACE_BT709 = 0
    COLOR_SPACE_BT601 = 1
    COLOR_SPACE_BUTT = 2


class DataType(Enum):
    INT8 = 0
    FP16 = 1
    FP32 = 2
    BF16 = 3


class value_t(ctypes.Union):
    _fields_ = [
        ("fp16", ctypes.c_uint16),
        ("bf16", ctypes.c_uint16),
        ("fp32", ctypes.c_float),
        ("int8", ctypes.c_int8),
    ]


class ImageType(Enum):
    XI_TILE_YUV_NV12_TYPE = 0
    XI_TILE_YUV_I420_TYPE = 1
    XI_TILE_RGB888_TYPE = 2
    XI_TILE_RGB_PLANAR_TYPE = 3
    XI_TILE_BAYER_BG_TYPE = 4
    XI_TILE_BAYER_GB_TYPE = 5
    XI_TILE_BAYER_RG_TYPE = 6
    XI_TILE_BAYER_GR_TYPE = 7
    XI_TILE_GRAY_TYPE = 8


class rgb_planar_t(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_uint64 * 3),
        ("pitch", ctypes.c_int32 * 3),
    ]


class yuv_nv12_t(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_uint64 * 2),
        ("pitch_y", ctypes.c_int32),
        ("pitch_uv", ctypes.c_int32),
    ]


class yuv_i420_t(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_uint64 * 3),
        ("pitch_y", ctypes.c_int32),
        ("pitch_u", ctypes.c_int32),
        ("pitch_v", ctypes.c_int32),
    ]


class rgb_888_t(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_uint64),
        ("pitch", ctypes.c_int32),
    ]


class gray_t(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_uint64),
        ("pitch", ctypes.c_int32),
    ]


class ptr_t(ctypes.Union):
    _fields_ = [
        ("rgb_planar", rgb_planar_t),
        ("yuv_nv12", yuv_nv12_t),
        ("yuv_i420", yuv_i420_t),
        ("rgb_888", rgb_888_t),
        ("gray", gray_t),
    ]


class common_img_obj_t(ctypes.Structure):
    _fields_ = [
        ("img_type", ctypes.c_int),
        ("cspace", ctypes.c_uint16),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("ptr", ptr_t),
    ]


class NormalType(Enum):
    NORMAL_EQUAL = 0
    NORMAL_MINUSMEAN_DIVSTD = 1
    NORMAL_DIV255_MINUSMEAN_DIVSTD = 2
    NORMAL_DIV127_5_MINUSONE = 3
    NORMAL_DIV255 = 4


class nt_3ch_para_t(ctypes.Structure):
    _fields_ = [
        ("in_image", common_img_obj_t),
        ("dst", ctypes.c_uint64),
        ("out_width", ctypes.c_uint32),
        ("out_height", ctypes.c_uint32),
        ("ch_pitch", ctypes.c_uint32),
        ("in_dtype", ctypes.c_int32),
        ("out_dtype", ctypes.c_int32),
        ("norma_type", ctypes.c_int32),
        ("mean", ctypes.c_uint16 * 3),
        ("std", ctypes.c_uint16 * 3),
        ("scale", ctypes.c_uint16 * 3),
        ("interleave_width", ctypes.c_int32),
        ("interleave_height", ctypes.c_int32),
        ("skip_norma_quant", ctypes.c_int32),
        ("is_need_swap_blue", ctypes.c_int32),
        ("padding", value_t),
    ]


class NormaTensor3ChOp(CustomOpBase):
    def __init__(self, op_name, elf_file, device_id=0) -> None:
        super().__init__(op_name=op_name, elf_file=elf_file, device_id=device_id)
        self.profiling = False
        self.output_tensor = None
        self.custom_op_.set_callback_info(
            [(1, int(376 * 1.5), 500)], [(1, int(376 * 1.5), 500)]
        )

    def process(
        self, input: Union[List[np.ndarray], List[vsx.Tensor], np.ndarray, vsx.Tensor]
    ):
        if isinstance(input, list):
            if isinstance(input[0], np.ndarray):
                return self.process([vsx.from_numpy(x, self.device_id_) for x in input])
            else:
                return self.process_impl(input)
        return self.process([input])[0]

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        dummy = np.random.rand(*input_shape).astype(dtype)
        self.profiling = True
        self.output_tensor = vsx.from_numpy(
            np.random.rand(*input_shape).astype(np.float16), self.device_id_
        )
        if ctx == "CPU":
            return [dummy] * batch_size
        else:
            vacc_dummy = vsx.from_numpy(dummy, self.device_id_)
            return [vacc_dummy] * batch_size

    def process_impl(self, inputs):
        outputs = []
        for input in inputs:
            channel, height, width = input.shape[-3:]
            plane_offset = width * height
            op_params = nt_3ch_para_t()
            addr_r = input.addr
            addr_g = addr_r + plane_offset
            addr_b = addr_g + plane_offset

            op_params.in_image.img_type = ImageType.XI_TILE_RGB_PLANAR_TYPE.value
            op_params.in_image.cspace = ColorSpace.COLOR_SPACE_BT601.value
            op_params.in_image.width = width
            op_params.in_image.height = height
            op_params.in_image.ptr.rgb_planar.addr[0] = addr_r
            op_params.in_image.ptr.rgb_planar.addr[1] = addr_g
            op_params.in_image.ptr.rgb_planar.addr[2] = addr_b
            op_params.in_image.ptr.rgb_planar.pitch[0] = width
            op_params.in_image.ptr.rgb_planar.pitch[1] = width
            op_params.in_image.ptr.rgb_planar.pitch[2] = width

            op_params.out_width = width
            op_params.out_height = height
            op_params.ch_pitch = 4
            op_params.in_dtype = DataType.INT8.value
            op_params.out_dtype = DataType.FP16.value
            op_params.norma_type = NormalType.NORMAL_DIV255_MINUSMEAN_DIVSTD.value
            op_params.mean[0] = 22520
            op_params.mean[1] = 22520
            op_params.mean[2] = 22520
            op_params.std[0] = 15360
            op_params.std[1] = 15360
            op_params.std[2] = 15360
            op_params.interleave_width = 0
            op_params.interleave_height = 0
            op_params.skip_norma_quant = 0
            op_params.is_need_swap_blue = 0
            op_params.padding.fp16 = 0

            if self.profiling:
                op_params.dst = self.output_tensor.addr
            else:
                self.output_tensor = vsx.from_numpy(
                    np.random.rand(channel, height, width).astype(np.float16),
                    self.device_id_,
                )
                op_params.dst = self.output_tensor.addr

            op_conf_size = ctypes.sizeof(nt_3ch_para_t)

            self.custom_op_.run_sync(
                tensors=[input],
                config=ctypes.string_at(ctypes.byref(op_params), op_conf_size),
                output_info=[([channel, height, width], vsx.TypeFlag.FLOAT16)],
            )
            outputs.append(self.output_tensor)
        return outputs
