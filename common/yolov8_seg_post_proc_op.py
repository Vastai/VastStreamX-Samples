#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from .custom_op_base import CustomOpBase, vsx
import ctypes


class image_shape_layout_t(ctypes.Structure):
    _fields_ = [
        ("height", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("h_pitch", ctypes.c_int32),
        ("w_pitch", ctypes.c_int32),
    ]


class yolov8_seg_op_t(ctypes.Structure):
    _fields_ = [
        ("model_in_shape", image_shape_layout_t),
        ("model_out_shape", image_shape_layout_t),
        ("origin_image_shape", image_shape_layout_t),
        ("k", ctypes.c_uint32),
        ("retina_masks", ctypes.c_uint32),
        ("max_detect_num", ctypes.c_uint32),
    ]


class Yolov8SegPostProcOp(CustomOpBase):
    def __init__(self, op_name, elf_file, device_id=0, retina_masks=True) -> None:
        super().__init__(op_name=op_name, elf_file=elf_file, device_id=device_id)
        self.custom_op_.set_callback_info(
            [(1, int(376 * 1.5), 500)] * 5, [(1, int(376 * 1.5), 500)] * 6
        )
        self.retina_masks_ = retina_masks

    def process(self, input_tensors, model_input_shape, image_shape):
        inputs = [input.clone(self.device_id_) for input in input_tensors]
        mask_shape = inputs[4].shape
        classes_shape = inputs[0].shape
        model_in_height, model_in_width = model_input_shape[-2:]
        model_out_height, model_out_width = mask_shape[-2:]
        image_height, image_width = image_shape[-2:]
        max_detect_num = classes_shape[1]
        mask_ch_num = 32  # Don't change this parameter

        op_conf = yolov8_seg_op_t()
        op_conf.model_in_shape.height = model_in_height
        op_conf.model_in_shape.width = model_in_width
        op_conf.model_in_shape.h_pitch = model_in_height
        op_conf.model_in_shape.w_pitch = model_in_width

        op_conf.model_out_shape.height = model_out_height
        op_conf.model_out_shape.width = model_out_width
        op_conf.model_out_shape.h_pitch = model_out_height
        op_conf.model_out_shape.w_pitch = model_out_width
        op_conf.origin_image_shape.height = image_height
        op_conf.origin_image_shape.width = image_width
        op_conf.origin_image_shape.h_pitch = image_height
        op_conf.origin_image_shape.w_pitch = image_width

        op_conf.k = mask_ch_num
        op_conf.retina_masks = 1 if self.retina_masks_ else 0
        op_conf.max_detect_num = max_detect_num

        op_conf_size = ctypes.sizeof(yolov8_seg_op_t)

        mask_out_h = model_in_height
        mask_out_w = model_in_width
        if self.retina_masks_:
            mask_out_h = image_height
            mask_out_w = image_width

        buffer_size = (max_detect_num + 3) * max(
            model_in_width * model_in_height, image_height * image_width
        )
        config_bytes = ctypes.string_at(ctypes.byref(op_conf), op_conf_size)
        outputs = self.custom_op_.run_sync(
            tensors=inputs,
            config=config_bytes,
            output_info=[
                ([max_detect_num], vsx.TypeFlag.FLOAT16),
                ([max_detect_num], vsx.TypeFlag.FLOAT16),
                ([max_detect_num, 4], vsx.TypeFlag.FLOAT16),
                ([max_detect_num, mask_out_h, mask_out_w], vsx.TypeFlag.UINT8),
                ([2], vsx.TypeFlag.UINT32),
                ([buffer_size], vsx.TypeFlag.UINT8),
            ],
        )
        res = outputs[:5]
        return res
