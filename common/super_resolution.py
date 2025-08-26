#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from .custom_op_base import CustomOpBase, vsx
from .model_cv import ModelCV
from typing import Union, List
import ctypes
from enum import Enum


class image_shape_t(ctypes.Structure):
    _fields_ = [
        ("iimage_width", ctypes.c_uint32),
        ("iimage_height", ctypes.c_uint32),
    ]


class ConversionType(Enum):
    FLOAT16_TO_UINT8 = 0


class type_conversion_t(ctypes.Structure):
    _fields_ = [
        ("iimage_shape", image_shape_t),
        ("type", ctypes.c_int32),
        ("threshold", ctypes.c_float),
        ("scale", ctypes.c_int32),
        ("coef", ctypes.c_float),
        ("base", ctypes.c_float),
    ]


class PostProcessOp(CustomOpBase):
    def __init__(
        self,
        coef=1.0,
        scale=1,
        base=0.0,
        device_id=0,
        elf_file="/opt/vastai/vaststreamx/data/elf/postprocessimage",
        op_name="custom_op_tensor2image",
        threshold=0.0,
    ):
        super().__init__(op_name=op_name, elf_file=elf_file, device_id=device_id)
        self.coef_ = coef
        self.scale_ = scale
        self.base_ = base
        self.threshold_ = threshold

    def process(self, input: Union[List[vsx.Tensor], vsx.Tensor]):
        if isinstance(input, vsx.Tensor):
            return self.process([input])[0]
        else:
            return self.process_impl(input)

    def process_impl(self, inputs):
        outputs = []
        for input in inputs:
            assert len(input.shape) >= 3
            c, h, w = input.shape[-3:]
            op_conf = type_conversion_t()
            op_conf.iimage_shape.iimage_width = w
            op_conf.iimage_shape.iimage_height = h * c
            op_conf.type = ConversionType.FLOAT16_TO_UINT8.value
            op_conf.threshold = self.threshold_
            op_conf.scale = self.scale_
            op_conf.coef = self.coef_
            op_conf.base = self.base_
            op_conf_size = ctypes.sizeof(type_conversion_t)
            outs = self.custom_op_.run_sync(
                tensors=[input],
                config=ctypes.string_at(ctypes.byref(op_conf), op_conf_size),
                output_info=[([c, h, w], vsx.TypeFlag.UINT8)],
            )
            outputs.append(outs[0])
        return outputs


class SuperResolution(ModelCV):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        postproc_elf="/opt/vastai/vaststreamx/data/elf/postprocessimage",
        device_id=0,
        coef=1.0,
        scale=1,
        base=0.0,
        batch_size=1,
        hw_config="",
    ) -> None:
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)
        self.postproc_op_ = PostProcessOp(
            coef=coef,
            scale=scale,
            base=base,
            device_id=device_id,
            elf_file=postproc_elf,
        )

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)

        post_outs = [self.post_process(output[0]) for output in outputs]
        return [vsx.as_numpy(out) for out in post_outs]

    def post_process(self, fp16_tensor):
        output = self.postproc_op_.process(fp16_tensor)
        return output
