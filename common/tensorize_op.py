#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import ctypes
from enum import Enum
from .custom_op_base import CustomOpBase, vsx
from typing import Union, List
import numpy as np


class TENSORIZE_FMT(Enum):
    TENSORIZE_FMT_CHW = 0
    TENSORIZE_FMT_HWC = 1


class tensorize_op_t(ctypes.Structure):
    _fields_ = [
        ("src_dim", ctypes.c_uint32 * 3),
        ("src_pitch", ctypes.c_uint32 * 3),
        ("ele_size", ctypes.c_uint32),
        ("src_fmt", ctypes.c_uint32),
    ]


class TensorizeOp(CustomOpBase):
    def __init__(
        self,
        op_name="tensorize_op",
        elf_file="/opt/vastai/vastpipe/data/elf/tensorize_ext_op",
        device_id=0,
    ):
        super().__init__(op_name=op_name, elf_file=elf_file, device_id=device_id)

    def process(
        self, input: Union[List[np.ndarray], List[vsx.Tensor], np.ndarray, vsx.Tensor]
    ):
        if isinstance(input, list):
            if isinstance(input[0], np.ndarray):
                return self.process([vsx.from_numpy(x, self.device_id_) for x in input])
            else:
                return self.process_impl(input)
        return self.process([input])[0]

    def process_impl(self, inputs):
        assert len(inputs) == 1
        outputs = []
        for input in inputs:
            c, h, w = input.shape[-3:]

            op_conf = tensorize_op_t()
            op_conf.src_dim = (c, h, w)
            op_conf.src_pitch = (c, h, w)
            op_conf.ele_size = ctypes.sizeof(ctypes.c_short)
            op_conf.src_fmt = TENSORIZE_FMT.TENSORIZE_FMT_CHW.value

            op_conf_size = ctypes.sizeof(tensorize_op_t)

            outs = self.custom_op_.run_sync(
                tensors=[input],
                config=ctypes.string_at(ctypes.byref(op_conf), op_conf_size),
                output_info=[([c, h, w], vsx.TypeFlag.FLOAT16)],
            )
            print(f"custom_op_ shape:{outs[0].shape}")
            outputs.append(outs[0])
        return outputs
