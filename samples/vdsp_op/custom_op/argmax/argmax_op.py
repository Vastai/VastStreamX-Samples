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


class planar_argmax_tensor_t(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_uint64),
        ("channel", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("h_pitch", ctypes.c_int32),
        ("w_pitch", ctypes.c_int32),
    ]


class planar_argmax_cfg_t(ctypes.Structure):
    _fields_ = [
        ("input", planar_argmax_tensor_t),
        ("output", planar_argmax_tensor_t),
    ]


class ArgmaxOp(CustomOpBase):
    def __init__(self, op_name, elf_file, device_id=0) -> None:
        super().__init__(op_name=op_name, elf_file=elf_file, device_id=device_id)
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
        if ctx == "CPU":
            return [dummy] * batch_size
        else:
            vacc_dummy = vsx.from_numpy(dummy, self.device_id_)
            return [vacc_dummy] * batch_size

    def process_impl(self, inputs):
        outputs = []
        for input in inputs:
            channel, height, width = input.shape[-3:]
            op_param = planar_argmax_cfg_t()

            op_param.input.addr = input.addr
            op_param.input.channel = channel
            op_param.input.height = height
            op_param.input.width = width
            op_param.input.h_pitch = height
            op_param.input.w_pitch = width

            op_param.output.channel = 1
            op_param.output.height = height
            op_param.output.width = width
            op_param.output.h_pitch = height
            op_param.output.w_pitch = width

            op_conf_size = ctypes.sizeof(planar_argmax_cfg_t)

            outs = self.custom_op_.run_sync(
                tensors=[input],
                config=ctypes.string_at(ctypes.byref(op_param), op_conf_size),
                output_info=[([1, 1, height, width], vsx.TypeFlag.UINT16)],
            )
            outputs.append(outs[0])
        return outputs
