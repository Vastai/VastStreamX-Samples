#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import vaststreamx as vsx
import numpy as np
from typing import Union
import common.utils as utils
from common.tensorize_op import TensorizeOp

attr = vsx.AttrKey


class VastaiDynamicGaHa:
    def __init__(
        self,
        module_info,
        vdsp_config,
        max_input_size,
        batch_size=1,
        device_id=0,
        patch=256,
    ):
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(module_info)
        self.model_.set_input_shape(max_input_size)
        self.model_.set_batch_size(batch_size)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.preproc_ops_ = vsx.Operator.load_ops_from_json_file(vdsp_config)
        assert len(self.preproc_ops_) > 0, "load_ops_from_json_file failed"
        self.graph_ = vsx.Graph(vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE)
        self.graph_.add_operators(*self.preproc_ops_, self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()

        self.patch_ = patch

    def get_fusion_op_iimage_format(self):
        for op in self.preproc_ops_:
            if op.op_type >= 100:
                buildin_op = op.cast_to_buildin_operator()
                if "IIMAGE_FORMAT" in list(buildin_op.attributes.keys()):
                    imagetype = buildin_op.get_attribute(attr.IIMAGE_FORMAT)
                    return utils.imagetype_to_vsxformat(imagetype)
                else:
                    return vsx.ImageFormat.YUV_NV12
        assert False, "Can't find fusion op that op_type >= 100"

    def process(self, input: Union[np.ndarray, vsx.Image]):
        if isinstance(input, np.ndarray):
            input = utils.cv_rgb_image_to_vastai(input, self.device_id_)
        w, h = input.width, input.height
        p = self.patch_
        w_patch = (w + p - 1) // p * p
        h_patch = (h + p - 1) // p * p
        if w_patch > 1024 or h_patch > 1024:
            print("Warning: image size is too large, dynamic shape max to 1024x1024")
            return None
        top, left = 0, 0
        right = w_patch - w - left
        bottom = h_patch - h - top

        # print(f"w:{w}, h:{h}, new_w:{w_patch}, new_h:{h_patch},bottom:{bottom}, right:{right}")
        params = {
            "rgb_letterbox_ext": [],
            "dynamic_model_input_shapes": [[[1, 3, h_patch, w_patch]]],
        }
        params["rgb_letterbox_ext"].append((w, h, top, bottom, left, right))
        outputs = self.stream_.run_sync([input], params)[0]
        return [vsx.as_numpy(out).astype(np.float32) for out in outputs]


class VastaiDynamicHsChunk:
    def __init__(self, module_info, max_input_size, batch_size=1, device_id=1):
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(module_info)
        self.model_.set_input_shape(max_input_size)
        self.model_.set_batch_size(batch_size)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.graph_ = vsx.Graph(vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_HOST)
        self.graph_.add_operators(self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()

    def process(self, input: np.ndarray):
        aligned_input = utils.get_activation_aligned(input.astype(np.float16))
        vsx_tensor = vsx.from_numpy(aligned_input, self.device_id_)
        (n, c, h, w) = input.shape
        params = {"dynamic_model_input_shapes": [[[n, c, h, w]]]}
        outputs = self.stream_.run_sync([[vsx_tensor]], params)[0]
        return [vsx.as_numpy(out).astype(np.float32) for out in outputs]


class VastaiDynamicGs:
    def __init__(
        self,
        gs0_module_info,
        gs0_max_input_size,
        gs_module_info,
        gs_max_input_size,
        tensorize_elf_path,
        batch_size=1,
        device_id=0,
        gs0_hw_config="",
        gs_hw_config="",
    ):

        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0

        self.gs0_model_ = vsx.Model(gs0_module_info)
        self.gs0_model_.set_input_shape(gs0_max_input_size)
        self.gs0_model_.set_batch_size(batch_size)
        self.gs0_model_op_ = vsx.ModelOperator(self.gs0_model_)
        self.gs0_graph_ = vsx.Graph(vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_884_DEVICE)
        self.gs0_graph_.add_operators(self.gs0_model_op_)
        self.gs0_stream_ = vsx.Stream(self.gs0_graph_, vsx.StreamBalanceMode.RUN)
        self.gs0_stream_.register_operator_output(self.gs0_model_op_)
        self.gs0_stream_.build()
        self.tensor_op_ = TensorizeOp(
            elf_file=tensorize_elf_path,
            device_id=device_id,
        )

        self.gs_model_ = vsx.Model(gs_module_info)
        self.gs_model_.set_input_shape(gs_max_input_size)
        self.gs_model_.set_batch_size(batch_size)
        self.gs_model_op_ = vsx.ModelOperator(self.gs_model_)
        self.gs_graph_ = vsx.Graph(vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE)
        self.gs_graph_.add_operators(self.gs_model_op_)
        self.gs_stream_ = vsx.Stream(self.gs_graph_, vsx.StreamBalanceMode.RUN)
        self.gs_stream_.register_operator_output(self.gs_model_op_)
        self.gs_stream_.build()

    def process(self, input: np.ndarray):
        # aligned_input = utils.get_activation_aligned(input.astype(np.float16))
        # vsx_tensor = vsx.from_numpy(aligned_input, self.device_id_)
        vsx_tensor = self.tensor_op_.process(input.astype(np.float16))
        (n, c, h, w) = input.shape
        gs0_params = {"dynamic_model_input_shapes": [[[n, c, h, w]]]}
        gs0_outputs = self.gs0_stream_.run_sync([[vsx_tensor]], gs0_params)[0]
        self.gs0_output_shape_ = self.gs0_model_.output_shape
        gs_params = {"dynamic_model_input_shapes": [[gs0_outputs[0].shape]]}
        outputs = self.gs_stream_.run_sync([gs0_outputs], gs_params)[0]

        return [vsx.as_numpy(out).astype(np.float32) for out in outputs]
