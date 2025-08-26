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
import torch
from typing import Union, List
from common.tensorize_op import TensorizeOp

attr = vsx.AttrKey


class VastaiGaHa:
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
        patch=256,
    ):
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
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
        new_w = (w + p - 1) // p * p
        new_h = (h + p - 1) // p * p
        top, left = 0, 0
        right = new_w - w - left
        bottom = new_h - h - top
        params = {"rgb_letterbox_ext": []}
        # print(f"w:{w}, h:{h}, new_w:{new_w}, new_h:{new_h},bottom:{bottom}, right:{right}")

        params["rgb_letterbox_ext"].append((w, h, top, bottom, left, right))
        outputs = self.stream_.run_sync([input], params)[0]
        return [vsx.as_numpy(out).astype(np.float32) for out in outputs]


class VastaiHsChunk:
    def __init__(self, model_prefix, batch_size=1, device_id=1, hw_config=""):
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.graph_ = vsx.Graph(vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_HOST)
        self.graph_.add_operators(self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()

    def process(self, input: np.ndarray):
        aligned_input = utils.get_activation_aligned(input.astype(np.float16))
        vsx_tensor = vsx.from_numpy(aligned_input, self.device_id_)
        outputs = self.stream_.run_sync([[vsx_tensor]])[0]
        return [vsx.as_numpy(out).astype(np.float32) for out in outputs]


class VastaiGs:
    def __init__(
        self, model_prefix, tensorize_elf_path, batch_size=1, device_id=0, hw_config=""
    ):
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0

        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.graph_ = vsx.Graph(vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE)
        self.graph_.add_operators(self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()
        self.tensor_op_ = TensorizeOp(
            elf_file=tensorize_elf_path,
            device_id=device_id,
        )

    def process(self, input):
        # aligned_input = utils.get_activation_aligned(input.astype(np.float16))
        # vsx_tensor = vsx.from_numpy(aligned_input, self.device_id_)
        vsx_tensor = self.tensor_op_.process(input.astype(np.float16))
        outputs = self.stream_.run_sync([[vsx_tensor]])[0]
        return [vsx.as_numpy(out).astype(np.float32) for out in outputs]


class VastaiElicNoEntropy:
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
        patch=256,
    ):
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.preproc_ops_ = vsx.Operator.load_ops_from_json_file(vdsp_config)
        assert len(self.preproc_ops_) > 0, "load_ops_from_json_file failed"
        self.graph_ = vsx.Graph(
            vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_HOST, [0, 0, 1]
        )
        self.graph_.add_operators(*self.preproc_ops_, self.model_op_)
        # self.graph_.add_operators(self.model_op_)
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

    @property
    def batch_size(self):
        return self.model_.batch_size

    @property
    def max_batch_size(self):
        return self.model_.max_batch_size

    @property
    def input_count(self):
        return self.model_.input_count

    @property
    def output_count(self):
        return self.model_.output_count

    @property
    def input_shape(self):
        return self.model_.input_shape

    @property
    def output_shape(self):
        return self.model_.output_shape

    def get_test_data(self, dtype, input_shape, batch_size, ctx="VACC"):
        assert len(input_shape) >= 2
        height = input_shape[-2]
        width = input_shape[-1]
        dummy = np.zeros((height, width, 3), dtype=dtype)
        if ctx == "CPU":
            return [dummy] * batch_size
        else:
            vacc_dummy = vsx.create_image(
                dummy, vsx.ImageFormat.BGR_INTERLEAVE, width, height, self.device_id_
            )
            return [vacc_dummy] * batch_size

    # def process(self, input: Union[np.ndarray, vsx.Image]):
    def process(
        self, input: Union[List[np.ndarray], List[vsx.Image], np.ndarray, vsx.Image]
    ):
        if isinstance(input, list):
            if isinstance(input[0], np.ndarray):
                return self.process(
                    [utils.cv_rgb_image_to_vastai(x, self.device_id_) for x in input]
                )
            else:
                return self.process_impl(input)
        return self.process([input])

    def process_impl(self, inputs):
        params = {"rgb_letterbox_ext": []}
        for input in inputs:
            w, h = input.width, input.height
            p = self.patch_
            new_w = (w + p - 1) // p * p
            new_h = (h + p - 1) // p * p
            left = (new_w - w) // 2
            right = new_w - w - left
            top = (new_h - h) // 2
            bottom = new_h - h - top
            params["rgb_letterbox_ext"].append((w, h, top, bottom, left, right))
        outputs = self.stream_.run_sync(inputs, params)
        infer_output = []
        for output in outputs:
            infer_output.append(
                [vsx.as_numpy(out).astype(np.float32) for out in output]
            )
        return infer_output

    def inference(self, input: Union[np.ndarray, vsx.Image]):
        outputs = self.process(input)
        # print(f"len:{len(output)}")
        x_hat = torch.from_numpy(outputs[0][0])

        y_likelihoods = torch.from_numpy(outputs[0][1].astype(np.float32))
        # w, h = input.width, input.height
        p = self.patch_
        new_w = (input.width + p - 1) // p * p
        new_h = (input.height + p - 1) // p * p
        h = new_h // 64
        w = new_w // 64
        hw = h * w

        z_likelihoods_output = (
            outputs[0][2]
            .reshape(4, 192, 1, hw // 4 // 16, 16, 16)
            .transpose(1, 2, 4, 0, 3, 5)
            .reshape(192, 16, hw)
        )

        # exit(0)
        z_likelihoods_output = z_likelihoods_output[:, :1, :].reshape(1, 192, h, w)
        z_likelihoods = torch.from_numpy(z_likelihoods_output.astype(np.float32))

        replacement = torch.tensor(1e-9)
        y_likelihoods = torch.where(y_likelihoods == 0, replacement, y_likelihoods)
        z_likelihoods = torch.where(z_likelihoods == 0, replacement, z_likelihoods)
        return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}
