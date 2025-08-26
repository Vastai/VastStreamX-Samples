import vaststreamx as vsx
from .normalize_op import NormalizeOp, NormalType
from .space_to_depth_op import SpaceToDepthOp
import numpy as np
from typing import Union, List
import common.utils as utils


class Dinov2Model:
    def __init__(
        self,
        model_prefix,
        norm_op_elf,
        space2depth_op_elf,
        batch_size=1,
        device_id=0,
        hw_config="",
        output_type=vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE,
    ) -> None:
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0

        # normalize op
        mean = np.array([14275, 14156, 13951], dtype=np.uint16)
        std = np.array([13140, 13099, 13107], dtype=np.uint16)
        norm_type = NormalType.NORMAL_DIV255_MINUSMEAN_DIVSTD
        self.normalize_op_ = NormalizeOp(
            elf_file=norm_op_elf,
            device_id=device_id,
            mean=mean,
            std=std,
            norm_type=norm_type,
        )

        # space_to_depth op
        kh, kw, out_h, out_w = 14, 14, 256, 784
        self.space_to_depth_op_ = SpaceToDepthOp(
            kh=kh,
            kw=kw,
            oh_align=out_h,
            ow_align=out_w,
            elf_file=space2depth_op_elf,
            device_id=device_id,
        )

        # model
        self.model_ = vsx.Model(model_prefix, batch_size, hw_config)
        self.model_op_ = vsx.ModelOperator(self.model_)
        self.graph_ = vsx.Graph(output_type)
        self.graph_.add_operators(self.model_op_)
        self.stream_ = vsx.Stream(self.graph_, vsx.StreamBalanceMode.RUN)
        self.stream_.register_operator_output(self.model_op_)
        self.stream_.build()

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
        return self.process([input])[0]

    def process_impl(self, inputs):
        mod_h, mod_w = self.model_.input_shape[0][-2:]
        outputs = []
        infer_input = []
        for input in inputs:
            vacc_inputs = []
            cvtcolor_out = vsx.cvtcolor(input, vsx.ImageFormat.RGB_PLANAR)
            resize_out = vsx.resize(
                cvtcolor_out,
                vsx.ImageResizeType.BILINEAR_PILLOW,
                resize_width=mod_w,
                resize_height=mod_h,
            )

            norm_out = self.normalize_op_.process(resize_out)
            space_to_depth_out = self.space_to_depth_op_.process(norm_out)
            vacc_inputs.append(space_to_depth_out)
            infer_input.append(vacc_inputs)
        model_outs = self.stream_.run_sync(infer_input)
        for model_out in model_outs:
            outputs.append(vsx.as_numpy(model_out[0]).astype(np.float32))

        return outputs
