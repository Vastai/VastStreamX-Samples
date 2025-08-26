#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from .yolov8_seg_post_proc_op import Yolov8SegPostProcOp
from .model_cv import ModelCV, vsx
import numpy as np


class Yolov8Segmenter(ModelCV):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        elf_file,
        batch_size=1,
        device_id=0,
        hw_config="",
    ) -> None:
        super().__init__(
            model_prefix,
            vdsp_config,
            batch_size,
            device_id,
            hw_config,
            output_type=vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_HOST,
        )
        self.post_proc_op_ = Yolov8SegPostProcOp("yolov8_seg_op", elf_file, device_id)

    def process_impl(self, input):
        model_outs = self.stream_.run_sync(input)
        outputs = []
        for inp, mod_out in zip(input, model_outs):
            op_outs = self.post_process(mod_out, inp.width, inp.height)
            num = vsx.as_numpy(op_outs[4])[0]
            if num > 0:
                outs = [vsx.as_numpy(out) for out in op_outs]
                outputs.append(outs)
            else:
                outputs.append([])
        return outputs

    def post_process(self, fp16_tensors, image_width, image_height):
        return self.post_proc_op_.process(
            fp16_tensors, self.model_.input_shape[0], [image_height, image_width]
        )
