#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from .model_cv import ModelCV, vsx
import numpy as np


class Detector(ModelCV):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        threshold=0.2,
        hw_config="",
    ) -> None:
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)
        self.threshold_ = threshold

    def set_threshold(self, threshold):
        self.threshold_ = threshold

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        return [
            self.post_process(output, input[i].width, input[i].height)
            for i, output in enumerate(outputs)
        ]

    def post_process(self, fp16_tensors, image_width, image_height):
        data_count = fp16_tensors[0].size
        result_np = np.zeros((data_count, 6), dtype=np.float32) - 1.0
        # check tensor size validation
        assert (
            fp16_tensors[0].size == fp16_tensors[1].size
            and fp16_tensors[1].size * 4 == fp16_tensors[2].size
        ), f"Output tensor size error, sizes are:{fp16_tensors[0].size},{fp16_tensors[1].size},{fp16_tensors[2].size}"
        class_data = vsx.as_numpy(fp16_tensors[0]).squeeze()
        score_data = vsx.as_numpy(fp16_tensors[1]).squeeze()
        bbox_data = vsx.as_numpy(fp16_tensors[2]).squeeze()

        model_width = self.model_.input_shape[0][3]
        model_height = self.model_.input_shape[0][2]

        r = min(model_width / image_width, model_height / image_height)
        unpad_w = image_width * r
        unpad_h = image_height * r
        dw = (model_width - unpad_w) / 2
        dh = (model_height - unpad_h) / 2
        obj = 0
        for i in range(data_count):
            category = int(class_data[i])
            if category < 0:
                break
            score = score_data[i]
            if score > self.threshold_:
                bbox_xmin = (bbox_data[i][0] - dw) / r
                bbox_ymin = (bbox_data[i][1] - dh) / r
                bbox_xmax = (bbox_data[i][2] - dw) / r
                bbox_ymax = (bbox_data[i][3] - dh) / r
                bbox_width = bbox_xmax - bbox_xmin
                bbox_height = bbox_ymax - bbox_ymin
                result_np[obj][0] = category
                result_np[obj][1] = score
                result_np[obj][2] = bbox_xmin
                result_np[obj][3] = bbox_ymin
                result_np[obj][4] = bbox_width
                result_np[obj][5] = bbox_height
                obj += 1

        return result_np
