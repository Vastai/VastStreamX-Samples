#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from .model_cv import ModelCV, vsx


class Segmentator(ModelCV):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
        output_type=vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE,
    ) -> None:
        super().__init__(
            model_prefix, vdsp_config, batch_size, device_id, hw_config, output_type
        )
    
    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        return [out[0] for out in outputs] # not copy to host