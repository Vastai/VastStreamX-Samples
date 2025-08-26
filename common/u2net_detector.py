#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../../..")
sys.path.append(common_path)

from common.model_cv import ModelCV, vsx
import torch
import numpy as np


class U2netDetector(ModelCV):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
    ) -> None:
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)

    def _normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)
        # ma = np.max(d)
        # mi = np.min(d)
        dn = (d - mi) / (ma - mi)
        return dn

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        return [self.post_process(output) for i, output in enumerate(outputs)]

    def post_process(self, fp16_tensors):
        mask = self._normPRED(
            # vsx.as_numpy(fp16_tensors[0]).squeeze()
            torch.Tensor(vsx.as_numpy(fp16_tensors[0]).squeeze())
        )
        return mask
