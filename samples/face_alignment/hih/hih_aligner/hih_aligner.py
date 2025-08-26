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
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)

from common.model_cv import ModelCV, vsx
import torch
import numpy as np


def concat_output(pred):
    pred0 = torch.Tensor(vsx.as_numpy(pred[0]))
    pred1 = torch.Tensor(vsx.as_numpy(pred[1]))
    return pred0, pred1


def decode_woo_head(target_maps, offset_map=None):
    """
    Args:
        target_maps (n,98,64,64) tensor float32
        offset_map is None here.

    return :
        preds (n,98,2)
    """
    max_v, idx = torch.max(
        target_maps.view(
            target_maps.size(0),
            target_maps.size(1),
            target_maps.size(2) * target_maps.size(3),
        ),
        2,
    )
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    max_v = max_v.view(idx.size(0), idx.size(1), 1)
    pred_mask = max_v.gt(0).repeat(1, 1, 2).float()

    preds[..., 0].remainder_(target_maps.size(3))  # coordinate x
    preds[..., 1].div_(target_maps.size(2)).floor_()  # coordinate y

    preds.mul_(pred_mask)
    return preds


def decode_hih_head(results, heatmap_size=64):
    pred_heatmap, pred_offset = concat_output(results)
    preds = decode_woo_head(pred_heatmap).float()
    offsets = decode_woo_head(pred_offset) / torch.tensor(
        [pred_offset.size(3), pred_offset.size(2)], dtype=torch.float32
    )
    preds.add_(offsets)
    preds.div_(heatmap_size)
    return np.asarray(preds)


class Hih(ModelCV):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
    ) -> None:
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        return [self.post_process(output) for i, output in enumerate(outputs)]

    def post_process(self, fp32_tensors):
        landmarks = decode_hih_head(fp32_tensors).squeeze()
        return landmarks
