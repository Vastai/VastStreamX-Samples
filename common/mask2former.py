#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from .model_cv import ModelCV, vsx
import numpy as np

import torch
from torch.nn import functional as F


class Mask2Former(ModelCV):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        threshold=0.5,
        hw_config="",
    ) -> None:
        super().__init__(
            model_prefix,
            vdsp_config,
            batch_size,
            device_id,
            hw_config,
            vsx.GraphOutputType.GRAPH_OUTPUT_TYPE_NCHW_DEVICE,
        )
        self.threshold_ = threshold

    def process_impl(self, input):
        model_outs = self.stream_.run_sync(input)
        outputs = []
        for inp, mod_out in zip(input, model_outs):
            out = self.post_process(mod_out, inp.width, inp.height)
            outputs.append(out)
        return outputs

    def post_process(self, fp16_tensors, image_width, image_height):
        cls_result, seg_result = vsx.as_numpy(fp16_tensors[0]), vsx.as_numpy(
            fp16_tensors[1]
        )

        np_arrs = [vsx.as_numpy(tensor).astype(np.float32) for tensor in fp16_tensors]
        cls_result, seg_result = np_arrs[0], np_arrs[1]

        seg_result = seg_result.transpose(1, 0).reshape([1, -1, 256, 256])
        image_size = (image_height, image_width)

        cls_result = torch.from_numpy(cls_result)
        seg_result = torch.from_numpy(seg_result)

        seg_result = F.interpolate(
            seg_result,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )[0]

        num_classes = 80
        test_topk_per_image = 100
        num_queries = 100

        scores = F.softmax(cls_result, dim=-1)[:, :-1]

        labels = (
            torch.arange(num_classes, device="cpu")
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )

        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            test_topk_per_image, sorted=False
        )

        classes = labels[topk_indices]

        topk_indices = topk_indices // num_classes

        mask_pred = seg_result[topk_indices]
        pred_masks = (mask_pred > 0).float()

        mask_scores_per_image = (
            mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)
        ).sum(1) / (pred_masks.flatten(1).sum(1) + 1e-6)

        scores = scores_per_image * mask_scores_per_image
        scores, indices = torch.sort(scores, descending=True)

        scores = scores.numpy()
        classes = classes[indices].numpy()
        pred_masks = pred_masks[indices].numpy()

        scores = scores[np.where(scores > self.threshold_)]
        classes = classes[: len(scores)]
        pred_masks = pred_masks[: len(scores)]

        boxes = []
        for mask in pred_masks:
            rows, cols = np.where(mask > 0)
            if len(rows) > 2 and len(cols) > 2:
                boxes.append([cols.min(), rows.min(), cols.max(), rows.max()])
            else:
                boxes.append([-1, -1, -1, -1])

        return [
            np.array(classes),
            np.array(scores),
            np.array(boxes),
            np.array(pred_masks),
            np.array([len(scores)]),
        ]
