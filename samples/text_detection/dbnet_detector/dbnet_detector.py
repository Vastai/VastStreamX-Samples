#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import cv2
from shapely.geometry import Polygon
import pyclipper
import ctypes

current_file_path = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(current_file_path, "../../..")
sys.path.append(common_path)

from common.model_cv import ModelCV, vsx


class op_param_t(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("pitch", ctypes.c_int32),
        ("thresh", ctypes.c_float),
    ]


class DbnetDetector(ModelCV):
    def __init__(
        self,
        model_prefix,
        vdsp_config,
        batch_size=1,
        device_id=0,
        hw_config="",
        thresh=0.3,
        box_thresh=0.6,
        max_candidates=1024,
        unclip_ratio=1.5,
        use_dilation=False,
        score_mode="fast",
        elf_file="/opt/vastai/vaststreamx/data/elf/find_contours_ext_op",
    ):
        super().__init__(model_prefix, vdsp_config, batch_size, device_id, hw_config)
        self.thresh_ = thresh
        self.box_thresh_ = box_thresh
        self.max_candidates_ = max_candidates
        self.unclip_ratio_ = unclip_ratio
        self.use_dilation_ = use_dilation
        self.elf_file_ = elf_file
        self.min_size_ = 3
        assert score_mode in [
            "slow",
            "fast",
        ], f"Score mode must be in [slow, fast] but got: {score_mode}"
        self.score_mode_ = score_mode
        self.dilation_kernel_ = np.array([[1, 1], [1, 1]]) if use_dilation else None
        self.vdsp_op = vsx.CustomOperator(
            op_name="find_contours", elf_file_path=elf_file
        )
        self.vdsp_op.set_callback_info(
            [(1, int(376 * 1.5), 500)],
            [
                (1, int(376 * 1.5), 500),
                (1, int(376 * 1.5), 500),
                (1, int(376 * 1.5), 500),
                (1, int(376 * 1.5), 500),
            ],
        )

    def process_impl(self, input):
        outputs = self.stream_.run_sync(input)
        return [
            self.post_process(output, input[i].width, input[i].height)
            for i, output in enumerate(outputs)
        ]

    def post_process(self, fp16_tensors, image_width, image_height):
        height, width = fp16_tensors[0].shape[-2:]
        contours = self.run_op(fp16_tensors[0], width, height)
        num_contours = min(len(contours), self.max_candidates_)

        boxes = []
        scores = []
        pred = np.array(vsx.as_numpy(fp16_tensors[0])).astype(np.float32).squeeze()

        model_height, model_width = (
            self.model_.input_shape[0][-2],
            self.model_.input_shape[0][-1],
        )
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size_:
                continue
            points = np.array(points)
            if self.score_mode_ == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh_ > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size_ + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / model_width * image_width), 0, image_width
            )
            box[:, 1] = np.clip(
                np.round(box[:, 1] / model_height * image_height), 0, image_height
            )
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return [np.array(boxes, dtype=np.int16), scores]

    def run_op(self, tensor, width, height):
        op_param = op_param_t()
        op_param.width = width
        op_param.height = height
        op_param.pitch = width
        op_param.thresh = self.thresh_
        op_conf_size = ctypes.sizeof(op_param_t)

        output_list = self.vdsp_op.run_sync(
            tensors=[tensor],
            config=ctypes.string_at(ctypes.byref(op_param), op_conf_size),
            output_info=[
                ([height * width * 2], vsx.TypeFlag.INT16),
                ([self.max_candidates_, 5], vsx.TypeFlag.UINT32),
                ([1], vsx.TypeFlag.UINT32),
                ([4, height, width + 2], vsx.TypeFlag.UINT8),
            ],
        )
        outs = [vsx.as_numpy(out) for out in output_list]
        contours = []
        contours_num = outs[2][0]
        p = 0
        for i in range(contours_num):
            points = outs[1][i][4]
            contour = (
                outs[0][p : p + points * 2].reshape((points, 1, 2)).astype(np.int32)
            )
            p += points * 2
            contours.append(contour)
        return contours

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio_
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        return np.array(offset.Execute(distance))

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        """
        box_score_fast: use bbox mean score as the mean score
        """
        h, w = bitmap.shape[-2:]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        """
        box_score_slow: use polyon mean score as the mean score
        """
        h, w = bitmap.shape[-2:]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]
