#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from .yolo_world_text import YoloWorldText
from .yolo_world_image import YoloWorldImage
import numpy as np
from .yolo_world_post_process import get_postprocess, get_scores_batch
import common.utils as utils


class YoloWorld:
    def __init__(
        self,
        imgmod_prefix,
        imgmod_vdsp_config,
        txtmod_prefix,
        txtmod_vdsp_config,
        tokenizer_path,
        batch_size=1,
        device_id=0,
        score_thres=0.001,
        nms_pre=30000,
        iou_thres=0.7,
        max_per_image=300,
        imgmod_hw_config="",
        txtmod_hw_config="",
    ) -> None:
        self.imgmod_ = YoloWorldImage(
            imgmod_prefix,
            imgmod_vdsp_config,
            batch_size,
            device_id,
            imgmod_hw_config,
        )
        self.txtmod_ = YoloWorldText(
            txtmod_prefix,
            txtmod_vdsp_config,
            tokenizer_path,
            batch_size,
            device_id,
            txtmod_hw_config,
        )
        self.device_id_ = device_id
        self.score_thres_ = score_thres
        self.nms_pre_ = nms_pre
        self.iou_thres_ = iou_thres
        self.max_per_image_ = max_per_image

    def get_fusion_op_iimage_format(self):
        return self.imgmod_.get_fusion_op_iimage_format()

    def process(self, image, texts):
        text_features = self.process_texts(texts)
        return self.process_image(image, text_features)

    def process_texts(self, texts):
        txt_features = self.txtmod_.process(texts)
        text_feature = np.array(txt_features).squeeze()
        return (text_feature, utils.bert_get_activation_fp16_A(text_feature))

    def process_image(self, image, text_features):
        text_feature, text_884_feature = text_features
        img_features = self.imgmod_.process((image, text_884_feature))
        return self.post_process(image, img_features, text_feature)

    def post_process(self, image, img_features, txt_features):
        score_160, score_80, score_40 = get_scores_batch(img_features, txt_features)

        if isinstance(image, np.ndarray):
            ori_shape = (image.shape[0], image.shape[1])
        else:
            ori_shape = (image.height, image.width)

        imgmod_shape = self.imgmod_.input_shape[0]
        new_shape = (imgmod_shape[-2], imgmod_shape[-1])
        r = min(new_shape[0] / ori_shape[0], new_shape[1] / ori_shape[1])

        ratio = r, r  # width, height ratios
        new_unpad = int(round(ori_shape[1] * r)), int(round(ori_shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        pad = (dw, dh)

        res = get_postprocess(
            (score_160, score_80, score_40),
            (img_features[-3:]),
            ori_shape=ori_shape,
            scale_factor=ratio,
            pad_param=(pad[1], pad[1], pad[0], pad[0]),
            score_thr=self.score_thres_,
            nms_pre=self.nms_pre_,
            iou_threshold=self.iou_thres_,
            cfg_max_per_img=self.max_per_image_,
        )

        scores = res["scores"].numpy()
        bboxes = res["bboxes"].numpy()
        labels = res["labels"].numpy()

        result = {"scores": scores, "bboxes": bboxes, "labels": labels}
        return result

    def compute_tokens(self, text):
        assert isinstance(text, str), f"input text must be str"
        return self.txtmod_.make_tokens(text=text)
