#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from .clip_image import ClipImage
from .clip_text import ClipText

import numpy as np


def np_softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    res = e_x / e_x.sum(axis=axis, keepdims=True)
    return res


class ClipModel:
    def __init__(
        self,
        imgmod_prefix,
        norm_elf,
        space2depth_elf,
        txtmod_prefix,
        txtmod_vdsp_config,
        batch_size=1,
        device_id=0,
        imgmod_hw_config="",
        txtmod_hw_config="",
    ) -> None:
        self.imgmod_ = ClipImage(
            imgmod_prefix,
            norm_elf,
            space2depth_elf,
            batch_size,
            device_id,
            imgmod_hw_config,
        )
        self.txtmod_ = ClipText(
            txtmod_prefix, txtmod_vdsp_config, batch_size, device_id, txtmod_hw_config
        )
        self.device_id_ = device_id

    def process(self, image, texts):
        img_feature = self.process_image(image)
        txt_features = self.process_texts(texts)

        return self.post_process(img_feature, txt_features)

    def process_image(self, image):
        return self.imgmod_.process(image)

    def process_texts(self, texts):
        return self.txtmod_.process(texts)

    def post_process(self, img_feature, txt_features):
        img_feat = np.multiply(img_feature, 100.00000762939453)
        txt_feat = np.concatenate(txt_features, axis=0)
        txt_feat = np.transpose(txt_feat)
        feature = np.matmul(img_feat, txt_feat).astype("float32")
        scores = np_softmax(feature).squeeze()
        return scores

    def compute_tokens(self, text):
        assert isinstance(text, str), f"input text must be str"
        return self.txtmod_.make_tokens(text=text)
