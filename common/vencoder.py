#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from .media_base import MediaBase, vsx
import numpy as np
from typing import Union, List
from enum import Enum

from common.media_encode import MediaEncode


class Vencoder(MediaEncode):
    def __init__(
        self,
        codec_type,
        device_id,
        frames,
        frame_num,
        frame_width,
        frame_height,
        format,
        frame_rate=30,
    ) -> None:
        super().__init__(codec_type, device_id)
        self.frames_ = frames
        self.frame_num_ = frame_num
        self.width_ = frame_width
        self.height_ = frame_height
        self.format_ = format
        self.codec_ = codec_type
        self.video_encoder_ = vsx.VideoEncoder(
            codec_type,
            frame_width,
            frame_height,
            {"frame_rate_denominator": 1, "frame_rate_numerator": frame_rate},
        )

        self.index_ = 0
        self.pts_ = 0

    def process_impl(self, input, loop):
        if loop is True:
            self.video_encoder_.stop_send_image()
        else:
            img_arr = np.frombuffer(input, dtype=np.uint8)
            writable_array = np.require(
                img_arr, dtype=img_arr.dtype, requirements=["O", "w"]
            )

            writable_array = writable_array.reshape(
                (1, (int)(self.height_ * 3 / 2), self.width_)
            )

            vsx_image = vsx.create_image(
                array=writable_array,
                format=vsx.ImageFormat.YUV_NV12,
                width=self.width_,
                height=self.height_,
                device_id=self.device_id_,
            )
            frame_attr = vsx.FrameAttr()
            frame_attr.frame_dts = self.pts_
            frame_attr.frame_pts = self.pts_
            self.pts_ += 1
            return self.video_encoder_.send_image(vsx_image, frame_attr)

    def get_result_impl(self):
        return self.video_encoder_.recv_data()

    def get_test_data_impl(self, loop=True):
        if loop:
            nalu = self.frames_[self.index_ % self.frame_num_]
            self.index_ += 1
            return nalu

        if self.index_ < self.frame_num_:
            nalu = self.frames_[self.index_]
            self.index_ += 1
            return nalu
        else:
            return None
