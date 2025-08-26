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
from .media_encode import MediaEncode


class Jencoder(MediaEncode):
    def __init__(self, device_id, stream_list, width, height, format) -> None:
        super().__init__(vsx.CodecType.CODEC_TYPE_JPEG, device_id)
        self.stream_list_ = stream_list
        self.codec_type_ = vsx.CodecType.CODEC_TYPE_JPEG
        self.width = width
        self.height = height
        self.format = format
        self.image = vsx.create_image(
            rawdata_path=self.stream_list_,
            format=self.format,
            width=self.width,
            height=self.height,
            device_id=device_id,
        )
        self.jpeg_encoder = vsx.JpegEncoder()

    def process_impl(self, input, end_flag):
        if end_flag:
            return self.jpeg_encoder.stop_send_image()
        else:
            return self.jpeg_encoder.send_image(input)

    def get_result(self):
        return self.jpeg_encoder.recv_data()

    def get_test_data_impl(self, loop=True):
        return self.image
