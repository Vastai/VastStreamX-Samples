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
from .media_decode import MediaDecode


class Jdecoder(MediaDecode):
    def __init__(self, device_id, stream_list) -> None:
        super().__init__(vsx.CodecType.CODEC_TYPE_JPEG, device_id)
        self.stream_list_ = stream_list
        self.codec_type_ = vsx.CodecType.CODEC_TYPE_JPEG
        self.jpeg_decoder_ = vsx.JpegDecoder()

        # 打开H.264文件以二进制模式
        self.jpeg_content_ = bytes()
        with open(self.stream_list_, "rb") as file:
            # 读取文件内容
            self.jpeg_content_ = file.read()

    def process_impl(self, input, end_flag):
        if end_flag:
            self.jpeg_decoder_.stop_send_data()
        else:
            self.jpeg_decoder_.send_data(input)

    def get_result(self):
        return self.jpeg_decoder_.recv_image()

    def get_test_data_impl(self, loop=True):
        return self.jpeg_content_
