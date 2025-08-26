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

from common.media_decode import MediaDecode


class NaluType(Enum):
    NALU_TYPE_NONE = 0
    NALU_TYPE_H264_SPS = 7
    NALU_TYPE_H264_PPS = 8
    NALU_TYPE_H264_AUD = 9
    NALU_TYPE_H264_IDR = 5
    NALU_TYPE_H264_P = 1
    NAL_TYPE_HEVC_TRAIL_R = NALU_TYPE_H264_P

    NALU_TYPE_H264_SEI = 6

    NALU_TYPE_HEVC_IDR_W_RADL = 0x13
    NALU_TYPE_HEVC_IDR_N_LP = 0x14
    NALU_TYPE_HEVC_VPS = 0x20
    NALU_TYPE_HEVC_SPS = 0x21
    NALU_TYPE_HEVC_PPS = 0x22
    NALU_TYPE_HEVC_AUD = 0x23


class Vdecoder(MediaDecode):
    def __init__(self, codec_type, device_id, stream_list) -> None:
        super().__init__(codec_type, device_id)
        self.stream_list_ = stream_list
        self.codec_type_ = codec_type
        self.video_decoder_ = vsx.VideoDecoder(codec_type)

        # 初始化一个空列表来存储NALU数据
        self.nalus_ = []
        self.index_ = 0
        self.nalu_num_ = 0
        # 打开H.264文件以二进制模式
        if (
            codec_type == vsx.CodecType.CODEC_TYPE_H264
            or codec_type == vsx.CodecType.CODEC_TYPE_HEVC
        ):
            self.init_nalus()
        elif codec_type == vsx.CodecType.CODEC_TYPE_AV1:
            src = vsx.MediaSource()
            src.open(self.stream_list_)
            while True:
                data, b_eof = src.read()
                if b_eof:
                    break
                else:
                    self.nalus_.append(data)
                    self.nalu_num_ += 1

        else:
            print(f"undefined codec type:{codec_type}")

    # 定义一个函数来检测NALU的开始和结束

    def init_nalus(self):
        file_content = bytes()
        with open(self.stream_list_, "rb") as file:
            # 读取文件内容
            file_content = file.read()

        # 使用循环来遍历文件内容并提取NALU数据
        nalu_len = 0
        nalu_buf = []
        NALU_PREFIX = b"\x00\x00\x00\x01"
        NALU_PREFIX_LEN = len(NALU_PREFIX)
        NALU_PREFIX2 = b"\x00\x00\x01"
        NALU_PREFIX2_LEN = len(NALU_PREFIX2)
        get_frame = False

        def is_nalu_start(byte):
            if (
                file_content[byte] == 0x00
                and file_content[byte + 1] == 0x00
                and file_content[byte + 2] == 0x01
            ):
                return True, 3
            elif (
                file_content[byte] == 0x00
                and file_content[byte + 1] == 0x00
                and file_content[byte + 2] == 0x00
                and file_content[byte + 3] == 0x01
            ):
                return True, 4
            else:
                return False, 0

        i = 0
        content_len = len(file_content)
        while i + 4 < content_len:
            flag, _ = is_nalu_start(i)
            if not flag:
                i += 1
            else:
                if i != 0:
                    print(f"start not with 0x00")
                    file_content = file_content[i:]
                break
        if i + 4 >= content_len:
            print(f"i:{i}, content_len:{content_len}")
            return

        start = 0
        index = 4
        content_len = len(file_content)
        while index + 4 < content_len:
            flag2, prefix_len = is_nalu_start(index)
            if not flag2:
                index += 1
            else:
                nalu_buf += file_content[start:index]
                nalu_len += index - start
                nalu_type = 0
                nalu_header_len = 0
                if NALU_PREFIX[:] == file_content[start : start + NALU_PREFIX_LEN]:
                    nalu_header_len = NALU_PREFIX_LEN
                elif NALU_PREFIX2[:] == file_content[start : start + NALU_PREFIX2_LEN]:
                    nalu_header_len = NALU_PREFIX2_LEN
                else:
                    print(f"nalue_header_len:{nalu_header_len}")

                if self.codec_type_ == vsx.CodecType.CODEC_TYPE_H264:
                    nalu_ = file_content[start + nalu_header_len]
                    nalu_type = nalu_ & 0x1F
                    if nalu_type == 5 or nalu_type == 1:
                        get_frame = True
                elif self.codec_type_ == vsx.CodecType.CODEC_TYPE_HEVC:
                    nalu_type = (file_content[start + nalu_header_len] & 0x7E) >> 1
                    if (
                        nalu_type == NaluType.NALU_TYPE_HEVC_IDR_W_RADL.value
                        or nalu_type == NaluType.NALU_TYPE_HEVC_IDR_N_LP.value
                        or nalu_type == 1
                        or nalu_type == 0
                    ):
                        get_frame = True
                else:
                    print(f"undefined codec type:{self.codec_type_}")
                if get_frame and nalu_len > 0:
                    self.nalus_.append(bytes(nalu_buf[:nalu_len]))
                    self.nalu_num_ += 1
                    nalu_len = 0
                    nalu_buf = []
                    get_frame = False

                start = index
                index += prefix_len
        if start < content_len:
            self.nalus_.append(bytes(file_content[start:]))
            self.nalu_num_ += 1

    def process_impl(self, input, end_flag):
        if end_flag:
            self.video_decoder_.stop_send_data()
        else:
            self.video_decoder_.send_data(input)

    def get_test_data_impl(self, loop=True):
        if loop:
            nalu = self.nalus_[self.index_ % self.nalu_num_]
            self.index_ += 1
            return nalu
        if self.index_ < self.nalu_num_:
            nalu = self.nalus_[self.index_]
            self.index_ += 1
            return nalu
        else:
            return None

    def get_result_impl(self):
        return self.video_decoder_.recv_image()
