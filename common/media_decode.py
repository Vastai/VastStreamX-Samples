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


class MediaDecode(MediaBase):
    def __init__(self, codec_type, device_id) -> None:
        super().__init__(codec_type, device_id)
        self.codec_type_ = codec_type

    def process(self, input, end_flag=False):
        return self.process_impl(input, end_flag)

    def stop(self):
        return self.process_impl(input, True)

    def get_result(self):
        return self.get_result_impl()

    def get_test_data(self, loop=True):
        return self.get_test_data_impl(loop)

    # @abstractmethod
    def process_impl(self, input):
        pass

    # @abstractmethod
    def get_result_impl(self):
        pass

    # @abstractmethod
    def get_test_data_impl(loop):
        pass
