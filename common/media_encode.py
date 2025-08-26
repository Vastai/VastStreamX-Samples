#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from .media_base import MediaBase, vsx
import numpy as np
import threading
from typing import Union, List


class MediaEncode(MediaBase):
    def __init__(self, codec_type, device_id) -> None:
        super().__init__(codec_type, device_id)

    def process(self, input, loop=False):
        return self.process_impl(input, loop)

    def get_result(self):
        return self.get_result_impl()

    def stop(self):
        return self.process_impl(input, True)

    def get_test_data(self, loop=True):
        return self.get_test_data_impl(loop)

    def process_impl(self, input, loop):
        pass

    def get_result_impl(self):
        pass

    def get_test_data_impl(self, loop):
        pass
