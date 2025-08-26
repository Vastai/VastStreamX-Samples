#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import vaststreamx as vsx


class MediaBase:
    def __init__(self, codec_type, device_id=0) -> None:
        self.codec_type_ = codec_type
        self.device_id_ = device_id
        assert vsx.set_device(device_id) == 0
