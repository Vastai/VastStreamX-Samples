
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "glog/logging.h"
#include "vaststreamx/core/resource.h"
#include "vaststreamx/core/stream.h"
#include "vaststreamx/datatypes/frame_packet.h"
#include "vaststreamx/datatypes/image.h"

namespace vsx {

class MediaBase {
 public:
  explicit MediaBase(CodecType codec_type, uint32_t device_id = 0)
      : codec_type_(codec_type), device_id_(device_id) {
    CHECK(vsx::SetDevice(device_id) == 0)
        << "SetDevice " << device_id << " failed";
  }
  virtual bool IsKeyFrame() { return true; }

 protected:
  CodecType codec_type_;
  uint32_t device_id_;
};

}  // namespace vsx