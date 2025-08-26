
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <iostream>
#include <vector>

#include "media_base.hpp"
#include "utils.hpp"
#include "vaststreamx/datatypes/data_manager.h"
#include "vaststreamx/datatypes/image.h"

#define MAX_PATH_LEN (1024)

namespace vsx {

class MediaEncode : public MediaBase {
 public:
  MediaEncode(vsx::CodecType codec_type, uint32_t device_id)
      : MediaBase(codec_type, device_id) {}

  uint32_t Process(const vsx::Image& image, bool end_flag) {
    return ProcessImpl(image, end_flag);
  }

  void Stop() {
    vsx::Image image;
    ProcessImpl(image, true);
  }

  bool GetResult(std::shared_ptr<vsx::DataManager>& data) {
    return GetResultImpl(data);
  }

  vsx::Image GetTestData(bool loop = true) { return GetTestDataImpl(loop); }

 protected:
  virtual uint32_t ProcessImpl(const vsx::Image& data, bool end_flag) = 0;

  virtual bool GetResultImpl(std::shared_ptr<vsx::DataManager>& data) = 0;

  virtual vsx::Image GetTestDataImpl(bool loop = true) = 0;
};
}  // namespace vsx