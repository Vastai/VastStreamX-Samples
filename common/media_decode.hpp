
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <vector>

#include "media_base.hpp"
#include "utils.hpp"
#include "vaststreamx/datatypes/data_manager.h"

namespace vsx {

class MediaDecode : public MediaBase {
 public:
  MediaDecode(vsx::CodecType codec_type, uint32_t device_id)
      : MediaBase(codec_type, device_id) {}

  uint32_t Process(const std::shared_ptr<vsx::DataManager> &data,
                   bool end_flag) {
    return ProcessImpl(data, end_flag);
  }

  void Stop() { ProcessImpl(nullptr, true); }

  bool GetResult(vsx::Image &image) { return GetResultImpl(image); }

  std::shared_ptr<vsx::DataManager> GetTestData(bool loop) {
    return GetTestDataImpl(loop);
  }

 protected:
  virtual uint32_t ProcessImpl(const std::shared_ptr<vsx::DataManager> &data,
                               bool end_flag) = 0;

  virtual bool GetResultImpl(vsx::Image &image) = 0;

  virtual std::shared_ptr<vsx::DataManager> GetTestDataImpl(bool loop) = 0;
};
}  // namespace vsx