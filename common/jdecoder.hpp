
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <typeinfo>
#include <vector>

#include "common/media_decode.hpp"
#include "common/utils.hpp"
#include "vaststreamx/core/resource.h"
#include "vaststreamx/media/jpeg_decoder.h"

namespace vsx {

class Jdecoder : public MediaDecode {
 public:
  Jdecoder(uint32_t device_id, std::string stream_list)
      : MediaDecode(CODEC_TYPE_JPEG, device_id), codec_type_(CODEC_TYPE_JPEG) {
    vsx::SetDevice(device_id);
    jpeg_decoder_ = std::make_unique<vsx::JpegDecoder>();
    vsx::ReadBinaryFile(stream_list, data_manager_);
  }

 private:
  std::unique_ptr<vsx::JpegDecoder> jpeg_decoder_;
  vsx::CodecType codec_type_;
  std::string stream_list_;
  std::shared_ptr<vsx::DataManager> data_manager_;

 protected:
  uint32_t ProcessImpl(const std::shared_ptr<vsx::DataManager>& data,
                       bool end_flag) {
    int value = 0;
    if (data) {
      value = jpeg_decoder_->SendData(data);
    }
    if (end_flag) {
      value = jpeg_decoder_->StopSendData();
    }
    return value;
  }

  bool GetResultImpl(vsx::Image& image) {
    std::shared_ptr<FrameAttr> frame_attr;
    return jpeg_decoder_->RecvImage(image);
  }

  std::shared_ptr<vsx::DataManager> GetTestDataImpl(bool loop) {
    return data_manager_;
  }
};

}  // namespace vsx