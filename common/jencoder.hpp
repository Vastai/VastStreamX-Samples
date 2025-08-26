
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

#include "common/media_encode.hpp"
#include "common/utils.hpp"
#include "vaststreamx/core/resource.h"
#include "vaststreamx/media/jpeg_encoder.h"

namespace vsx {

inline std::shared_ptr<vsx::DataManager> CreateCpuDataManager(
    const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open file: " << filename;
    return nullptr;
  }
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::shared_ptr<vsx::DataManager> data_manager =
      std::make_shared<vsx::DataManager>(size, vsx::Context::CPU(0));
  if (nullptr == data_manager) {
    LOG(ERROR) << "Failed to create data manager";
    return nullptr;
  }
  file.read(reinterpret_cast<char*>(data_manager->GetDataAddress()), size);
  file.close();
  return data_manager;
}

inline uint32_t CreateCpuImage(const std::string& image_path,
                               vsx::Image& output_image, int image_width,
                               int image_height,
                               vsx::ImageFormat image_format) {
  auto data_manager = CreateCpuDataManager(image_path);
  if (nullptr == data_manager) {
    LOG(ERROR) << "Failed to create data manager from file: " << image_path;
    return 1;
  }
  output_image =
      vsx::Image(image_format, image_width, image_height, 0, 0, data_manager);
  return 0;
}

class Jencoder : public MediaEncode {
 public:
  Jencoder(uint32_t device_id, std::string stream_list, uint32_t frame_width,
           uint32_t frame_height, vsx::ImageFormat format)
      : MediaEncode(vsx::CODEC_TYPE_JPEG, device_id),
        input_file_path_(stream_list),
        width_(frame_width),
        height_(frame_height),
        format_(format) {
    vsx::Image image;
    CreateCpuImage(input_file_path_, image, width_, height_, format_);
    image_ = vsx::Image(image.Format(), image.Width(), image.Height(),
                        vsx::Context::VACC(device_id));
    image_.CopyFrom(image);
    jpeg_encoder_ = std::make_unique<vsx::JpegEncoder>();
  }

 private:
  std::unique_ptr<vsx::JpegEncoder> jpeg_encoder_;

  vsx::Image image_;

  std::string input_file_path_;
  uint32_t width_;
  uint32_t height_;
  vsx::ImageFormat format_;

 protected:
  uint32_t ProcessImpl(const vsx::Image& image, bool end_flag) {
    int value = 0;
    if (end_flag) {
      value = jpeg_encoder_->StopSendImage();
      return value;
    } else {
      if (image.Format() == vsx::ImageFormat::YUV_NV12 && image.GetDataPtr() &&
          image.GetDataBytes()) {
        value = jpeg_encoder_->SendImage(image);
      } else {
        LOG(ERROR) << "image format error";
        return -1;
      }
    }
    return value;
  }

  bool GetResultImpl(std::shared_ptr<vsx::DataManager>& data) {
    return jpeg_encoder_->RecvData(data);
  }

  vsx::Image GetTestDataImpl(bool loop) { return image_; }
};

}  // namespace vsx