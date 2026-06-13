
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "common/resnet.hpp"
#include "common/utils.hpp"
#include "vaststreamx/vaststreamx.h"

namespace vsx {

class DocImgOrientClassifier : public Resnet {
 public:
  DocImgOrientClassifier(const std::string& model_prefix,
                         const std::string& vdsp_config,
                         uint32_t batch_size = 1, uint32_t device_id = 0,
                         const std::string& hw_config = "")
      : Resnet(model_prefix, vdsp_config, batch_size, device_id, hw_config) {}

 protected:
  std::vector<vsx::Tensor> ProcessImpl(const std::vector<vsx::Image>& images) {
    vsx::StreamExtraRuntimeConfig extra_configs;
    extra_configs.crop_resize_config.reserve(images.size());
    for (const auto& image : images) {
      vsx::CropResizeExtConfig config;
      GetResize(image.Width(), image.Height(), resize_size_,
                config.resize_width, config.resize_height);
      GetCrop(config.resize_width, config.resize_height, model_size_,
              config.crop_x, config.crop_y);
      extra_configs.crop_resize_config.push_back(config);
    }
    auto outputs = stream_->RunSync(images, extra_configs);
    std::vector<vsx::Tensor> results;
    for (auto& output : outputs) {
      results.push_back(output[0]);
    }
    return results;
  }

 public:
  void PostProcess(const vsx::Tensor& model_output, int& index, float& score) {
    auto output = model_output.Clone();
    auto fp32_output = vsx::ConvertTensorFromFp16ToFp32(output);
    int max_index = 0;
    float max_score = -10000.0;
    // Find the index of the maximum value
    for (size_t i = 0; i < fp32_output.GetSize(); ++i) {
      if (fp32_output.Data<float>()[i] > max_score) {
        max_index = i;
        max_score = fp32_output.Data<float>()[i];
      }
    }
    index = max_index;
    score = max_score;
  }
};

}  // namespace vsx