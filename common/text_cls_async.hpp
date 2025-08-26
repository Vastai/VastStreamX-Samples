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

#include "common/model_cv_async.hpp"
#include "common/utils.hpp"

namespace vsx {

class TextClassifierAsync : public ModelCVAsync {
 public:
  TextClassifierAsync(const std::string& model_prefix,
                      const std::string& vdsp_config,
                      const std::vector<uint32_t>& labels,
                      uint32_t batch_size = 1, uint32_t device_id = 0,
                      const std::string& hw_config = "")
      : ModelCVAsync(model_prefix, vdsp_config, batch_size, device_id,
                     hw_config) {
    model_->GetInputShapeByIndex(0, input_shape_);
  }
  uint32_t ProcessAsyncImpl(const std::vector<vsx::Image>& images) {
    StreamExtraRuntimeConfig conf;
    for (size_t i = 0; i < images.size(); ++i) {
      RgbLetterBoxExtConfig box;
      box.padding_bottom = 0;
      box.padding_left = 0;
      box.padding_top = 0;
      // calc resize width and resize height and padding right
      int img_h = images[i].Height();
      int img_w = images[i].Width();
      float radio = static_cast<float>(img_w) / img_h;
      int resize_w = 0;
      int model_in_width = input_shape_[3];
      int model_in_height = input_shape_[2];
      if (model_in_height * radio > model_in_width) {
        resize_w = model_in_width;
      } else {
        resize_w = model_in_height * radio;
      }
      int right = model_in_width - resize_w;
      if (right < 0) {
        right = 0;
      }
      box.padding_right = right;
      box.resize_width = resize_w;
      box.resize_height = model_in_height;

      conf.rgb_letterbox_ext_config.push_back(box);
    }
    return stream_->RunAsync(images, conf);
  }
  bool GetOutput(std::vector<Tensor>& output) {
    std::vector<std::vector<Tensor>> model_outputs;
    if (stream_->GetOperatorOutput(model_op_, model_outputs)) {
      output.reserve(model_outputs.size());
      for (const auto& mo : model_outputs) {
        auto tensor_host = mo[0].Clone();
        auto tensor_fp32 = ConvertTensorFromFp16ToFp32(tensor_host);
        output.push_back(tensor_fp32);
      }
      return true;
    }
    return false;
  }
};
}  // namespace vsx