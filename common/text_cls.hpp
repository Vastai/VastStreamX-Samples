
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "common/model_cv.hpp"
#include "common/utils.hpp"
#include "vaststreamx/common/common_def.h"

namespace vsx {

class TextClassifier : public ModelCV {
 public:
  TextClassifier(const std::string& model_prefix,
                 const std::string& vdsp_config,
                 const std::vector<uint32_t>& labels, uint32_t batch_size = 1,
                 uint32_t device_id = 0, const std::string& hw_config = "")
      : ModelCV(model_prefix, vdsp_config, batch_size, device_id, hw_config) {
    model_->GetInputShapeByIndex(0, input_shape_);
  }

 protected:
  std::vector<vsx::Tensor> ProcessImpl(const std::vector<vsx::Image>& images) {
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
    auto outputs = stream_->RunSync(images, conf);

    std::vector<vsx::Tensor> results;
    results.reserve(outputs.size());
    for (const auto& out : outputs) {
      auto tensor_host = out[0].Clone();
      auto tensor_fp32 = ConvertTensorFromFp16ToFp32(tensor_host);
      results.push_back(tensor_fp32);
    }
    return results;
  }

 private:
  vsx::TShape input_shape_;
};

}  // namespace vsx