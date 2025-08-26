
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

class ClassifierAsync : public ModelCVAsync {
 public:
  ClassifierAsync(const std::string& model_prefix,
                  const std::string& vdsp_config, uint32_t batch_size = 1,
                  uint32_t device_id = 0, const std::string& hw_config = "")
      : ModelCVAsync(model_prefix, vdsp_config, batch_size, device_id,
                     hw_config) {
    int width, height;
    GetFusionOpResize(width, height);
    resize_size_ = width > height ? width : height;
    vsx::TShape input_shape;
    GetInputShapeByIndex(0, input_shape);
    model_size_ = input_shape[input_shape.ndim() - 1];
  }

  int GetFusionOpResize(int& resize_width, int& resize_height) {
    int ret = -1;
    resize_width = -1;
    resize_height = -1;
    for (auto op : preproc_ops_) {
      if (op->GetOpType() >= 100) {
        auto attri_keys = op->GetAttrKeys();
        if (vsx::HasAttribute(attri_keys, "kResizeWidth")) {
          auto fusion_op = static_cast<vsx::BuildInOperator*>(op.get());
          fusion_op->GetAttribute<vsx::AttrKey::kResizeWidth>(resize_width);
          ret = 0;
        }
        if (vsx::HasAttribute(attri_keys, "kResizeHeight")) {
          auto fusion_op = static_cast<vsx::BuildInOperator*>(op.get());
          fusion_op->GetAttribute<vsx::AttrKey::kResizeHeight>(resize_height);
          ret = 0;
        }
      }
    }
    CHECK(resize_width != -1 && resize_height != -1)
        << "Can't find ResizeWidth or ResizeHeight in fusion_op";
    return ret;
  }
  void GetResize(int img_w, int img_h, int model_size, int& resize_w,
                 int& resize_h) {
    if (img_w > img_h) {
      resize_w = model_size * img_w / img_h;
      resize_h = model_size;
    } else {
      resize_h = model_size * img_h / img_w;
      resize_w = model_size;
    }
  }
  void GetCrop(int img_w, int img_h, int model_size, int& crop_x, int& crop_y) {
    crop_x = (img_w - model_size + 1) / 2;
    crop_y = (img_h - model_size + 1) / 2;
  }
  uint32_t ProcessAsyncImpl(const std::vector<vsx::Image>& images) {
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
    return stream_->RunAsync(images, extra_configs);
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

 private:
  int resize_size_ = 0;
  int model_size_ = 0;
};

}  // namespace vsx