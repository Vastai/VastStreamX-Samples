
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

#include "common/model_cv.hpp"
#include "common/utils.hpp"

namespace vsx {

class MobileVit : public ModelCV {
 public:
  MobileVit(const std::string& model_prefix, const std::string& vdsp_config,
            uint32_t batch_size = 1, uint32_t device_id = 0,
            const std::string& hw_config = "")
      : ModelCV(model_prefix, vdsp_config, batch_size, device_id, hw_config) {
    for (auto& op : preproc_ops_) {
      if (op->GetOpType() >= 100) {
        fusion_op_ = static_cast<vsx::BuildInOperator*>(op.get());
        break;
      }
    }
    CHECK(fusion_op_ != nullptr)
        << "Can't find fusion_op in vdsp_op config json file.";
    model_->GetInputShapeByIndex(0, model_input_shape_);
    model_height_ = model_input_shape_[model_input_shape_.ndim() - 2];
    model_width_ = model_input_shape_[model_input_shape_.ndim() - 1];
    resize_height_ = static_cast<uint32_t>(256.0 / 224 * model_height_);
  }

 protected:
  std::vector<vsx::Tensor> ProcessImpl(const std::vector<vsx::Image>& images) {
    std::vector<vsx::Tensor> outputs;
    for (const auto& image : images) {
      int new_w, new_h;
      int img_w = image.Width();
      int img_h = image.Height();
      compute_size(img_w, img_h, resize_height_, new_w, new_h);
      int left = (new_w - model_width_) / 2;
      int top = (new_h - model_height_) / 2;
      fusion_op_->SetAttribute<vsx::AttrKey::kIimageWidth>(img_w);
      fusion_op_->SetAttribute<vsx::AttrKey::kIimageHeight>(img_h);
      fusion_op_->SetAttribute<vsx::AttrKey::kIimageWidthPitch>(img_w);
      fusion_op_->SetAttribute<vsx::AttrKey::kIimageHeightPitch>(img_h);
      fusion_op_->SetAttribute<vsx::AttrKey::kResizeWidth>(new_w);
      fusion_op_->SetAttribute<vsx::AttrKey::kResizeHeight>(new_h);
      fusion_op_->SetAttribute<vsx::AttrKey::kCropX>(left);
      fusion_op_->SetAttribute<vsx::AttrKey::kCropY>(top);

      auto model_outs = stream_->RunSync({image});
      for (const auto& output : model_outs) {
        auto tensor_host = output[0].Clone();
        auto tensor_fp32 = ConvertTensorFromFp16ToFp32(tensor_host);
        outputs.push_back(std::move(tensor_fp32));
      }
    }
    return outputs;
  }

  void compute_size(int img_w, int img_h, int size, int& new_w, int& new_h) {
    float r = std::max(size / static_cast<float>(img_w),
                       size / static_cast<float>(img_h));
    new_w = static_cast<int>(r * img_w);
    new_h = static_cast<int>(r * img_h);
  }

 protected:
  vsx::TShape model_input_shape_;
  uint32_t resize_height_ = 0;
  uint32_t model_width_ = 0, model_height_ = 0;
  vsx::BuildInOperator* fusion_op_ = nullptr;
};

}  // namespace vsx