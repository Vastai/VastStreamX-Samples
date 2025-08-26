
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include "common/clip_image.hpp"
#include "common/clip_text.hpp"
#include "common/utils.hpp"

namespace vsx {
class ClipModel {
 public:
  ClipModel(const std::string& imgmod_prefix, const std::string& norm_elf,
            const std::string& space2depth_elf,
            const std::string& txtmod_prefix,
            const std::string& txtmod_vdsp_config, uint32_t batch_size = 1,
            uint32_t device_id = 0, const std::string& imgmod_hw_config = "",
            const std::string& txtmod_hw_config = "") {
    CHECK(vsx::SetDevice(device_id) == 0)
        << "Failed to set device id: " << device_id;
    image_model_ = std::make_shared<vsx::ClipImage>(
        imgmod_prefix, norm_elf, space2depth_elf, batch_size, device_id,
        imgmod_hw_config);
    text_model_ = std::make_shared<vsx::ClipText>(
        txtmod_prefix, txtmod_vdsp_config, batch_size, device_id,
        txtmod_hw_config);
    device_id_ = device_id;
  }

  vsx::Tensor ProcessImage(const cv::Mat& input) {
    return image_model_->Process(input);
  }
  std::vector<vsx::Tensor> ProcessImage(const std::vector<cv::Mat>& inputs) {
    return image_model_->Process(inputs);
  }
  vsx::Tensor ProcessImage(const vsx::Image& input) {
    return image_model_->Process(input);
  }
  std::vector<vsx::Tensor> ProcessImage(const std::vector<vsx::Image>& inputs) {
    return image_model_->Process(inputs);
  }

  vsx::Tensor ProcessText(const std::vector<vsx::Tensor>& input) {
    std::vector<std::vector<vsx::Tensor>> inputs{input};
    return ProcessText(inputs)[0];
  }
  std::vector<vsx::Tensor> ProcessText(
      const std::vector<std::vector<vsx::Tensor>>& inputs) {
    auto outputs = text_model_->Process(inputs);
    // squeeze
    std::vector<vsx::Tensor> txt_feats;
    for (auto txt_feat : outputs) {
      txt_feats.push_back(txt_feat[0]);
    }
    return txt_feats;
  }

  vsx::Tensor Process(const cv::Mat& image,
                      const std::vector<std::vector<vsx::Tensor>>& tokens) {
    auto img_feature = ProcessImage(image);
    auto txt_features = ProcessText(tokens);

    return PostProcess(img_feature, txt_features);
  }
  vsx::Tensor Process(const vsx::Image& image,
                      const std::vector<std::vector<vsx::Tensor>>& tokens) {
    auto img_feature = ProcessImage(image);
    auto txt_features = ProcessText(tokens);

    return PostProcess(img_feature, txt_features);
  }
  vsx::Tensor PostProcess(const vsx::Tensor& img_feature,
                          const std::vector<vsx::Tensor>& txt_features) {
    std::vector<double> scores;
    scores.reserve(txt_features.size());
    for (auto txt_feat : txt_features) {
      double sum = 0;
      int len = txt_feat.GetSize();
      const float* img_data = img_feature.Data<float>();
      const float* txt_data = txt_feat.Data<float>();
      for (int i = 0; i < len; i++) {
        sum += img_data[i] * txt_data[i] * 100.00000762939453;
      }
      scores.push_back(sum);
    }
    // softmax
    auto result = vsx::softmax(scores);
    vsx::Tensor out({(int64)result.size()}, vsx::Context::CPU(),
                    vsx::TypeFlag::kFloat32);
    float* out_data = out.MutableData<float>();
    for (size_t i = 0; i < result.size(); i++) {
      out_data[i] = static_cast<float>(result[i]);
    }
    return out;
  }

 protected:
  uint32_t device_id_;
  std::shared_ptr<vsx::ClipImage> image_model_;
  std::shared_ptr<vsx::ClipText> text_model_;
};
}  // namespace vsx
