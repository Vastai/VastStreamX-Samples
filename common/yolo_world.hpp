
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include "common/utils.hpp"
#include "yolo_world_image.hpp"
#include "yolo_world_post_process.hpp"
#include "yolo_world_text.hpp"
namespace vsx {
class YoloWorld {
 public:
  YoloWorld(const std::string& imgmod_prefix,
            const std::string& immod_vdsp_config,
            const std::string& txtmod_prefix,
            const std::string& txtmod_vdsp_config, uint32_t batch_size = 1,
            uint32_t device_id = 0, float score_thresh = 0.001,
            int nms_pre = 30000, float iou_thresh = 0.7,
            int max_per_image = 300, int nms_threads = 20,
            const std::string& imgmod_hw_config = "",
            const std::string& txtmod_hw_config = "") {
    CHECK(vsx::SetDevice(device_id) == 0)
        << "Failed to set device id: " << device_id;
    image_model_ = std::make_shared<vsx::YoloWorldImage>(
        imgmod_prefix, immod_vdsp_config, batch_size, device_id,
        imgmod_hw_config);
    text_model_ = std::make_shared<vsx::YoloWorldText>(
        txtmod_prefix, txtmod_vdsp_config, batch_size, device_id,
        txtmod_hw_config);
    device_id_ = device_id;

    score_threshold_ = score_thresh;
    nms_pre_ = nms_pre;
    iou_threshold_ = iou_thresh;
    max_per_image_ = max_per_image;
    nms_threads_ = nms_threads;
  }
  vsx::ImageFormat GetFusionOpIimageFormat() {
    return image_model_->GetFusionOpIimageFormat();
  }

  std::vector<std::vector<vsx::Tensor>> ProcessText(
      const std::vector<std::vector<vsx::Tensor>>& tokens) {
    return text_model_->Process(tokens);
  }
  std::vector<vsx::Tensor> ProcessImage(
      const vsx::Image& image,
      const std::vector<std::vector<vsx::Tensor>>& text_features) {
    vsx::Tensor txt_tensor({static_cast<int>(text_features.size()), 512},
                           vsx::Context::CPU(), vsx::TypeFlag::kFloat16);
    char* dst = txt_tensor.MutableData<char>();
    for (auto& feat : text_features) {
      memcpy(dst, feat[0].Data<void>(), feat[0].GetDataBytes());
      dst += feat[0].GetDataBytes();
    }

    auto align_tensor = vsx::bert_get_activation_fp16_A(txt_tensor);
    auto img_features =
        image_model_->Process(std::make_pair(image, align_tensor));

    auto txt_feature_fp32 = vsx::ConvertTensorFromFp16ToFp32(txt_tensor);

    auto result = PostProcess(img_features, txt_feature_fp32, image.Width(),
                              image.Height());
    return result;
  }

  std::vector<vsx::Tensor> Process(
      const vsx::Image& image,
      const std::vector<std::vector<vsx::Tensor>>& tokens) {
    auto txt_features = text_model_->Process(tokens);

    vsx::Tensor txt_tensor({static_cast<int>(txt_features.size()), 512},
                           vsx::Context::CPU(), vsx::TypeFlag::kFloat16);
    char* dst = txt_tensor.MutableData<char>();
    for (auto& feat : txt_features) {
      memcpy(dst, feat[0].Data<void>(), feat[0].GetDataBytes());
      dst += feat[0].GetDataBytes();
    }

    auto align_tensor = vsx::bert_get_activation_fp16_A(txt_tensor);
    auto img_features =
        image_model_->Process(std::make_pair(image, align_tensor));

    auto txt_feature_fp32 = vsx::ConvertTensorFromFp16ToFp32(txt_tensor);

    auto result = PostProcess(img_features, txt_feature_fp32, image.Width(),
                              image.Height());
    return result;
  }

  std::vector<vsx::Tensor> PostProcess(
      const std::vector<vsx::Tensor>& img_features,
      const vsx::Tensor& txt_feature, int image_width, int image_height) {
    auto scores = GetScoresBatch(img_features, txt_feature);
    int model_width = image_model_->input_shape_[3];
    int model_height = image_model_->input_shape_[2];

    float scale_factor = std::min(model_height * 1.0f / image_height,
                                  model_width * 1.0f / image_width);
    float unpad_w = image_width * scale_factor;
    float unpad_h = image_height * scale_factor;
    float dw = (model_width - unpad_w) / 2;
    float dh = (model_height - unpad_h) / 2;

    int count = img_features.size();

    auto result =
        GetPostProcess(scores,
                       {
                           img_features[count - 3],
                           img_features[count - 2],
                           img_features[count - 1],
                       },
                       scale_factor, {dh, dh, dw, dw}, score_threshold_,
                       nms_pre_, iou_threshold_, max_per_image_, nms_threads_);
    return result;
  }

 protected:
  uint32_t device_id_;
  std::shared_ptr<vsx::YoloWorldImage> image_model_;
  std::shared_ptr<vsx::YoloWorldText> text_model_;

  float score_threshold_;
  float iou_threshold_;
  int nms_pre_;
  int max_per_image_;
  int nms_threads_;
};
}  // namespace vsx
