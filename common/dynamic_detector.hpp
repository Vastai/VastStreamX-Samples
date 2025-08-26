
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

#include "common/dynamic_model_cv.hpp"
#include "common/utils.hpp"

namespace vsx {

const int kDetectionOffset = 6 + 1;

class DynamicDetector : public DynamicModelCV {
 public:
  DynamicDetector(const std::string &module_info,
                  const std::string &vdsp_config,
                  const std::vector<vsx::TShape> &max_input_shape,
                  uint32_t batch_size = 1, uint32_t device_id = 0,
                  float threshold = 0.2)
      : DynamicModelCV(module_info, vdsp_config, max_input_shape, batch_size,
                       device_id),
        threshold_(threshold) {}

  void SetThreshold(float threshold) { threshold_ = threshold; }

 protected:
  std::vector<vsx::Tensor> ProcessImpl(const std::vector<vsx::Image> &images) {
    CHECK(images.size() == input_shape_.size());
    StreamExtraRuntimeConfig ext_configs;
    for (auto &shape : input_shape_) {
      ext_configs.dynamic_model_input_shapes.push_back({TShapeToVector(shape)});
    }
    auto outputs = stream_->RunSync(images, ext_configs);
    std::vector<vsx::Tensor> results;
    results.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++) {
      const auto &output = outputs[i];
      std::vector<vsx::Tensor> tensor_host;
      for (const auto &out : output) {
        tensor_host.push_back(out.Clone());
      }
      auto tensor_fp32 = ConvertTensorFromFp16ToFp32(tensor_host);
      results.push_back(PostProcess(tensor_fp32, threshold_, input_shape_[0][3],
                                    input_shape_[0][2], images[i].Width(),
                                    images[i].Height()));
    }
    return results;
  }

  vsx::Tensor PostProcess(const std::vector<vsx::Tensor> &fp32_tensors,
                          float threshold, int model_width, int model_height,
                          int image_width, int image_height) {
    int data_count = fp32_tensors[0].GetSize();
    vsx::Tensor result({data_count, kDetectionOffset}, vsx::Context::CPU(0),
                       vsx::kFloat32);
    float *result_ptr = result.MutableData<float>();
    result_ptr[0] = -1;
    // check tensor size validation
    if (fp32_tensors[0].GetSize() != fp32_tensors[1].GetSize() ||
        fp32_tensors[1].GetSize() * 4 != fp32_tensors[2].GetSize()) {
      LOG(ERROR) << "Output tensor size error, sizes are: "
                 << fp32_tensors[0].GetSize() << ","
                 << fp32_tensors[1].GetSize() << ","
                 << fp32_tensors[2].GetSize();
      return result;
    }
    const float *class_data = fp32_tensors[0].Data<float>();
    const float *score_data = fp32_tensors[1].Data<float>();
    const float *bbox_data = fp32_tensors[2].Data<float>();
    float r = std::min(model_width * 1.0 / image_width,
                       model_height * 1.0 / image_height);

    float unpad_w = image_width * r;
    float unpad_h = image_height * r;
    float dw = (model_width - unpad_w) / 2;
    float dh = (model_height - unpad_h) / 2;

    for (int i = 0; i < data_count; i++) {
      int category = static_cast<int>(class_data[i]);
      if (category < 0) break;
      float score = score_data[i];
      if (score >= threshold) {
        float xmin = bbox_data[4 * i + 0];
        float ymin = bbox_data[4 * i + 1];
        float xmax = bbox_data[4 * i + 2];
        float ymax = bbox_data[4 * i + 3];
        float bbox_xmin = (xmin - dw) / r;
        bbox_xmin = std::max(bbox_xmin, 0.0f);
        bbox_xmin = std::min(bbox_xmin, static_cast<float>(image_width));
        float bbox_ymin = (ymin - dh) / r;
        bbox_ymin = std::max(bbox_ymin, 0.0f);
        bbox_ymin = std::min(bbox_ymin, static_cast<float>(image_height));
        float bbox_xmax = (xmax - dw) / r;
        bbox_xmax = std::max(bbox_xmax, 0.0f);
        bbox_xmax = std::min(bbox_xmax, static_cast<float>(image_width));
        float bbox_ymax = (ymax - dh) / r;
        bbox_ymax = std::max(bbox_ymax, 0.0f);
        bbox_ymax = std::min(bbox_ymax, static_cast<float>(image_height));
        float bbox_width = bbox_xmax - bbox_xmin;
        float bbox_height = bbox_ymax - bbox_ymin;
        result_ptr[0] = category;
        result_ptr[1] = score;
        result_ptr[2] = bbox_xmin;
        result_ptr[3] = bbox_ymin;
        result_ptr[4] = bbox_width;
        result_ptr[5] = bbox_height;
        result_ptr += kDetectionOffset;
      }
    }
    if (size_t(result_ptr - result.Data<float>()) < result.GetSize()) {
      result_ptr[0] = -1;
    }
    return result;
  }

 private:
  float threshold_ = 0.2;
};

}  // namespace vsx