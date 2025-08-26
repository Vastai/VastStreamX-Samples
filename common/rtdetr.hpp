
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

const int kDetectionOffset = 6 + 1;

class RtDetrModel : public ModelCV {
 public:
  RtDetrModel(const std::string &model_prefix, const std::string &vdsp_config,
              uint32_t batch_size = 1, uint32_t device_id = 0,
              float threshold = 0.2, const std::string &hw_config = "")
      : ModelCV(model_prefix, vdsp_config, batch_size, device_id, hw_config),
        threshold_(threshold) {
    model_->GetInputShapeByIndex(0, input_shape_);
  }

  void SetThreshold(float threshold) { threshold_ = threshold; }

 protected:
  std::vector<vsx::Tensor> ProcessImpl(const std::vector<vsx::Image> &images) {
    auto outputs = stream_->RunSync(images);
    std::vector<vsx::Tensor> results;
    results.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++) {
      const auto &output = outputs[i];
      std::vector<vsx::Tensor> tensors_host;
      for (const auto &out : output) {
        tensors_host.push_back(out.Clone());
      }
      auto tensors_fp32 = ConvertTensorFromFp16ToFp32(tensors_host);
      results.push_back(
          PostProcess(tensors_fp32, images[i].Width(), images[i].Height()));
    }
    return results;
  }

  std::vector<std::vector<double>> sigmoid(const vsx::Tensor &tensor) {
    const float *data = tensor.Data<float>();
    int cols = tensor.Shape()[tensor.Shape().ndim() - 1];
    int rows = tensor.Shape()[tensor.Shape().ndim() - 2];

    std::vector<std::vector<double>> result;
    result.reserve(rows);

    for (int r = 0; r < rows; r++) {
      std::vector<double> res;
      res.reserve(cols);
      for (int c = 0; c < cols; c++) {
        int index = r * cols + c;
        res.push_back(vsx::sigmoid(data[index]));
      }
      result.push_back(std::move(res));
    }
    return result;
  }

  vsx::Tensor PostProcess(const std::vector<vsx::Tensor> &fp32_tensors,
                          int image_width, int image_height) {
    auto out_logits = fp32_tensors[0];
    auto out_bbox = fp32_tensors[1];
    auto prob = sigmoid(out_logits);

    std::vector<float> scores;
    std::vector<int> class_ids;
    scores.reserve(prob.size());
    class_ids.reserve(prob.size());

    for (size_t i = 0; i < prob.size(); i++) {
      auto v_prob = prob[i];
      double max = v_prob[0];
      int max_id = 0;
      for (size_t j = 1; j < v_prob.size() - 1; j++) {
        if (max < v_prob[j]) {
          max = v_prob[j];
          max_id = j;
        }
      }
      scores.push_back(static_cast<float>(max));
      class_ids.push_back(max_id);
    }

    int data_count = scores.size();
    vsx::Tensor result({data_count, kDetectionOffset}, vsx::Context::CPU(0),
                       vsx::kFloat32);

    const float *boxes_data = out_bbox.Data<float>();
    float *result_ptr = result.MutableData<float>();

    for (int i = 0; i < data_count; i++) {
      float score = scores[i];
      if (score >= threshold_) {
        float xc = boxes_data[4 * i + 0];
        float yc = boxes_data[4 * i + 1];
        float w = boxes_data[4 * i + 2];
        float h = boxes_data[4 * i + 3];

        float xmin = (xc - 0.5 * w) * image_width;
        float ymin = (yc - 0.5 * h) * image_height;
        float xmax = (xc + 0.5 * w) * image_width;
        float ymax = (yc + 0.5 * h) * image_height;

        float bbox_xmin =
            std::min(std::max(xmin, 0.0f), static_cast<float>(image_width));
        float bbox_ymin =
            std::min(std::max(ymin, 0.0f), static_cast<float>(image_height));
        float bbox_xmax =
            std::min(std::max(xmax, 0.0f), static_cast<float>(image_width));
        float bbox_ymax =
            std::min(std::max(ymax, 0.0f), static_cast<float>(image_height));

        result_ptr[0] = class_ids[i];
        result_ptr[1] = score;
        result_ptr[2] = bbox_xmin;
        result_ptr[3] = bbox_ymin;
        result_ptr[4] = bbox_xmax - bbox_xmin;
        result_ptr[5] = bbox_ymax - bbox_ymin;
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
  vsx::TShape input_shape_;
};

}  // namespace vsx