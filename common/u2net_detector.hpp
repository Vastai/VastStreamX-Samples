
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <algorithm>
#include <typeinfo>
#include <vector>

#include "common/model_cv.hpp"
#include "common/utils.hpp"

namespace vsx {

class U2netDetector : public ModelCV {
 public:
  U2netDetector(const std::string& model_prefix, const std::string& vdsp_config,
                uint32_t batch_size = 1, uint32_t device_id = 0,
                const std::string& hw_config = "")
      : ModelCV(model_prefix, vdsp_config, batch_size, device_id, hw_config) {}

 protected:
  std::vector<vsx::Tensor> ProcessImpl(const std::vector<vsx::Image>& images) {
    auto outputs = stream_->RunSync(images);
    std::vector<vsx::Tensor> results;
    results.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++) {
      const auto& output = outputs[i];
      std::vector<vsx::Tensor> tensor_host;
      for (const auto& out : output) {
        tensor_host.push_back(out.Clone());
      }
      results.push_back(PostProcess(tensor_host));
    }
    return results;
  }

  vsx::Tensor PostProcess(const std::vector<vsx::Tensor>& fp16_tensors) {
    vsx::Tensor fp32_tensor = vsx::ConvertTensorFromFp16ToFp32(fp16_tensors[0]);
    const float* output_ptr = fp32_tensor.Data<float>();

    vsx::Tensor mask(fp32_tensor.Shape(), vsx::Context::CPU(0), vsx::kFloat32);
    float* mask_ptr = mask.MutableData<float>();

    normPRED(mask_ptr, output_ptr, fp32_tensor.GetSize());
    return mask;
  }

  void normPRED(float* mask, const float* model_out, int length) {
    int max_index = std::max_element(model_out, model_out + length) - model_out;
    int min_index = std::min_element(model_out, model_out + length) - model_out;
    auto maxV = *(model_out + max_index);
    auto minV = *(model_out + min_index);
    auto divider = maxV - minV;
    for (int i = 0; i < length; i++) {
      auto v = model_out[i];
      v = (v - minV) / divider;
      mask[i] = v;
    }
    return;
  }
};

}  // namespace vsx