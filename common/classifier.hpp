
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

class Classifier : public ModelCV {
 public:
  Classifier(const std::string& model_prefix, const std::string& vdsp_config,
             uint32_t batch_size = 1, uint32_t device_id = 0,
             const std::string& hw_config = "")
      : ModelCV(model_prefix, vdsp_config, batch_size, device_id, hw_config) {}

 protected:
  std::vector<vsx::Tensor> ProcessImpl(const std::vector<vsx::Image>& images) {
    auto outputs = stream_->RunSync(images);
    std::vector<vsx::Tensor> results;
    results.reserve(outputs.size());
    for (const auto& out : outputs) {
      auto tensor_host = out[0].Clone();
      auto tensor_fp32 = ConvertTensorFromFp16ToFp32(tensor_host);
      results.push_back(tensor_fp32);
    }
    return results;
  }
};

}  // namespace vsx