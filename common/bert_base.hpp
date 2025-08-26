
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

#include "common/model_nlp.hpp"
#include "common/utils.hpp"

namespace vsx {
class Bert : public ModelNLP {
 public:
  Bert(const std::string& model_prefix, const std::string& vdsp_config,
       uint32_t batch_size = 1, uint32_t device_id = 0,
       const std::string& hw_config = "")
      : ModelNLP(model_prefix, vdsp_config, batch_size, device_id, hw_config) {}

 protected:
  std::vector<std::vector<vsx::Tensor>> ProcessImpl(
      const std::vector<std::vector<vsx::Tensor>>& tensors) {
    auto outputs = stream_->RunSync(tensors);
    std::vector<std::vector<vsx::Tensor>> results;
    results.reserve(outputs.size());
    for (const auto& output : outputs) {
      std::vector<vsx::Tensor> result;
      result.reserve(output.size());
      for (const auto& out : output) {
        auto tensor_host = out.Clone();
        auto tensor_fp32 = ConvertTensorFromFp16ToFp32(tensor_host);
        result.push_back(tensor_fp32);
      }
      results.push_back(result);
    }
    return std::move(results);
  }
};
}  // namespace vsx
