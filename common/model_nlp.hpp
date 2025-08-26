
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "common/model_base.hpp"

namespace vsx {
class ModelNLP : public ModelBase {
 public:
  ModelNLP(const std::string& model_prefix, const std::string& vdsp_config,
           uint32_t batch_size = 1, uint32_t device_id = 0,
           const std::string& hw_config = "",
           vsx::GraphOutputType output_type =
               vsx::GraphOutputType::kGRAPH_OUTPUT_TYPE_NCHW_DEVICE)
      : ModelBase(model_prefix, vdsp_config, batch_size, device_id, hw_config,
                  output_type) {}

  std::vector<vsx::Tensor> Process(const std::vector<vsx::Tensor>& tensors) {
    return ProcessImpl({tensors})[0];
  }
  std::vector<std::vector<vsx::Tensor>> Process(
      const std::vector<std::vector<vsx::Tensor>>& tensors) {
    return ProcessImpl(tensors);
  }
  std::vector<std::vector<vsx::Tensor>> GetTestData(
      uint32_t bsize, uint32_t dtype, const Context& context,
      const std::vector<TShape>& input_shapes) {
    std::vector<std::vector<vsx::Tensor>> tensors;
    tensors.reserve(bsize);

    for (uint32_t i = 0; i < bsize; i++) {
      std::vector<vsx::Tensor> temp;
      temp.reserve(input_shapes.size());
      for (auto& shape : input_shapes) {
        temp.emplace_back(shape, context, dtype);
      }
      tensors.push_back(temp);
    }
    return tensors;
  }

 protected:
  virtual std::vector<std::vector<vsx::Tensor>> ProcessImpl(
      const std::vector<std::vector<vsx::Tensor>>& tensors) = 0;
};
}  // namespace vsx
