
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include "model_nlp.hpp"
#include "utils.hpp"
namespace vsx {
class ClipText : public ModelNLP {
 public:
  ClipText(const std::string& model_prefix, const std::string& vdsp_config,
           uint32_t batch_size = 1, uint32_t device_id = 0,
           const std::string& hw_config = "")
      : ModelNLP(model_prefix, vdsp_config, batch_size, device_id, hw_config) {}

  std::vector<std::vector<vsx::Tensor>> GetTestData(
      uint32_t bsize, uint32_t dtype, const Context& context,
      const std::vector<TShape>& input_shapes) {
    std::vector<std::vector<vsx::Tensor>> tensors;
    tensors.reserve(bsize);
    CHECK(test_tensors_.size()) << "Test data is empty, call SetCPUTestData "
                                   "api before profile clip_text model.";
    if (context.dev_type == vsx::Context::kVACC) {
      for (uint32_t i = 0; i < bsize; i++) {
        std::vector<vsx::Tensor> tmp_vacc;
        for (auto tensor : test_tensors_) {
          tmp_vacc.push_back(std::move(tensor.Clone(context)));
        }
        tensors.push_back(tmp_vacc);
      }
    } else {
      for (uint32_t i = 0; i < bsize; i++) {
        tensors.push_back(test_tensors_);
      }
    }
    return tensors;
  }

  int SetCPUTestData(const std::vector<vsx::Tensor>& test_tensors) {
    std::vector<vsx::Tensor> empty;
    std::swap(empty, test_tensors_);
    for (const auto& tensor : test_tensors) {
      CHECK(tensor.GetContext().dev_type == vsx::Context::kCPU);
      test_tensors_.push_back(tensor);
    }
    return 0;
  }

 protected:
  std::vector<std::vector<vsx::Tensor>> ProcessImpl(
      const std::vector<std::vector<vsx::Tensor>>& tensors) {
    auto outputs = stream_->RunSync(tensors);
    std::vector<std::vector<vsx::Tensor>> results;
    results.reserve(outputs.size());
    for (const auto& output : outputs) {
      std::vector<vsx::Tensor> result;
      for (const auto& out : output) {
        auto tensor_host = out.Clone();
        auto tensor_fp32 = vsx::ConvertTensorFromFp16ToFp32(tensor_host);
        result.push_back(tensor_fp32);
      }
      results.push_back(result);
    }
    return results;
  }

 protected:
  std::vector<vsx::Tensor> test_tensors_;
};

}  // namespace vsx
