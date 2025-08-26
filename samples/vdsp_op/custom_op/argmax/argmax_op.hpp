
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "common/custom_op_base.hpp"
#include "opencv2/opencv.hpp"

namespace vsx {
uint32_t getInputCount(const char* op_name) { return 1; }
uint32_t getOutputCount(const char* op_name) { return 1; }

vsx::CustomOperatorCallback callback{
    getInputCount, nullptr, getOutputCount, nullptr, nullptr, nullptr, 0, 0};

typedef struct {
  uint64_t addr;

  //! currently channel number is limited to less than or equal to 96, argmax is
  //! performed through channel direction
  int32_t channel;
  int32_t height;
  int32_t width;
  int32_t h_pitch;
  int32_t w_pitch;
} planar_argmax_tensor_t;

typedef struct {
  // input is planar layout, c, h, w, element type is fp16
  planar_argmax_tensor_t in;
  // output is planar layout, 1, h, w, element type is int16_t
  planar_argmax_tensor_t out;
} planar_argmax_cfg_t;

class ArgmaxOp : public CustomOpBase {
 public:
  ArgmaxOp(const std::string& op_name, const std::string& elf_file,
           uint32_t device_id = 0)
      : CustomOpBase(op_name, elf_file, device_id) {
    custom_op_->SetCallback(callback);
  }

  vsx::Tensor Process(const vsx::Tensor& tensor) {
    std::vector<vsx::Tensor> tensors = {tensor};
    return Process(tensors)[0];
  }

  std::vector<vsx::Tensor> Process(const std::vector<vsx::Tensor>& tensors) {
    return ProcessImpl(tensors);
  }
  std::vector<vsx::Tensor> GetTestData(
      uint32_t bsize, uint32_t dtype, const Context& context,
      const std::vector<TShape>& input_shapes) {
    const auto& input_shape = input_shapes[0];
    std::vector<vsx::Tensor> images;
    images.reserve(bsize);

    auto tensor = vsx::Tensor(input_shape, context, dtype);
    for (uint32_t i = 0; i < bsize; i++) {
      images.push_back(tensor);
    }
    return images;
  }

 protected:
  virtual std::vector<vsx::Tensor> ProcessImpl(
      const std::vector<vsx::Tensor>& tensors) {
    std::vector<vsx::Tensor> results;
    for (const auto tensor : tensors) {
      auto shape = tensor.Shape();
      CHECK(shape.ndim() >= 3);
      auto width = shape[shape.ndim() - 1];
      auto height = shape[shape.ndim() - 2];
      auto channels = shape[shape.ndim() - 3];
      planar_argmax_cfg_t op_params = {0};

      vsx::Tensor tensor_vacc;
      if (tensor.GetContext().dev_type == vsx::Context::kVACC) {
        tensor_vacc = tensor;
      } else {
        tensor_vacc = tensor.Clone(vsx::Context::VACC(device_id_));
      }

      vsx::Tensor output_tensor({1, 1, height, width},
                                vsx::Context::VACC(device_id_),
                                vsx::TypeFlag::kUint16);

      op_params.in.addr = tensor_vacc.GetDataAddress();
      op_params.in.channel = channels;
      op_params.in.width = width;
      op_params.in.height = height;
      op_params.in.w_pitch = width;
      op_params.in.h_pitch = height;

      op_params.out.addr = output_tensor.GetDataAddress();
      op_params.out.channel = 1;
      op_params.out.width = width;
      op_params.out.height = height;
      op_params.out.w_pitch = width;
      op_params.out.h_pitch = height;

      // custom_op_->SetConfig(&op_params, sizeof(planar_argmax_cfg_t));
      std::vector<vsx::Tensor> inputs{tensor_vacc};
      std::vector<vsx::Tensor> outputs{output_tensor};
      // custom_op_->Execute(inputs, outputs);
      custom_op_->RunSync(inputs, outputs, &op_params,
                          sizeof(planar_argmax_cfg_t));
      results.push_back(outputs[0]);
    }
    return results;
  }
};
}  // namespace vsx
