
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include "custom_op_base.hpp"

namespace vsx {

typedef enum {
  DATA_TYPE_FP16,
  DATA_TYPE_INT8,
  DATA_TYPE_BFP16,
  DATA_TYPE_FLOAT32,
  DATA_TYPE_DOUBLE,
  DATA_TYPE_INT32,
  DATA_TYPE_RESERVED
} data_type_enum;

typedef enum {
  NCHW_layout,
  MatrixA_layout,
} layout_type;

typedef struct {
  uint32_t input_dims_num;
  uint32_t input_dims[4];
  uint32_t input_align_dims[4];
  layout_type input_layout;  // NCHW, N = 1
  data_type_enum input_type;
  uint32_t kh;  // block_size height.
  uint32_t kw;  // block size width.
  uint32_t output_dims_num;
  uint32_t output_dims[4];
  uint32_t output_align_dims[4];
  layout_type output_layout;  // MatrixA
} space_to_depth_t;

class SpaceToDepthOp : public CustomOpBase {
 public:
  SpaceToDepthOp(int kh, int kw, int oh_align, int ow_align,
                 const std::string& op_name = "opf_space_to_depth_out_matrix",
                 const std::string& elf_file =
                     "/opt/vastai/vastpipe/data/elf/space_to_depth",
                 uint32_t device_id = 0)
      : CustomOpBase(op_name, elf_file, device_id) {
    CHECK(oh_align / 16 * 16 == oh_align)
        << "oh_align " << oh_align << " must be aligned to 16";
    CHECK(ow_align / 16 * 16 == ow_align)
        << "ow_align " << ow_align << " must be aligned to 16";
    kh_ = kh;
    kw_ = kw;
    oh_align_ = oh_align;
    ow_align_ = ow_align;
  }

  vsx::Tensor Process(const vsx::Tensor& tensor) {
    std::vector<vsx::Tensor> tensors = {tensor};
    return std::move(Process(tensors)[0]);
  }

  std::vector<vsx::Tensor> Process(const std::vector<vsx::Tensor>& tensors) {
    std::vector<vsx::Tensor> tensors_vacc;
    tensors_vacc.reserve(tensors.size());
    for (auto& tensor : tensors) {
      if (tensor.GetContext().dev_type != vsx::Context::kVACC) {
        tensors_vacc.push_back(tensor.Clone(vsx::Context::VACC(device_id_)));
      } else {
        tensors_vacc.push_back(tensor);
      }
    }
    return std::move(ProcessImpl(tensors_vacc));
  }

 protected:
  virtual std::vector<vsx::Tensor> ProcessImpl(
      const std::vector<vsx::Tensor>& tensors) {
    std::vector<vsx::Tensor> results;
    for (const auto tensor : tensors) {
      auto shape = tensor.Shape();
      CHECK(shape.ndim() == 3 || shape.ndim() == 4);
      int n, c, h, w;
      if (shape.ndim() == 3) {
        n = 1;
        c = shape[0], h = shape[1], w = shape[2];
      } else {
        n = shape[0], c = shape[1], h = shape[2], w = shape[3];
      }

      int out_h = (h / kh_) * (w / kw_);
      int out_w = c * kh_ * kw_;

      CHECK(out_h <= oh_align_)
          << "error: real output height " << out_h
          << " must be smaller than oh_align " << oh_align_;
      CHECK(out_w <= ow_align_)
          << "error: real output height " << out_w
          << " must be smaller than oh_align " << ow_align_;

      space_to_depth_t op_conf;
      op_conf.input_dims_num = 4;
      op_conf.input_dims[0] = n;
      op_conf.input_dims[1] = c;
      op_conf.input_dims[2] = (static_cast<int>(h) / kh_) * kh_;
      op_conf.input_dims[3] = (static_cast<int>(w) / kw_) * kw_;
      op_conf.input_align_dims[0] = n;
      op_conf.input_align_dims[1] = c;
      op_conf.input_align_dims[2] = h;

      op_conf.input_align_dims[3] = w;
      op_conf.input_layout = layout_type::NCHW_layout;
      op_conf.input_type = data_type_enum::DATA_TYPE_FP16;
      op_conf.kh = kh_;
      op_conf.kw = kw_;
      op_conf.output_dims_num = 2;
      op_conf.output_dims[0] = out_h;
      op_conf.output_dims[1] = out_w;
      op_conf.output_align_dims[0] = oh_align_;
      op_conf.output_align_dims[1] = ow_align_;
      op_conf.output_layout = layout_type::MatrixA_layout;

      vsx::Tensor output_tensor({oh_align_, ow_align_},
                                vsx::Context::VACC(device_id_),
                                vsx::TypeFlag::kFloat16);
      std::vector<vsx::Tensor> inputs{tensor};
      std::vector<vsx::Tensor> outputs{output_tensor};

      custom_op_->RunSync(inputs, outputs, &op_conf, sizeof(space_to_depth_t));

      results.push_back(outputs[0]);
    }
    return results;
  }

 protected:
  int kh_, kw_, oh_align_, ow_align_;
};
}  // namespace vsx
