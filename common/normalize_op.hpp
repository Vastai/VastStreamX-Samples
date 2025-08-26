
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
  NORMAL_EQUAL,             ///< R = R*scale_R; R/G/B is the same process as R;
  NORMAL_MINUSMEAN_DIVSTD,  ///< R = R*scale_R/std -mean_R*scale_R/std; R/G/B is
                            ///< the same process as R;
  NORMAL_DIV255_MINUSMEAN_DIVSTD,  ///< R = R*scale_R/(255*std)
                                   ///< -mean_R*scale_R/std; R/G/B is the same
                                   ///< process as R;
  NORMAL_DIV127_5_MINUSONE,  ///< R = R*scale_R/127.5 -1*scale_R; R/G/B is the
                             ///< same process as R;
  NORMAL_DIV255,  ///< R = R*scale_R/255; R/G/B is the same process as R;
} normal_type_t;
#define MAX_NORMA_CH_NUM 4
typedef struct {
  uint32_t width;             ///< width of the image
  uint32_t height;            ///< height of the image
  uint32_t in_width_pitch;    ///< input width pitch of the image, in pixel
  uint32_t in_height_pitch;   ///< input height pitch of the image, in pixel
  uint32_t out_width_pitch;   ///< output width pitch of the image, in pixel
  uint32_t out_height_pitch;  ///< output height pitch of the image,in pixel
  uint32_t ch_num;            ///< channel number; should be 1, 2, 3, or 4
  normal_type_t norma_type;   ///< normalization type; see
                              ///< resize_normal_quant_api.h for details
  uint16_t mean[MAX_NORMA_CH_NUM];  ///< mean of each channel; datatype is fp16
  uint16_t std[MAX_NORMA_CH_NUM];   ///< std of each channel; datatype is fp16
} opf_normalize_t;

class NormalizeOp : public CustomOpBase {
 public:
  NormalizeOp(
      const std::string& op_name = "opf_normalize",
      const std::string& elf_file = "/opt/vastai/vastpipe/data/elf/normalize",
      uint32_t device_id = 0, const std::vector<uint16_t>& mean_v = {},
      const std::vector<uint16_t>& std_v = {},
      normal_type_t normal_type = normal_type_t::NORMAL_EQUAL)
      : CustomOpBase(op_name, elf_file, device_id) {
    if (normal_type == normal_type_t::NORMAL_MINUSMEAN_DIVSTD) {
      CHECK(mean_v.size() == 3)
          << "len " << mean_v.size()
          << " of mean_v should be 3 when norm_tye is NORMAL_MINUSMEAN_DIVSTD";
      CHECK(std_v.size() == 3)
          << "len " << std_v.size()
          << " of std_v should be 3 when norm_tye is NORMAL_MINUSMEAN_DIVSTD";
    } else if (normal_type == normal_type_t::NORMAL_DIV255_MINUSMEAN_DIVSTD) {
      CHECK(mean_v.size() == 3) << "len " << mean_v.size()
                                << " of mean_v should be 3 when norm_tye is "
                                   "NORMAL_DIV255_MINUSMEAN_DIVSTD";
      CHECK(std_v.size() == 3) << "len " << std_v.size()
                               << " of std_v should be 3 when norm_tye is "
                                  "NORMAL_DIV255_MINUSMEAN_DIVSTD";
    }
    mean_ = mean_v;
    std_ = std_v;
    normal_type_ = normal_type;
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
      CHECK(shape.ndim() >= 3);
      int c, h, w;
      c = shape[shape.ndim() - 3];
      h = shape[shape.ndim() - 2];
      w = shape[shape.ndim() - 1];
      opf_normalize_t op_conf;
      op_conf.width = w;
      op_conf.height = h;
      op_conf.in_width_pitch = w;
      op_conf.in_height_pitch = h;
      op_conf.out_width_pitch = w;
      op_conf.out_height_pitch = h;
      op_conf.ch_num = c;
      op_conf.norma_type = normal_type_;
      if (normal_type_ == normal_type_t::NORMAL_DIV255_MINUSMEAN_DIVSTD ||
          normal_type_ == normal_type_t::NORMAL_MINUSMEAN_DIVSTD) {
        op_conf.mean[0] = mean_[0];
        op_conf.mean[1] = mean_[1];
        op_conf.mean[2] = mean_[2];
        op_conf.std[0] = std_[0];
        op_conf.std[1] = std_[1];
        op_conf.std[2] = std_[2];
      }

      vsx::Tensor output_tensor({c, h, w}, vsx::Context::VACC(device_id_),
                                vsx::TypeFlag::kFloat16);

      std::vector<vsx::Tensor> inputs{tensor};
      std::vector<vsx::Tensor> outputs{output_tensor};

      custom_op_->RunSync(inputs, outputs, &op_conf, sizeof(opf_normalize_t));

      results.push_back(outputs[0]);
    }
    return results;
  }

 protected:
  std::vector<uint16_t> mean_;
  std::vector<uint16_t> std_;
  normal_type_t normal_type_;
};
}  // namespace vsx