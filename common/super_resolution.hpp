
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <chrono>
#include <vector>

#include "custom_op_base.hpp"
#include "model_cv.hpp"
#include "opencv2/opencv.hpp"
#include "utils.hpp"
namespace vsx {

typedef struct {
  uint32_t iimage_width;
  uint32_t iimage_height;
} image_shape_t;

enum type_conversion {
  FLOAT16_TO_UINT8 = 0,  // (x * coef + base) * scale
};

typedef struct {
  image_shape_t iimage_shape;
  int32_t type;
  float threshold;
  int32_t scale;
  float coef;
  float base;
} type_conversion_t;

class PostProcessOp : public CustomOpBase {
 public:
  PostProcessOp(float coef = 1.0f, int scale = 1, float base = 0.0,
                uint32_t device_id = 0,
                const std::string& elf_file =
                    "/opt/vastai/vaststreamx/data/elf/postprocessimage",
                const std::string& op_name = "custom_op_tensor2image",
                float threshold = 0.0f)
      : CustomOpBase(op_name, elf_file, device_id) {
    coef_ = coef;
    scale_ = scale;
    base_ = base;
    threshold_ = threshold;
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
      int c, h, w;
      if (shape.ndim() == 3) {
        c = shape[0], h = shape[1], w = shape[2];
      } else {
        c = shape[1], h = shape[2], w = shape[3];
      }

      type_conversion_t op_conf;
      op_conf.iimage_shape.iimage_width = w;
      op_conf.iimage_shape.iimage_height = h * c;
      op_conf.type = FLOAT16_TO_UINT8;
      op_conf.threshold = threshold_;
      op_conf.scale = scale_;
      op_conf.base = base_;
      op_conf.coef = coef_;

      vsx::Tensor output_tensor({c, h, w}, vsx::Context::VACC(device_id_),
                                vsx::TypeFlag::kUint8);
      std::vector<vsx::Tensor> inputs{tensor};
      std::vector<vsx::Tensor> outputs{output_tensor};

      custom_op_->RunSync(inputs, outputs, &op_conf, sizeof(type_conversion_t));

      results.push_back(outputs[0]);
    }
    return results;
  }

 private:
  float coef_ = 1.0f;
  int scale_ = 1;
  float base_ = 0.0;
  float threshold_ = 0.0f;
};

class SuperResolution : public ModelCV {
 public:
  SuperResolution(const std::string& model_prefix,
                  const std::string& vdsp_config,
                  const std::string& postproc_elf =
                      "/opt/vastai/vaststreamx/data/elf/postprocessimage",
                  uint32_t device_id = 0, float coef = 1.0f, int scale = 1,
                  float base = 0.0, uint32_t batch_size = 1,
                  const std::string& hw_config = "")
      : ModelCV(model_prefix, vdsp_config, batch_size, device_id, hw_config) {
    postproc_op_ = std::make_shared<vsx::PostProcessOp>(
        coef, scale, base, device_id_, postproc_elf);
  }

 protected:
  virtual std::vector<vsx::Tensor> ProcessImpl(
      const std::vector<vsx::Image>& images) {
    auto outputs = stream_->RunSync(images);
    std::vector<vsx::Tensor> results;
    results.reserve(outputs.size());
    for (const auto& output : outputs) {
      auto post_out = PostProcess(output[0]);
      results.push_back(post_out.Clone());
    }
    return results;
  }

  vsx::Tensor PostProcess(const vsx::Tensor& input) {
    return postproc_op_->Process(input);
  }

 private:
  std::shared_ptr<vsx::PostProcessOp> postproc_op_;
};

}  // namespace vsx