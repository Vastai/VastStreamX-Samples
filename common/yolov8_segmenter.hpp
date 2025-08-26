
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

#include "common/model_cv2.hpp"
#include "common/utils.hpp"
#include "common/yolov8_seg_post_proc_op.hpp"

namespace vsx {

class Yolov8Segmenter : public ModelCV2 {
 public:
  Yolov8Segmenter(const std::string &model_prefix,
                  const std::string &vdsp_config, const std::string &elf_file,
                  uint32_t batch_size = 1, uint32_t device_id = 0,
                  const std::string &hw_config = "")
      : ModelCV2(model_prefix, vdsp_config, batch_size, device_id, hw_config,
                 vsx::GraphOutputType::kGRAPH_OUTPUT_TYPE_NCHW_HOST) {
    post_proc_op_ = std::make_shared<vsx::Yolov8SegPostProcOp>(
        "yolov8_seg_op", elf_file, device_id);
    model_->GetInputShapeByIndex(0, input_shape_);
  }

 protected:
  std::vector<std::vector<vsx::Tensor>> ProcessImpl(
      const std::vector<vsx::Image> &images) {
    auto outputs = stream_->RunSync(images);

    std::vector<std::vector<vsx::Tensor>> results;
    results.reserve(outputs.size());
    for (size_t i = 0; i < images.size(); i++) {
      results.push_back(
          PostProcess(outputs[i], images[i].Width(), images[i].Height()));
    }
    return results;
  }

  std::vector<vsx::Tensor> PostProcess(
      const std::vector<vsx::Tensor> &fp16_tensors, int image_width,
      int image_height) {
    std::vector<vsx::Tensor> tensors;
    for (auto &tensor : fp16_tensors) {
      vsx::Tensor vacc_tensor = tensor;
      if (tensor.GetContext().dev_type == vsx::Context::kCPU) {
        vacc_tensor = tensor.Clone(vsx::Context::VACC(device_id_));
      }
      tensors.push_back(vacc_tensor);
    }

    return post_proc_op_->Process(tensors, input_shape_,
                                  {image_height, image_width});
  }

 private:
  vsx::TShape input_shape_;
  std::shared_ptr<vsx::Yolov8SegPostProcOp> post_proc_op_;
};
}  // namespace vsx