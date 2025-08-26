
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "common/custom_op_base.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"
namespace vsx {
uint32_t getInputCount(const char* op_name) { return 5; }
uint32_t getOutputCount(const char* op_name) { return 6; }

vsx::CustomOperatorCallback callback{
    getInputCount, nullptr, getOutputCount, nullptr, nullptr, nullptr, 0, 0};

// 算子参数的结构体
struct image_shape_layout_t {
  int height;
  int width;
  int h_pitch;
  int w_pitch;
};

struct yolov8_seg_op_t {
  image_shape_layout_t model_in_shape;
  image_shape_layout_t model_out_shape;
  image_shape_layout_t origin_image_shape;
  uint32_t k;
  uint32_t retina_masks;
  uint32_t max_detect_num;
};

class Yolov8SegPostProcOp : public CustomOpBase {
 public:
  Yolov8SegPostProcOp(const std::string& op_name, const std::string& elf_file,
                      uint32_t device_id = 0, bool retina_masks = true)
      : CustomOpBase(op_name, elf_file, device_id),
        retina_masks_(retina_masks) {
    custom_op_->SetCallback(callback);
  }

 public:
  std::vector<vsx::Tensor> Process(
      const std::vector<vsx::Tensor>& model_outputs,
      const vsx::TShape& model_input_shape, const vsx::TShape& image_shape) {
    const auto& mask_shape = model_outputs[4].Shape();
    const auto& classes_shape = model_outputs[0].Shape();
    int model_in_height = model_input_shape[model_input_shape.ndim() - 2];
    int model_in_width = model_input_shape[model_input_shape.ndim() - 1];
    int model_out_height = mask_shape[2];
    int model_out_width = mask_shape[3];
    int image_height = image_shape[image_shape.ndim() - 2];
    int image_width = image_shape[image_shape.ndim() - 1];
    int max_detect_num = classes_shape[1];
    int mask_ch_num = 32;  // Don't change this parameter
    yolov8_seg_op_t op_conf;
    op_conf.model_in_shape.height = model_in_height;
    op_conf.model_in_shape.width = model_in_width;
    op_conf.model_in_shape.h_pitch = model_in_height;
    op_conf.model_in_shape.w_pitch = model_in_width;

    op_conf.model_out_shape.height = model_out_height;
    op_conf.model_out_shape.width = model_out_width;
    op_conf.model_out_shape.h_pitch = model_out_height;
    op_conf.model_out_shape.w_pitch = model_out_width;
    op_conf.origin_image_shape.height = image_height;
    op_conf.origin_image_shape.width = image_width;
    op_conf.origin_image_shape.h_pitch = image_height;
    op_conf.origin_image_shape.w_pitch = image_width;

    op_conf.k = mask_ch_num;
    op_conf.retina_masks = (retina_masks_ ? 1 : 0);
    op_conf.max_detect_num = max_detect_num;
    // custom_op_->SetConfig((void*)(&op_conf), sizeof(yolov8_seg_op_t));

    // determine mask's shape
    int mask_out_h = model_in_height, mask_out_w = model_in_width;
    if (retina_masks_) {
      mask_out_h = image_height;
      mask_out_w = image_width;
    }

    int buffer_size =
        (max_detect_num + 3) *
        std::max(model_in_width * model_in_height, image_height * image_width);
    vsx::Tensor classes({max_detect_num}, vsx::Context::VACC(device_id_),
                        vsx::TypeFlag::kFloat16);
    vsx::Tensor scores({max_detect_num}, vsx::Context::VACC(device_id_),
                       vsx::TypeFlag::kFloat16);
    vsx::Tensor boxes({max_detect_num, 4}, vsx::Context::VACC(device_id_),
                      vsx::TypeFlag::kFloat16);
    vsx::Tensor mask({max_detect_num, mask_out_h, mask_out_w},
                     vsx::Context::VACC(device_id_), vsx::TypeFlag::kUint8);
    vsx::Tensor nums({2}, vsx::Context::VACC(device_id_),
                     vsx::TypeFlag::kUint32);

    vsx::Tensor vdsp_buffer({buffer_size}, vsx::Context::VACC(device_id_),
                            vsx::TypeFlag::kUint8);

    std::vector<vsx::Tensor> outputs{classes, scores, boxes,
                                     mask,    nums,   vdsp_buffer};

    // custom_op_->Execute(model_outputs, outputs);
    custom_op_->RunSync(model_outputs, outputs, &op_conf,
                        sizeof(yolov8_seg_op_t));

    vsx::Tensor num_host = nums.Clone();
    uint32_t det_num = num_host.Data<uint32_t>()[0];
    if (det_num == 0) {
      return {};
    }
    vsx::Tensor classes_host({det_num}, vsx::Context::CPU(),
                             vsx::TypeFlag::kFloat16);
    vsx::Memcpy(classes.MutableData<void>(), classes_host.MutableData<void>(),
                classes_host.GetSize() * classes_host.GetSizeOfDType(),
                vsx::COPY_FROM_DEVICE);
    vsx::Tensor scores_host({det_num}, vsx::Context::CPU(),
                            vsx::TypeFlag::kFloat16);
    vsx::Memcpy(scores.MutableData<void>(), scores_host.MutableData<void>(),
                scores_host.GetSize() * scores_host.GetSizeOfDType(),
                vsx::COPY_FROM_DEVICE);
    vsx::Tensor boxes_host({det_num, 4}, vsx::Context::CPU(),
                           vsx::TypeFlag::kFloat16);
    vsx::Memcpy(boxes.MutableData<void>(), boxes_host.MutableData<void>(),
                boxes_host.GetSize() * boxes_host.GetSizeOfDType(),
                vsx::COPY_FROM_DEVICE);
    vsx::Tensor masks_host({det_num, mask_out_h, mask_out_w},
                           vsx::Context::CPU(), vsx::TypeFlag::kUint8);
    vsx::Memcpy(mask.MutableData<void>(), masks_host.MutableData<void>(),
                masks_host.GetSize() * masks_host.GetSizeOfDType(),
                vsx::COPY_FROM_DEVICE);

    std::vector<vsx::Tensor> fp32_host;
    fp32_host.push_back(ConvertTensorFromFp16ToFp32(classes_host));
    fp32_host.push_back(ConvertTensorFromFp16ToFp32(scores_host));
    fp32_host.push_back(ConvertTensorFromFp16ToFp32(boxes_host));

    fp32_host.emplace_back(masks_host);
    fp32_host.emplace_back(num_host);

    return fp32_host;
  }

 protected:
  bool retina_masks_ = true;
};

}  // namespace vsx
