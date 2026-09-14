
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <iostream>
#include <memory>

#include "common/custom_op_base.hpp"
#define MAX_CHAN 4

namespace vsx {

typedef enum {
  BORDER_CONSTANT = 0,
} BorderType;

typedef enum {
  //    WARP_PERSPECTIVE_INTER_NEAREST = 0,
  INTER_LINEAR = 1,
  //    WARP_PERSPECTIVE_INTER_CUBIC   = 2,
} InterpolationType;

typedef enum {
  FORWARD_MAP = 0,
  INVERSE_MAP = 16,
} Flags;

typedef enum {
  PLANAR = 0,
} ImageType;

typedef struct {
  int32_t channel;
  int32_t height;
  int32_t width;
  int32_t c_pitch;
  int32_t h_pitch;
  int32_t w_pitch;
} image_shape_t;

typedef struct {
  int32_t c_start;
  int32_t h_start;
  int32_t w_start;
  int32_t c_len;
  int32_t h_len;
  int32_t w_len;
} roi_t;  // region of interest of output, for multi core processing

typedef struct {
  image_shape_t in_image_shape;   //
  image_shape_t out_image_shape;  //
  ImageType image_type;
  double M[9];  // transform matrix
  InterpolationType inter_type;
  BorderType border_type;
  int32_t border_value;
  Flags flags;
  roi_t roi;
} warp_perspective_param_t;

class WarpPerspectiveOp : public CustomOpBase {
  // 仅支持 RGB PLANAR 格式
 public:
  WarpPerspectiveOp(
      const std::string& elf_file = "/opt/vastai/vaststream/lib/op/vdsp_op",
      uint32_t device_id = 0,
      const std::string& op_name = "warp_perspective_u8_op")
      : CustomOpBase(op_name, elf_file, device_id) {}

  vsx::Image Process(const vsx::Image& image_rgb_planar,
                     const std::vector<double>& matrix, int crop_width,
                     int crop_height) {
    if (matrix.size() != 9) {
      std::cerr << "Error! matrix size must be 9, now it's " << matrix.size()
                << std::endl;
      return vsx::Image();
    }

    int in_width = image_rgb_planar.Width();
    int in_height = image_rgb_planar.Height();
    int out_width = crop_width;
    int out_height = crop_height;

    warp_perspective_param_t op_param = {0};

    op_param.in_image_shape.channel = 3;  // RGB_PLANAR
    op_param.in_image_shape.height = in_height;
    op_param.in_image_shape.width = in_width;
    op_param.in_image_shape.c_pitch = 3;
    op_param.in_image_shape.h_pitch = in_height;
    op_param.in_image_shape.w_pitch = in_width;

    op_param.out_image_shape.channel = 3;
    op_param.out_image_shape.height = out_height;
    op_param.out_image_shape.width = out_width;
    op_param.out_image_shape.c_pitch = 3;
    op_param.out_image_shape.h_pitch = out_height;
    op_param.out_image_shape.w_pitch = out_width;

    for (int i = 0; i < 9; i++) {
      op_param.M[i] = matrix[i];
    }

    op_param.image_type = ImageType::PLANAR;
    op_param.inter_type = InterpolationType::INTER_LINEAR;
    op_param.border_type = BorderType::BORDER_CONSTANT;
    op_param.border_value = 0;
    op_param.flags = Flags::FORWARD_MAP;
    op_param.roi.c_start = 0;
    op_param.roi.h_start = 0;
    op_param.roi.w_start = 0;
    op_param.roi.c_len = 3;
    op_param.roi.h_len = out_height;
    op_param.roi.w_len = out_width;

    vsx::Image input_vacc;
    if (image_rgb_planar.GetContext().dev_type != vsx::Context::kVACC) {
      input_vacc = image_rgb_planar.Clone(vsx::Context::VACC(device_id_));
    } else {
      input_vacc = image_rgb_planar;
    }

    auto output_vacc = vsx::Image(image_rgb_planar.Format(), out_width,
                                  out_height, vsx::Context::VACC(device_id_));

    std::vector<vsx::Image> vacc_inputs = {input_vacc};
    std::vector<vsx::Image> vacc_outputs = {output_vacc};

    custom_op_->RunSync(vacc_inputs, vacc_outputs, &op_param, sizeof(op_param));

    return output_vacc;
  }
};
}  // namespace vsx
