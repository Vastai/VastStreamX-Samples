
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
  ROTATE_DEGREE_90 = 0,
  ROTATE_DEGREE_180 = 1,
  ROTATE_DEGREE_270 = 2,
  ROTATE_DEGREE_NEG270 = 3,
  ROTATE_DEGREE_NEG90 = 4,
  ROTATE_DEGREE_END
} rotate_degree_e;

typedef enum {
  DATA_TYPE_U8 = 0,
  DATA_TYPE_F16 = 1,
  DATA_TYPE_F32 = 2,
  DATA_TYPE_END
} input_data_type_e;

typedef enum {
  IMAGE_DEFAULT = 0,
  IMAGE_NV12 = 1,
} image_type_e;

struct image_shape_layout_t {
  int height, width;
  int h_pitch, w_pitch;
};

typedef struct {
  image_shape_layout_t inimg_shape;
  image_shape_layout_t outimg_shape;
} rotate_param_chan_t;

typedef struct {
  int32_t channel;
  rotate_degree_e rotate_degree;
  input_data_type_e data_type;
  image_type_e image_type;
  rotate_param_chan_t rotate_param[MAX_CHAN];
} rotate_param_t;

class RotateOp : public CustomOpBase {
  // 仅支持 RGB PLANAR 格式
 public:
  RotateOp(
      const std::string& elf_file =
          "/opt/vastai/vaststream/lib/op/ext_op/video/simple_rotate_ext_op",
      uint32_t device_id = 0, const std::string& op_name = "simple_rotate_op")
      : CustomOpBase(op_name, elf_file, device_id) {}

  vsx::Image Process(const vsx::Image& image_rgb_planar, int angle) {
    if (angle == 0) {
      return image_rgb_planar;
    }
    rotate_degree_e rotate_angle = rotate_degree_e::ROTATE_DEGREE_END;
    if (angle == 90) {
      rotate_angle = rotate_degree_e::ROTATE_DEGREE_90;
    } else if (angle == 180) {
      rotate_angle = rotate_degree_e::ROTATE_DEGREE_180;
    } else if (angle == 270) {
      rotate_angle = rotate_degree_e::ROTATE_DEGREE_270;
    } else if (angle == -90) {
      rotate_angle = rotate_degree_e::ROTATE_DEGREE_NEG90;
    } else if (angle == -270) {
      rotate_angle = rotate_degree_e::ROTATE_DEGREE_NEG270;
    }
    if (rotate_angle == rotate_degree_e::ROTATE_DEGREE_END) {
      std::cerr << "Error: Unsupport rotate angle: " << angle << std::endl;
      return vsx::Image();
    }
    return Process(image_rgb_planar, rotate_angle);
  }

  vsx::Image Process(const vsx::Image& image_rgb_planar,
                     rotate_degree_e rotate_angle) {
    int in_width = image_rgb_planar.Width();
    int in_height = image_rgb_planar.Height();

    rotate_param_t op_params = {0};
    int out_width = in_width;
    int out_height = in_height;

    if (rotate_angle == rotate_degree_e::ROTATE_DEGREE_90 ||
        rotate_angle == rotate_degree_e::ROTATE_DEGREE_NEG90 ||
        rotate_angle == rotate_degree_e::ROTATE_DEGREE_270 ||
        rotate_angle == rotate_degree_e::ROTATE_DEGREE_NEG270) {
      out_width = in_height;
      out_height = in_width;
    } else if (rotate_angle != rotate_degree_e::ROTATE_DEGREE_180) {
      std::cerr << "ERROR: Unsupport rotate angle:" << rotate_angle
                << std::endl;
      return vsx::Image();
    }

    int channel = 3;
    for (int i = 0; i < channel; i++) {
      op_params.rotate_param[i].inimg_shape.width = in_width;
      op_params.rotate_param[i].inimg_shape.height = in_height;
      op_params.rotate_param[i].inimg_shape.w_pitch = in_width;
      op_params.rotate_param[i].inimg_shape.h_pitch = in_height;

      op_params.rotate_param[i].outimg_shape.width = out_width;
      op_params.rotate_param[i].outimg_shape.height = out_height;
      op_params.rotate_param[i].outimg_shape.w_pitch = out_width;
      op_params.rotate_param[i].outimg_shape.h_pitch = out_height;
    }
    op_params.channel = channel;
    op_params.rotate_degree = rotate_angle;
    op_params.data_type = input_data_type_e::DATA_TYPE_U8;
    op_params.image_type = image_type_e::IMAGE_DEFAULT;

    vsx::Image input_vacc;
    if (image_rgb_planar.GetContext().dev_type != vsx::Context::kVACC) {
      input_vacc = image_rgb_planar.Clone(vsx::Context::VACC(device_id_));
    } else {
      input_vacc = image_rgb_planar;
    }

    auto output_vacc = vsx::Image(image_rgb_planar.Format(), out_width,
                                  out_height, vsx::Context::VACC(device_id_));

    std::vector<vsx::Image> vacc_inputs = SplitImage(input_vacc, channel);
    std::vector<vsx::Image> vacc_outputs = SplitImage(output_vacc, channel);

    custom_op_->RunSync(vacc_inputs, vacc_outputs, &op_params,
                        sizeof(op_params));

    return output_vacc;
  }

 private:
  std::vector<vsx::Image> SplitImage(const vsx::Image& input, int channel) {
    int in_width = input.Width();
    int in_height = input.Height();
    const uint8_t* base_data_ptr = input.Data<uint8_t>();
    uint32_t size = in_height * in_width;
    auto ctx = input.GetContext();
    std::vector<vsx::Image> outputs;
    for (int i = 0; i < channel; i++) {
      auto data_ptr = base_data_ptr + i * size;
      auto data_buffer = std::make_shared<vsx::DataManager>(
          size, ctx, reinterpret_cast<uint64_t>(data_ptr), [](void* ptr) {});
      vsx::Image gray(in_width, in_height, vsx::ImageFormat::GRAY, data_buffer);
      outputs.push_back(gray);
    }
    return outputs;
  }
};
}  // namespace vsx
