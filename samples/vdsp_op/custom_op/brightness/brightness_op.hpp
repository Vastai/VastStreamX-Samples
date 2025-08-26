
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

struct yuv_nv12_shape_t {
  int height, width;
  int h_pitch, w_pitch;
};
struct brightness_param_t {
  yuv_nv12_shape_t iimage_shape, oimage_shape;
  float scale;
};

class BrightnessOp : public CustomOpBase {
 public:
  BrightnessOp(const std::string& op_name, const std::string& elf_file,
               uint32_t device_id = 0, float scale = 2.2)
      : CustomOpBase(op_name, elf_file, device_id), scale_(scale) {
    custom_op_->SetCallback(callback);
  }
  void SetScale(float scale) { scale_ = scale; }

  vsx::Image Process(const vsx::Image& image) {
    std::vector<vsx::Image> images = {image};
    return Process(images)[0];
  }

  std::vector<vsx::Image> Process(const std::vector<vsx::Image>& images) {
    return ProcessImpl(images);
  }
  std::vector<vsx::Image> GetTestData(uint32_t bsize, uint32_t dtype,
                                      const Context& context,
                                      const std::vector<TShape>& input_shapes) {
    const auto& input_shape = input_shapes[0];

    std::vector<vsx::Image> images;
    images.reserve(bsize);
    int width, height;
    CHECK(input_shape.ndim() >= 2);
    height = input_shape[input_shape.ndim() - 2];
    width = input_shape[input_shape.ndim() - 1];
    auto image = vsx::Image(vsx::ImageFormat::YUV_NV12, width, height, context);
    for (uint32_t i = 0; i < bsize; i++) {
      images.push_back(image);
    }
    return images;
  }

 protected:
  virtual std::vector<vsx::Image> ProcessImpl(
      const std::vector<vsx::Image>& images) {
    std::vector<vsx::Image> results;
    for (const auto image : images) {
      int w_pitch = image.WidthPitch() ? image.WidthPitch() : image.Width();
      int h_pitch = image.HeightPitch() ? image.HeightPitch() : image.Height();
      brightness_param_t op_params = {0};
      op_params.iimage_shape.width = image.Width();
      op_params.iimage_shape.height = image.Height();
      op_params.iimage_shape.w_pitch = w_pitch;
      op_params.iimage_shape.h_pitch = h_pitch;

      op_params.oimage_shape.width = image.Width();
      op_params.oimage_shape.height = image.Height();
      op_params.oimage_shape.w_pitch = w_pitch;
      op_params.oimage_shape.h_pitch = h_pitch;

      op_params.scale = scale_;

      vsx::Image input_vacc;
      if (image.GetContext().dev_type != vsx::Context::kVACC) {
        input_vacc =
            vsx::Image(image.Format(), image.Width(), image.Height(),
                       vsx::Context::VACC(device_id_), image.WidthPitch(),
                       image.HeightPitch(), image.GetDType());
        input_vacc.CopyFrom(image);
      } else {
        input_vacc = image;
      }
      auto output_vacc =
          vsx::Image(image.Format(), image.Width(), image.Height(),
                     vsx::Context::VACC(device_id_), image.WidthPitch(),
                     image.HeightPitch(), image.GetDType());

      std::vector<vsx::Image> inputs{input_vacc};
      std::vector<vsx::Image> outputs{output_vacc};

      custom_op_->RunSync(inputs, outputs, &op_params,
                          sizeof(brightness_param_t));

      results.push_back(output_vacc);
    }

    return results;
  }

 protected:
  float scale_;
};
}  // namespace vsx
