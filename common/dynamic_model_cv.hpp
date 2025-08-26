
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <vector>

#include "dynamic_model_base.hpp"
#include "opencv2/opencv.hpp"
#include "utils.hpp"

namespace vsx {

class DynamicModelCV : public DynamicModelBase {
 public:
  DynamicModelCV(const std::string& module_info, const std::string& vdsp_config,
                 const std::vector<vsx::TShape>& max_input_shape,
                 uint32_t batch_size = 1, uint32_t device_id = 0)
      : DynamicModelBase(module_info, vdsp_config, max_input_shape, batch_size,
                         device_id) {}

  vsx::Tensor Process(const cv::Mat& image) {
    std::vector<cv::Mat> images = {image};
    return Process(images)[0];
  }

  vsx::Tensor Process(const vsx::Image& image) {
    std::vector<vsx::Image> images = {image};
    return Process(images)[0];
  }

  std::vector<vsx::Tensor> Process(const std::vector<cv::Mat>& images) {
    std::vector<vsx::Image> va_images;
    va_images.reserve(images.size());
    for (const auto& image : images) {
      auto data_manager = std::make_shared<vsx::DataManager>(
          image.total() * image.channels(), vsx::Context::CPU(),
          reinterpret_cast<uint64_t>(image.data), [](void* data) {});
      va_images.emplace_back(vsx::BGR_INTERLEAVE, image.cols, image.rows, 0, 0,
                             data_manager);
    }
    return Process(va_images);
  }

  std::vector<vsx::Tensor> Process(const std::vector<vsx::Image>& images) {
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
    auto image =
        vsx::Image(vsx::ImageFormat::BGR_INTERLEAVE, width, height, context);
    for (uint32_t i = 0; i < bsize; i++) {
      images.push_back(image);
    }
    return images;
  }
  uint32_t SetInputShape(const std::vector<TShape>& model_input_shape) {
    input_shape_ = model_input_shape;
    return 0;
  }

 protected:
  virtual std::vector<vsx::Tensor> ProcessImpl(
      const std::vector<vsx::Image>& images) {
    auto outputs = stream_->RunSync(images);
    std::vector<vsx::Tensor> results;
    results.reserve(outputs.size());
    for (const auto& output : outputs) {
      results.push_back(output[0].Clone());
    }
    return results;
  }

 protected:
  std::vector<vsx::TShape> input_shape_;
};

}  // namespace vsx