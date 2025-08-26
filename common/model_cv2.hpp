
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <vector>

#include "model_base.hpp"
#include "opencv2/opencv.hpp"
#include "utils.hpp"

namespace vsx {

class ModelCV2 : public ModelBase {
 public:
  ModelCV2(const std::string& model_prefix, const std::string& vdsp_config,
           uint32_t batch_size = 1, uint32_t device_id = 0,
           const std::string& hw_config = "",
           vsx::GraphOutputType output_type =
               vsx::GraphOutputType::kGRAPH_OUTPUT_TYPE_NCHW_DEVICE)
      : ModelBase(model_prefix, vdsp_config, batch_size, device_id, hw_config,
                  output_type) {}

  std::vector<vsx::Tensor> Process(const cv::Mat& image) {
    std::vector<cv::Mat> images = {image};
    return Process(images)[0];
  }

  std::vector<vsx::Tensor> Process(const vsx::Image& image) {
    std::vector<vsx::Image> images = {image};
    return Process(images)[0];
  }

  std::vector<std::vector<vsx::Tensor>> Process(
      const std::vector<cv::Mat>& images) {
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

  std::vector<std::vector<vsx::Tensor>> Process(
      const std::vector<vsx::Image>& images) {
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

 protected:
  virtual std::vector<std::vector<vsx::Tensor>> ProcessImpl(
      const std::vector<vsx::Image>& images) {
    auto outputs = stream_->RunSync(images);
    std::vector<std::vector<vsx::Tensor>> results;
    results.reserve(outputs.size());
    for (const auto& output : outputs) {
      std::vector<vsx::Tensor> res;
      for (const auto& out : output) {
        res.push_back(std::move(out.Clone()));
      }
      results.push_back(std::move(res));
    }
    return results;
  }
};

}  // namespace vsx