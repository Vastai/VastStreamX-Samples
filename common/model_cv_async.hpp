
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

class ModelCVAsync : public ModelBase {
 public:
  ModelCVAsync(const std::string& model_prefix, const std::string& vdsp_config,
               uint32_t batch_size = 1, uint32_t device_id = 0,
               const std::string& hw_config = "")
      : ModelBase(model_prefix, vdsp_config, batch_size, device_id, hw_config) {
  }

  uint32_t ProcessAsync(const cv::Mat& image) {
    std::vector<cv::Mat> images = {image};
    return ProcessAsync(images);
  }

  uint32_t ProcessAsync(const vsx::Image& image) {
    std::vector<vsx::Image> images = {image};
    return ProcessAsync(images);
  }

  uint32_t ProcessAsync(const std::vector<cv::Mat>& images) {
    std::vector<vsx::Image> va_images;
    va_images.reserve(images.size());
    for (const auto& image : images) {
      auto data_manager = std::make_shared<vsx::DataManager>(
          image.total() * image.channels(), vsx::Context::CPU(),
          reinterpret_cast<uint64_t>(image.data), [](void* data) {});
      va_images.emplace_back(vsx::BGR_INTERLEAVE, image.cols, image.rows, 0, 0,
                             data_manager);
    }
    return ProcessAsync(va_images);
  }

  uint32_t ProcessAsync(const std::vector<vsx::Image>& images) {
    return ProcessAsyncImpl(images);
  }

  virtual bool GetOutput(std::vector<Tensor>& output) {
    std::vector<std::vector<vsx::Tensor>> model_output;
    if (stream_->GetOperatorOutput(model_op_, model_output)) {
      for (auto mo : model_output) {
        output.push_back(mo[0]);
      }
      return true;
    }
    return false;
  }

  uint32_t WaitUntilDone() { return stream_->WaitUntilDone(); }

  uint32_t CloseInput() { return stream_->CloseInput(); }

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
  virtual uint32_t ProcessAsyncImpl(const std::vector<vsx::Image>& images) {
    return stream_->RunAsync(images);
  }
};

}  // namespace vsx