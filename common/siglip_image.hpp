
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "common/utils.hpp"
#include "glog/logging.h"
#include "normalize_op.hpp"
#include "opencv2/opencv.hpp"
#include "space_to_depth_op.hpp"
#include "vaststreamx/core/resource.h"
#include "vaststreamx/core/stream.h"
#include "vaststreamx/datatypes/image.h"
namespace vsx {

class SiglipImage {
 public:
  SiglipImage(const std::string &model_prefix, const std::string &norm_op_elf,
              const std::string &space2depth_op_elf, uint32_t batch_size = 1,
              uint32_t device_id = 0, const std::string &hw_config = "",
              vsx::GraphOutputType output_type =
                  vsx::GraphOutputType::kGRAPH_OUTPUT_TYPE_NCHW_DEVICE)
      : device_id_(device_id), batch_size_(batch_size) {
    CHECK(vsx::SetDevice(device_id) == 0)
        << "SetDevice " << device_id << " failed";
    // normalize op
    std::vector<uint16_t> mean_v{14260, 14163, 13960};
    std::vector<uint16_t> std_v{13388, 13358, 13418};
    normal_type_t norm_type = normal_type_t::NORMAL_DIV255_MINUSMEAN_DIVSTD;
    normalize_op_ = std::make_shared<vsx::NormalizeOp>(
        "opf_normalize", norm_op_elf, device_id, mean_v, std_v, norm_type);

    // space_to_depth op
    int kh = 14, kw = 14, oh_align = 736, ow_align = 784;
    space2depth_op_ = std::make_shared<vsx::SpaceToDepthOp>(
        kh, kw, oh_align, ow_align, "opf_space_to_depth_out_matrix",
        space2depth_op_elf, device_id);

    model_ = std::make_shared<vsx::Model>(model_prefix, batch_size_, hw_config);
    model_op_ = std::make_shared<vsx::ModelOperator>(model_);
    graph_ = std::make_shared<vsx::Graph>(output_type);
    CHECK(graph_->AddOperators({model_op_}) == 0)
        << "graph AddOperators failed";
    stream_ =
        std::make_shared<vsx::Stream>(graph_, vsx::StreamBalanceMode::kBM_RUN);
    CHECK(stream_->RegisterModelOperatorOutput(model_op_) == 0)
        << "stream RegisterModelOperatorOutput failed";
    CHECK(stream_->Build() == 0) << "stream Build failed";

    model_->GetInputShapeByIndex(0, model_input_shape_);
  }

  uint32_t GetBatchSize(uint32_t &batch_size) {
    return model_->GetBatchSize(batch_size);
  }
  uint32_t GetMaxBatchSize(uint32_t &max_batch_size) {
    return model_->GetMaxBatchSize(max_batch_size);
  }
  uint32_t GetInputCount(uint32_t &count) {
    return model_->GetInputCount(count);
  }
  uint32_t GetOutputCount(uint32_t &count) {
    return model_->GetOutputCount(count);
  }
  uint32_t GetInputShapeByIndex(int32_t index, vsx::TShape &shape) {
    return model_->GetInputShapeByIndex(index, shape);
  }
  uint32_t GetOutputShapeByIndex(int32_t index, vsx::TShape &shape) {
    return model_->GetOutputShapeByIndex(index, shape);
  }

  vsx::Tensor Process(const cv::Mat &image) {
    std::vector<cv::Mat> images = {image};
    return Process(images)[0];
  }

  vsx::Tensor Process(const vsx::Image &image) {
    std::vector<vsx::Image> images = {image};
    return Process(images)[0];
  }

  std::vector<vsx::Tensor> Process(const std::vector<cv::Mat> &images) {
    std::vector<vsx::Image> va_images;
    va_images.reserve(images.size());
    for (const auto &image : images) {
      auto data_manager = std::make_shared<vsx::DataManager>(
          image.total() * image.channels(), vsx::Context::CPU(),
          reinterpret_cast<uint64_t>(image.data), [](void *data) {});
      va_images.emplace_back(vsx::BGR_INTERLEAVE, image.cols, image.rows, 0, 0,
                             data_manager);
    }
    return Process(va_images);
  }

  std::vector<vsx::Tensor> Process(const std::vector<vsx::Image> &images) {
    return ProcessImpl(images);
  }

  std::vector<vsx::Image> GetTestData(uint32_t bsize, uint32_t dtype,
                                      const Context &context,
                                      const std::vector<TShape> &input_shapes) {
    const auto &input_shape = input_shapes[0];
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
  void compute_size(int img_w, int img_h, int size_w, int size_h, int &new_w,
                    int &new_h) {
    float r = std::max(size_w / static_cast<float>(img_w),
                       size_h / static_cast<float>(img_h));
    new_w = std::round(r * img_w);
    new_h = std::round(r * img_h);
  }
  virtual std::vector<vsx::Tensor> ProcessImpl(
      const std::vector<vsx::Image> &images) {
    std::vector<vsx::Tensor> outputs;
    int mod_h = model_input_shape_[model_input_shape_.ndim() - 2];
    int mod_w = model_input_shape_[model_input_shape_.ndim() - 1];
    for (auto image : images) {
      int new_w, new_h;
      int img_w = image.Width();
      int img_h = image.Height();
      compute_size(img_w, img_h, mod_w, mod_h, new_w, new_h);

      vsx::Image cvtcolor_out;
      vsx::CvtColor(image, cvtcolor_out, vsx::ImageFormat::RGB_PLANAR);
      vsx::Image resize_out;
      vsx::Resize(cvtcolor_out, resize_out,
                  vsx::ImageResizeType::kRESIZE_TYPE_BICUBIC_PILLOW, new_w,
                  new_h);

      int left = (new_w - mod_w) / 2;
      int top = (new_h - mod_h) / 2;
      vsx::Image crop_out;
      vsx::Rect crop_rect;
      crop_rect.x = left;
      crop_rect.y = top;
      crop_rect.w = mod_w;
      crop_rect.h = mod_h;
      vsx::Crop(resize_out, crop_out, crop_rect);
      auto norm_out = normalize_op_->Process(crop_out);
      auto space_to_depth_out = space2depth_op_->Process(norm_out);
      auto model_outs = stream_->RunSync({{space_to_depth_out}});
      auto out = model_outs[0][0].Clone();
      auto out_fp32 = vsx::ConvertTensorFromFp16ToFp32(out);
      outputs.push_back(std::move(out_fp32));
    }
    return outputs;
  }

 protected:
  std::shared_ptr<vsx::NormalizeOp> normalize_op_;
  std::shared_ptr<vsx::SpaceToDepthOp> space2depth_op_;

  std::shared_ptr<vsx::ModelOperator> model_op_;
  std::shared_ptr<vsx::Model> model_;
  std::shared_ptr<vsx::Graph> graph_;
  std::shared_ptr<vsx::Stream> stream_;
  uint32_t device_id_;
  uint32_t batch_size_;
  vsx::TShape model_input_shape_;
};

}  // namespace vsx
