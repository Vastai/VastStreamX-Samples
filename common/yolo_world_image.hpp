
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

#include "common/utils.hpp"

namespace vsx {

class YoloWorldImage {
 public:
  YoloWorldImage(const std::string& model_prefix,
                 const std::string& vdsp_config, uint32_t batch_size = 1,
                 uint32_t device_id = 0, const std::string& hw_config = "")
      : device_id_(device_id), batch_size_(batch_size) {
    CHECK(vsx::SetDevice(device_id) == 0)
        << "SetDevice " << device_id << " failed";
    model_ = std::make_shared<vsx::Model>(model_prefix, batch_size_, hw_config);
    model_->GetInputShapeByIndex(0, input_shape_);

    ops_ = vsx::Operator::LoadOpsFromJsonFile(vdsp_config);
    fusion_op_ = static_cast<vsx::BuildInOperator*>(ops_[0].get());

    oimage_width_ = input_shape_[3];
    oimage_height_ = input_shape_[2];

    fusion_op_->SetAttribute<vsx::AttrKey::kOimageWidth>(oimage_width_);
    fusion_op_->SetAttribute<vsx::AttrKey::kOimageHeight>(oimage_height_);

    graph_ = std::make_shared<vsx::Graph>(
        vsx::GraphOutputType::kGRAPH_OUTPUT_TYPE_NCHW_DEVICE);

    model_op_ = std::make_shared<vsx::ModelOperator>(model_);

    CHECK(graph_->AddOperators({model_op_}) == 0)
        << "graph AddOperators failed";
    stream_ =
        std::make_shared<vsx::Stream>(graph_, vsx::StreamBalanceMode::kBM_RUN);
    CHECK(stream_->RegisterModelOperatorOutput(model_op_) == 0)
        << "stream RegisterModelOperatorOutput failed";
    CHECK(stream_->Build() == 0) << "stream Build failed";
  }
  uint32_t GetBatchSize(uint32_t& batch_size) {
    return model_->GetBatchSize(batch_size);
  }
  uint32_t GetMaxBatchSize(uint32_t& max_batch_size) {
    return model_->GetMaxBatchSize(max_batch_size);
  }
  uint32_t GetInputCount(uint32_t& count) {
    return model_->GetInputCount(count);
  }
  uint32_t GetOutputCount(uint32_t& count) {
    return model_->GetOutputCount(count);
  }
  uint32_t GetInputShapeByIndex(int32_t index, vsx::TShape& shape) {
    return model_->GetInputShapeByIndex(index, shape);
  }
  uint32_t GetOutputShapeByIndex(int32_t index, vsx::TShape& shape) {
    return model_->GetOutputShapeByIndex(index, shape);
  }
  vsx::ImageFormat GetFusionOpIimageFormat() {
    for (auto op : ops_) {
      if (op->GetOpType() >= 100) {
        auto attri_keys = op->GetAttrKeys();
        if (vsx::HasAttribute(attri_keys, "kIimageFormat")) {
          auto fusion_op = static_cast<vsx::BuildInOperator*>(op.get());
          vsx::BuildInOperatorAttrImageType format;
          fusion_op->GetAttribute<vsx::AttrKey::kIimageFormat>(format);
          return vsx::ConvertToVsxFormat(format);
        } else {
          return vsx::ImageFormat::YUV_NV12;
        }
      }
    }
    CHECK(false) << "Can't find fusion op that op_type >= 100";
    return vsx::ImageFormat::YUV_NV12;
  }

  std::pair<std::vector<vsx::Image>, std::vector<vsx::Tensor>> GetTestData(
      uint32_t bsize, uint32_t dtype, const Context& context,
      const std::vector<TShape>& input_shapes) {
    const auto& input_shape = input_shapes[0];
    int width, height;
    CHECK(input_shape.ndim() >= 2);
    height = input_shape[input_shape.ndim() - 2];
    width = input_shape[input_shape.ndim() - 1];

    std::vector<vsx::Image> images;
    std::vector<vsx::Tensor> tensors;

    auto image =
        vsx::Image(vsx::ImageFormat::BGR_INTERLEAVE, width, height, context);
    auto tokens_cpu =
        vsx::Tensor({1203, 512}, vsx::Context::CPU(), vsx::TypeFlag::kFloat16);
    float* data = tokens_cpu.MutableData<float>();
    memset(data, 0, tokens_cpu.GetDataBytes());
    tokens_cpu = vsx::bert_get_activation_fp16_A(tokens_cpu);
    auto tokens = tokens_cpu.Clone(context);

    for (uint32_t i = 0; i < bsize; i++) {
      images.push_back(image);
      tensors.push_back(tokens);
    }
    auto test_datas = std::make_pair(images, tensors);
    return test_datas;
  }
  std::vector<vsx::Tensor> Process(
      const std::pair<cv::Mat, vsx::Tensor>& input) {
    std::vector<cv::Mat> mats{input.first};
    std::vector<vsx::Tensor> tensors{input.second};
    return Process(std::make_pair(mats, tensors))[0];
  }

  std::vector<vsx::Tensor> Process(
      const std::pair<vsx::Image, vsx::Tensor>& input) {
    std::vector<vsx::Image> images{input.first};
    std::vector<vsx::Tensor> tensors{input.second};
    return Process(std::make_pair(images, tensors))[0];
  }

  std::vector<std::vector<vsx::Tensor>> Process(
      const std::pair<std::vector<cv::Mat>, std::vector<vsx::Tensor>>& input) {
    std::vector<vsx::Image> va_images;
    va_images.reserve(input.first.size());
    for (const auto& image : input.first) {
      auto data_manager = std::make_shared<vsx::DataManager>(
          image.total() * image.channels(), vsx::Context::VACC(device_id_));
      vsx::Memcpy(reinterpret_cast<void*>(image.data),
                  data_manager->GetDataPtr(), image.total() * image.channels(),
                  vsx::COPY_TO_DEVICE);
      va_images.emplace_back(image.cols, image.rows, vsx::BGR_INTERLEAVE,
                             data_manager);
    }
    return Process(std::make_pair(va_images, input.second));
  }

  std::vector<std::vector<vsx::Tensor>> Process(
      const std::pair<std::vector<vsx::Image>, std::vector<vsx::Tensor>>&
          input) {
    std::vector<vsx::Image> dev_images;
    for (const auto& img : input.first) {
      if (img.GetContext().dev_type == vsx::Context::DeviceType::kCPU) {
        dev_images.emplace_back(img.Clone(vsx::Context::VACC(device_id_)));
      } else {
        dev_images.push_back(img);
      }
    }

    std::vector<vsx::Tensor> dev_tensors;
    for (auto& tensor : input.second) {
      if (tensor.GetContext().dev_type == vsx::Context::DeviceType::kCPU) {
        dev_tensors.emplace_back(tensor.Clone(vsx::Context::VACC(device_id_)));
      } else {
        dev_tensors.push_back(tensor);
      }
    }
    return ProcessImpl(std::make_pair(dev_images, dev_tensors));
  }

 protected:
  std::vector<std::vector<vsx::Tensor>> ProcessImpl(
      const std::pair<std::vector<vsx::Image>, std::vector<vsx::Tensor>>
          inputs) {
    const auto& images = inputs.first;
    const auto& txt_tensors = inputs.second;

    std::vector<vsx::Tensor> img_tensors;
    for (size_t i = 0; i < images.size(); i++) {
      int width = images[i].Width();
      int height = images[i].Height();

      fusion_op_->SetAttribute<vsx::AttrKey::kIimageWidth>(width);
      fusion_op_->SetAttribute<vsx::AttrKey::kIimageWidthPitch>(width);
      fusion_op_->SetAttribute<vsx::AttrKey::kIimageHeight>(height);
      fusion_op_->SetAttribute<vsx::AttrKey::kIimageHeightPitch>(height);

      std::vector<vsx::Tensor> input{images[i]};
      auto vdsp_out =
          vsx::Tensor({1, 160, 160, 1, 256}, vsx::Context::VACC(device_id_),
                      vsx::TypeFlag::kFloat16);
      std::vector<vsx::Tensor> output{vdsp_out};

      fusion_op_->Execute(input, output);
      img_tensors.push_back(vdsp_out);
    }

    std::vector<std::vector<vsx::Tensor>> batch_inputs;
    for (size_t i = 0; i < img_tensors.size(); i++) {
      std::vector<vsx::Tensor> one_input;
      one_input.push_back(img_tensors[i]);
      one_input.push_back(txt_tensors[i]);
      batch_inputs.push_back(std::move(one_input));
    }
    auto outputs = stream_->RunSync(batch_inputs);

    std::vector<std::vector<vsx::Tensor>> results;
    for (auto& output : outputs) {
      std::vector<vsx::Tensor> host;
      for (size_t i = 0; i < output.size(); i++) {
        auto fp32_tensor = ConvertTensorFromFp16ToFp32(output[i].Clone());
        if (i < 3) {
          host.push_back(fp32_tensor);
        } else {
          size_t size = fp32_tensor.GetSize();
          int height = size / 4;
          int shape = static_cast<int>(sqrt(height));
          auto trans = vsx::Tensor({1, 4, shape, shape}, vsx::Context::CPU(),
                                   vsx::TypeFlag::kFloat32);
          const float* src = fp32_tensor.Data<float>();
          float* dst = trans.MutableData<float>();
          for (int h = 0; h < height; h++) {
            for (int w = 0; w < 4; w++) {
              dst[w * height + h] = src[h * 4 + w];
            }
          }
          host.push_back(trans);
        }
      }
      results.push_back(std::move(host));
    }
    return results;
  }

 public:
  vsx::TShape input_shape_;

 protected:
  std::vector<std::shared_ptr<vsx::Operator>> ops_;
  vsx::BuildInOperator* fusion_op_ = nullptr;
  int oimage_width_, oimage_height_;
  std::shared_ptr<vsx::ModelOperator> model_op_;
  std::shared_ptr<vsx::Model> model_;
  std::shared_ptr<vsx::Graph> graph_;
  std::shared_ptr<vsx::Stream> stream_;
  uint32_t device_id_;
  uint32_t batch_size_;
};

}  // namespace vsx
