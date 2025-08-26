
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

#include "common/model_cv_async.hpp"
#include "common/utils.hpp"
#include "glog/logging.h"

namespace vsx {

inline float GetScoreFromTensor(const vsx::Tensor& tensor) {
  return *tensor.Data<float>();
}

inline std::string GetStringFromTensor(const vsx::Tensor& tensor) {
  const char* c_str = tensor.Data<char>() + 8;
  return c_str;
}

class TextRecognizerAsync : public ModelCVAsync {
 public:
  TextRecognizerAsync(
      const std::string& model_prefix, const std::string& vdsp_config,
      uint32_t batch_size = 1, uint32_t device_id = 0,
      const std::string& label_file = "../data/labels/key_37.txt",
      const std::string& hw_config = "")
      : ModelCVAsync(model_prefix, vdsp_config, batch_size, device_id,
                     hw_config) {
    model_->GetInputShapeByIndex(0, input_shape_);
    // read label file
    std::ifstream ifs(label_file);
    CHECK(ifs.is_open()) << "Open label_file: " << label_file << " failed";

    int index = 0;
    while (!ifs.eof()) {
      std::string line;
      std::getline(ifs, line);
      if (!line.empty()) {
        key_map_[index++] = line;
      }
    }
    ifs.close();
  }
  bool GetOutput(std::vector<Tensor>& outputs) {
    std::vector<std::vector<Tensor>> model_outputs;
    if (stream_->GetOperatorOutput(model_op_, model_outputs)) {
      for (size_t t_index = 0; t_index < model_outputs.size(); ++t_index) {
        auto& outs = model_outputs[t_index];
        std::vector<vsx::Tensor> cpu_tensors;
        for (auto& tensor : outs) {
          cpu_tensors.push_back(tensor.Clone());
        }
        auto fp32_tensors = vsx::ConvertTensorFromFp16ToFp32(cpu_tensors);

        std::vector<vsx::Tensor> res_tensors;
        CHECK(fp32_tensors.size() == 1);
        for (size_t batch = 0; batch < fp32_tensors.size(); batch++) {
          // [L,C]
          std::string result = "";
          std::vector<float> conf;
          int32_t last_index;

          // Shape [L*C] C固定=key_map_.size()+1  L后期可能支持不定长
          // int32_t C = key_map_.size() + 1;
          // int32_t L = fp32_tensors[batch].GetSize() / C;
          auto shape = fp32_tensors[batch].Shape();
          // LOG(INFO) << "shape " << shape;
          int32_t stride = shape[1];
          int32_t C = key_map_.size() + 1;
          // int32_t L = 25;
          int32_t L = fp32_tensors[batch].GetSize() / C;
          // LOG(INFO) << "decode input shape: " << L << ", " << C;
          vsx::Tensor output({L + 9}, vsx::Context::CPU(), vsx::kUint8);
          memset(output.MutableData<char>(), 0, (L + 9));
          float* score = output.MutableData<float>();
          int* len = output.MutableData<int>() + 1;
          char* text = output.MutableData<char>() + 8;

          // vsx::SaveTensor("output_fp16.npy", cpu_tensors[batch]);
          // vsx::SaveTensor("output_fp32.npy", fp32_tensors[batch]);

          const float* confidence_data = fp32_tensors[batch].Data<float>();
          for (int32_t i = 0; i < L; ++i) {
            float max_score = 0.0;
            int32_t max_index = 0;
            for (int32_t j = 0; j < C; ++j) {
              auto cur_score = confidence_data[i * stride + j];
              if (cur_score > max_score) {
                max_score = cur_score;
                max_index = j;
              }
            }
            if (max_index > 0 && (!(i > 0 && max_index == last_index))) {
              result += key_map_[max_index - 1];
              conf.push_back(max_score);
            }

            last_index = max_index;
          }

          if (result.size() > 0 && conf.size() > 0) {
            float mean_conf = 0.0;
            std::for_each(conf.begin(), conf.end(),
                          [&](float n) { mean_conf += n; });
            *score = mean_conf / conf.size();
            CHECK(result.length() <= static_cast<size_t>(L));
            *len = result.length();
            // strcpy(text, result.c_str());
            snprintf(text, L, "%s", result.c_str());
          }
          res_tensors.emplace_back(output);
        }
        outputs.emplace_back(res_tensors[0]);
      }
      return true;
    }
    return false;
  }

 protected:
  uint32_t ProcessAsyncImpl(const std::vector<vsx::Image>& images) {
    StreamExtraRuntimeConfig conf;
    for (size_t i = 0; i < images.size(); ++i) {
      RgbLetterBoxExtConfig box;
      box.padding_bottom = 0;
      box.padding_left = 0;
      box.padding_top = 0;
      // calc resize width and resize height and padding right
      int img_h = images[i].Height();
      int img_w = images[i].Width();
      float radio = static_cast<float>(img_w) / img_h;
      int resize_w = 0;
      int model_in_width = input_shape_[3];
      int model_in_height = input_shape_[2];
      if (model_in_height * radio > model_in_width) {
        resize_w = model_in_width;
      } else {
        resize_w = model_in_height * radio;
      }
      int right = model_in_width - resize_w;
      if (right < 0) {
        right = 0;
      }
      box.padding_right = right;
      box.resize_width = resize_w;
      box.resize_height = model_in_height;
      conf.rgb_letterbox_ext_config.push_back(box);
    }
    return stream_->RunAsync(images, conf);
  }

 private:
  vsx::TShape input_shape_;
  std::unordered_map<int, std::string> key_map_;
};
}  // namespace vsx
