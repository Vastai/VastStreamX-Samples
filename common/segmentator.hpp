
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

#include "common/model_cv.hpp"
#include "common/utils.hpp"

namespace vsx {

class Segmentator : public ModelCV {
 protected:
  vsx::TShape output_shape_;

 public:
  Segmentator(const std::string& model_prefix, const std::string& vdsp_config,
              uint32_t batch_size = 1, uint32_t device_id = 0,
              const std::string& hw_config = "")
      : ModelCV(model_prefix, vdsp_config, batch_size, device_id, hw_config) {
    model_->GetOutputShapeByIndex(0, output_shape_);
    CHECK(output_shape_.ndim() == 4)
        << "Dimension of model output shape is not 4";
  }

  Tensor PostProcess(const Tensor& fp16_tensor) {
    int out_width = output_shape_[3];
    int out_height = output_shape_[2];
    Tensor out_tensor(TShape({1, out_height, out_width}), Context::CPU(),
                      vsx::kUint8);
    u_int8_t* dst = out_tensor.MutableData<u_int8_t>();
    int class_count = fp16_tensor.GetSize() / (out_width * out_height);
    size_t offset = out_height * out_width;
    const int16_t* channel_data = fp16_tensor.Data<int16_t>();
    for (size_t i = 0; i < offset; i++) {
      int16_t max = -32767;
      u_int8_t type = 255;
      for (int j = 0; j < class_count; j++) {
        if (channel_data[i + j * offset] > max) {
          max = channel_data[i + j * offset];
          type = j;
        }
      }
      dst[i] = type;
    }
    return out_tensor;
  }
};

}  // namespace vsx