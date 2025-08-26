
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <algorithm>
#include <typeinfo>
#include <vector>

#include "common/model_cv.hpp"
#include "common/utils.hpp"

namespace vsx {

class Hih : public ModelCV {
 public:
  Hih(const std::string& model_prefix, const std::string& vdsp_config,
      uint32_t batch_size = 1, uint32_t device_id = 0,
      const std::string& hw_config = "")
      : ModelCV(model_prefix, vdsp_config, batch_size, device_id, hw_config) {}

 protected:
  std::vector<vsx::Tensor> ProcessImpl(const std::vector<vsx::Image>& images) {
    auto outputs = stream_->RunSync(images);
    std::vector<vsx::Tensor> results;
    results.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++) {
      const auto& output = outputs[i];
      std::vector<vsx::Tensor> tensor_host;
      for (const auto& out : output) {
        tensor_host.push_back(out.Clone());
      }
      results.push_back(PostProcess(tensor_host));
    }
    return results;
  }

  vsx::Tensor PostProcess(const std::vector<vsx::Tensor>& fp32_tensors) {
    vsx::Tensor preds({num_nb * hihOffset, 1}, vsx::Context::CPU(0),
                      vsx::kFloat32);
    float* result_ptr = preds.MutableData<float>();
    const int16_t* preds_heatmap = fp32_tensors[0].Data<int16_t>();
    const int16_t* preds_offset = fp32_tensors[1].Data<int16_t>();
    for (int i = 0; i < num_nb; ++i) {
      std::vector<float> preds_off(2, .0f);
      decode_woo_head(result_ptr, preds_heatmap, heatmap_size,
                      predHeatmapOffset);
      decode_woo_head(preds_off.data(), preds_offset, offsetmap_size,
                      predOffsetmapOffset);
      result_ptr[0] =
          (result_ptr[0] + preds_off[0] / static_cast<float>(offsetmap_size)) /
          static_cast<float>(heatmap_size);
      result_ptr[1] =
          (result_ptr[1] + preds_off[1] / static_cast<float>(offsetmap_size)) /
          static_cast<float>(heatmap_size);
      result_ptr += hihOffset;
      preds_heatmap += predHeatmapOffset;
      preds_offset += predOffsetmapOffset;
    }
    return preds;
  }

  const int16_t* my_max_element(const int16_t* vec, int size) {
    const int16_t* ptr = vec++;
    for (int i = 0; i < size; ++i) {
      if (*vec > *ptr) {
        ptr = vec;
      }
      vec += 1;
    }
    return ptr;
  }

  void decode_woo_head(float* preds, const int16_t* target_map, int map_size,
                       int length) {
    // int index = std::max_element(target_map, target_map + length) -
    // target_map;
    int index = my_max_element(target_map, length) - target_map;
    if (target_map[index] <= 0) {
      preds[0] = .0f;
      preds[1] = .0f;
      return;
    }
    preds[1] = index / map_size;
    preds[0] = index - preds[1] * map_size;
    return;
  }

 private:
  int num_nb = 98;
  int heatmap_size = 64;
  int offsetmap_size = 8;
  int predHeatmapOffset = heatmap_size * heatmap_size;
  int predOffsetmapOffset = offsetmap_size * offsetmap_size;
  int hihOffset = 2;
};

}  // namespace vsx