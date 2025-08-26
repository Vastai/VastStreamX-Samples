
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "ByteTracker/include/BYTETracker.h"
#include "common/detector.hpp"
#include "vaststreamx/core/resource.h"
#include "vaststreamx/core/stream.h"
#include "vaststreamx/datatypes/image.h"

namespace vsx {

class ByteTracker : public Detector {
 public:
  ByteTracker(const std::string& model_prefix, const std::string& vdsp_config,
              uint32_t batch_size = 1, uint32_t device_id = 0,
              float det_thresh = 0.2, int track_buffer = 30,
              float track_thresh = 0.6, int fps = 30,
              const std::string& hw_config = "")
      : Detector(model_prefix, vdsp_config, batch_size, device_id, det_thresh,
                 hw_config) {
    tracker_ = std::make_shared<BYTETracker>(fps, track_buffer, track_thresh);
    SetBoxAdaptEdge(false);
  }

 protected:
  std::vector<vsx::Tensor> ProcessImpl(const std::vector<vsx::Image>& images) {
    auto results = Detector::ProcessImpl(images);
    std::vector<vsx::Tensor> outputs;
    for (size_t j = 0; j < results.size(); j++) {
      const auto& result = results[j];
      std::vector<Object> objects;

      auto res_shape = result.Shape();
      const float* res_data = result.Data<float>();
      for (int i = 0; i < res_shape[0]; i++) {
        if (res_data[0] < 0) break;
        Object obj;
        obj.label = static_cast<int>(res_data[0]);
        obj.prob = res_data[1];
        obj.rect.x = res_data[2];
        obj.rect.y = res_data[3];
        obj.rect.width = res_data[4];
        obj.rect.height = res_data[5];
        objects.push_back(obj);
        res_data += kDetectionOffset;
      }

      vector<STrack> output_stracks = tracker_->update(objects);

      vsx::Tensor temp(
          {static_cast<int64_t>(output_stracks.size()), kDetectionOffset},
          vsx::Context::CPU(), vsx::kFloat32);
      float* tmp_data = temp.MutableData<float>();
      for (const auto& eachtrack : output_stracks) {
        tmp_data[0] = 0;  // all objects are person
        tmp_data[1] = eachtrack.score;
        tmp_data[2] = eachtrack.tlwh[0];
        tmp_data[3] = eachtrack.tlwh[1];
        tmp_data[4] = eachtrack.tlwh[2];
        tmp_data[5] = eachtrack.tlwh[3];
        tmp_data[6] = static_cast<float>(eachtrack.track_id);
        tmp_data += kDetectionOffset;
      }
      outputs.push_back(std::move(temp));
    }
    return outputs;
  }

 private:
  std::shared_ptr<BYTETracker> tracker_;
};

}  // namespace vsx
