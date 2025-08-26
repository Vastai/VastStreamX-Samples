
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "utils.hpp"
#include "vaststreamx/vaststreamx.h"
namespace vsx {

class DynamicModelBase {
 public:
  DynamicModelBase(const std::string& module_info,
                   const std::string& vdsp_config,
                   const std::vector<vsx::TShape>& max_input_shape,
                   uint32_t batch_size = 1, uint32_t device_id = 0)
      : device_id_(device_id), batch_size_(batch_size) {
    CHECK(vsx::SetDevice(device_id) == 0)
        << "SetDevice " << device_id << " failed";
    model_ = std::make_shared<vsx::Model>(module_info);
    CHECK(model_->SetDynamicModelInputShape(max_input_shape) == 0)
        << "Set Dynamic model input shape failed.";
    CHECK(model_->SetBatchSize(batch_size) == 0)
        << "SetBatchSize " << batch_size << " failed.";
    model_op_ = std::make_shared<vsx::ModelOperator>(model_);
    preproc_ops_ = vsx::Operator::LoadOpsFromJsonFile(vdsp_config);
    CHECK(preproc_ops_.size() > 0) << "LoadOpsFromJsonFile failed";
    graph_ = std::make_shared<vsx::Graph>();
    std::vector<std::shared_ptr<vsx::Operator>> all_ops(preproc_ops_.begin(),
                                                        preproc_ops_.end());
    all_ops.push_back(model_op_);
    CHECK(graph_->AddOperators(all_ops) == 0) << "graph AddOperators failed";
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
    for (auto op : preproc_ops_) {
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

 protected:
  std::vector<std::shared_ptr<vsx::Operator>> preproc_ops_;
  std::shared_ptr<vsx::ModelOperator> model_op_;
  std::shared_ptr<vsx::Model> model_;
  std::shared_ptr<vsx::Graph> graph_;
  std::shared_ptr<vsx::Stream> stream_;
  uint32_t device_id_;
  uint32_t batch_size_;
};

}  // namespace vsx