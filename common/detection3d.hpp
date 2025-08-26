
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include "custom_op_base.hpp"
namespace vsx {

constexpr int score_out_size = 1 * 500 * sizeof(uint16_t);
constexpr int label_out_size = 1 * 500 * sizeof(uint16_t);
constexpr int box_out_size = 1 * 500 * 7 * sizeof(uint16_t);

#define PP_MODEL_OUTPUT_SIZE 1024
#define MAX_PP_MODEL_NUM 10

static uint32_t GetInputCount(const char* op_name) { return 1; }
static uint32_t GetOutputCount(const char* op_name) { return 4; }
static vacmShape* GetInputShape(const char* op_name, uint32_t index) {
  static vacmShape shape;
  shape.ndims = 4;
  shape.shapes[0] = 1;
  shape.shapes[1] = 3;
  shape.shapes[2] = 720;
  shape.shapes[3] = 1280;
  return &shape;
}
static vacmShape* GetOutputShape(const char* op_name, uint32_t index) {
  std::vector<std::vector<int64_t>> inner_oshape = {{1, 3, 720, 1280},
                                                    {1, 3, 540, 960},
                                                    {1, 3, 360, 512},
                                                    {1, 3, 1080, 1920}};
  static vacmShape shape;
  shape.ndims = 4;
  shape.shapes[0] = inner_oshape[index][0];
  shape.shapes[1] = inner_oshape[index][1];
  shape.shapes[2] = inner_oshape[index][2];
  shape.shapes[3] = inner_oshape[index][3];
  return &shape;
}
static uint32_t GetInputSize(const char* op_name, uint32_t index) {
  return 100;
}
static uint32_t GetOutputSize(const char* op_name, uint32_t index) {
  return 1024;
}
static vsx::CustomOperatorCallback callback{GetInputCount,
                                            GetInputShape,
                                            GetOutputCount,
                                            GetOutputShape,
                                            GetInputSize,
                                            GetOutputSize,
                                            0,
                                            0};
typedef struct {
  int32_t num_points;
  int32_t num_features;
  float voxel_size[3];
  float coors_range[6];
  int32_t feature_width;
  int32_t feature_height;
  int32_t pts_per_voxel_max_num;
  int32_t shuffle_enabled;
  int32_t normalize_enabled;  // added for centerpoint normalize_enabled is 1,
                              // for pointpillar normalize_enabled is 0,

  int32_t tmp_buffer_size;
  uint64_t tmp_buffer_ptr64;

  // model part
  int32_t valid_model_num;
  int32_t max_voxel_num[MAX_PP_MODEL_NUM];
  uint64_t model_addr_list[MAX_PP_MODEL_NUM];

  // 3 output part
  uint64_t score_ptr64;
  uint64_t label_ptr64;
  uint64_t box_ptr64;
  uint64_t place_holder[8];
} pointpillar_model_ext_op_t;

struct PPModelConfig {
  std::string model_prefix;
  int32_t max_voxel_num;
  std::string hw_config;
};

const std::string pointpillar_op_name = "custom_op_pointpillar";

class Detection3D : public CustomOpBase {
 public:
  Detection3D(std::vector<PPModelConfig>& model_configs,
              const std::string& elf_file, std::vector<float>& voxel_sizes,
              std::vector<float>& coors_range, uint32_t device_id = 0,
              int max_points_num = 120000, int shuffle_enabled = 0,
              int normalize_enabled = 0, int max_feature_width = 864,
              int max_feature_height = 496, int actual_feature_width = 480,
              int actual_feature_height = 480, int num_feature = 4,
              int pts_per_voxel_max_num = 32)
      : CustomOpBase(pointpillar_op_name, elf_file, device_id) {
    uint32_t batch_size = 1;
    CHECK(voxel_sizes.size() == 3);
    CHECK(coors_range.size() == 6);

    custom_op_->SetCallback(callback);

    op_conf_.valid_model_num = model_configs.size();
    for (size_t i = 0; i < model_configs.size(); ++i) {
      auto model =
          std::make_shared<vsx::Model>(model_configs[i].model_prefix,
                                       batch_size, model_configs[i].hw_config);
      uint64_t model_addr = 0;
      model->GetModelAddress(model_addr);
      op_conf_.max_voxel_num[i] = model_configs[i].max_voxel_num;
      op_conf_.model_addr_list[i] = model_addr;
      models_.push_back(model);
    }

    op_conf_.num_features = num_feature;
    op_conf_.voxel_size[0] = voxel_sizes[0];
    op_conf_.voxel_size[1] = voxel_sizes[1];
    op_conf_.voxel_size[2] = voxel_sizes[2];
    op_conf_.coors_range[0] = coors_range[0];
    op_conf_.coors_range[1] = coors_range[1];
    op_conf_.coors_range[2] = coors_range[2];
    op_conf_.coors_range[3] = coors_range[3];
    op_conf_.coors_range[4] = coors_range[4];
    op_conf_.coors_range[5] = coors_range[5];
    op_conf_.feature_width = actual_feature_width;
    op_conf_.feature_height = actual_feature_height;
    op_conf_.pts_per_voxel_max_num = pts_per_voxel_max_num;
    op_conf_.shuffle_enabled = shuffle_enabled;
    op_conf_.normalize_enabled = normalize_enabled;

    int fix_buffer_size =
        (max_points_num * 16 + max_feature_height * max_feature_width * 2 +
         model_configs[0].max_voxel_num * 7 +
         pts_per_voxel_max_num * model_configs[0].max_voxel_num * 24 +
         21 * 1024 * 1024);

    tmp_buffer_tensor_ =
        vsx::Tensor(vsx::TShape({fix_buffer_size}),
                    vsx::Context::VACC(device_id), vsx::TypeFlag::kUint8);
    tmp_out_tensor_ =
        vsx::Tensor({PP_MODEL_OUTPUT_SIZE}, vsx::Context::VACC(device_id_),
                    vsx::TypeFlag::kUint8);
    score_tensor_ =
        vsx::Tensor({score_out_size / 2}, vsx::Context::VACC(device_id_),
                    vsx::TypeFlag::kFloat16);
    label_tensor_ =
        vsx::Tensor({label_out_size / 2}, vsx::Context::VACC(device_id_),
                    vsx::TypeFlag::kFloat16);
    box_tensor_ =
        vsx::Tensor({box_out_size / 2}, vsx::Context::VACC(device_id_),
                    vsx::TypeFlag::kFloat16);

    op_conf_.tmp_buffer_size = fix_buffer_size;
    op_conf_.tmp_buffer_ptr64 = tmp_buffer_tensor_.GetDataAddress();
    op_conf_.score_ptr64 = label_tensor_.GetDataAddress();
    op_conf_.label_ptr64 = score_tensor_.GetDataAddress();
    op_conf_.box_ptr64 = box_tensor_.GetDataAddress();
  }

  std::vector<vsx::Tensor> Process(const vsx::Tensor& tensor) {
    std::vector<vsx::Tensor> tensors{tensor};
    return Process(tensors)[0];
  }

  std::vector<std::vector<vsx::Tensor>> Process(
      const std::vector<vsx::Tensor>& tensors) {
    std::vector<vsx::Tensor> tensors_vacc;
    for (const auto& tensor : tensors) {
      if (tensor.GetContext().dev_type == vsx::Context::kCPU) {
        tensors_vacc.push_back(tensor.Clone(vsx::Context::VACC(device_id_)));
      } else {
        tensors_vacc.push_back(tensor);
      }
    }
    return ProcessImpl(tensors_vacc);
  }

  std::vector<vsx::Tensor> GetTestData(
      uint32_t bsize, uint32_t dtype, const Context& context,
      const std::vector<TShape>& input_shapes) {
    const auto& input_shape = input_shapes[0];
    std::vector<vsx::Tensor> images;
    images.reserve(bsize);

    auto tensor = vsx::Tensor(input_shape, context, dtype);
    for (uint32_t i = 0; i < bsize; i++) {
      images.push_back(tensor);
    }
    return images;
  }

 protected:
  virtual std::vector<std::vector<vsx::Tensor>> ProcessImpl(
      const std::vector<vsx::Tensor>& tensors) {
    std::vector<std::vector<vsx::Tensor>> results;
    for (const auto tensor : tensors) {
      op_conf_.num_points = tensor.GetSize() / 4;
      std::vector<vsx::Tensor> inputs{tensor};
      std::vector<vsx::Tensor> outputs{tmp_out_tensor_, score_tensor_,
                                       label_tensor_, box_tensor_};

      custom_op_->RunSync(inputs, outputs, &op_conf_,
                          sizeof(pointpillar_model_ext_op_t));

      std::vector<vsx::Tensor> outs_fp16{
          score_tensor_.Clone(), label_tensor_.Clone(), box_tensor_.Clone()};
      results.push_back(std::move(outs_fp16));
    }
    return results;
  }

 private:
  std::vector<std::shared_ptr<vsx::Model>> models_;
  pointpillar_model_ext_op_t op_conf_ = {0};
  vsx::Tensor tmp_buffer_tensor_;
  vsx::Tensor tmp_out_tensor_;
  vsx::Tensor score_tensor_;
  vsx::Tensor label_tensor_;
  vsx::Tensor box_tensor_;
};

}  // namespace vsx