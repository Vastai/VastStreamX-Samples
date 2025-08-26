
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include "common/model_cv.hpp"
#include "common/utils.hpp"
#include "db_post_process.hpp"

namespace vsx {

class DBnetDetector : public ModelCV {
 public:
  DBnetDetector(const std::string &model_prefix, const std::string &vdsp_config,
                const std::string &elf_path, uint32_t batch_size = 1,
                uint32_t device_id = 0, const std::string &hw_config = "")
      : ModelCV(model_prefix, vdsp_config, batch_size, device_id, hw_config) {
    auto shape = vsx::TShape();
    model_->GetInputShapeByIndex(0, shape);  // this model only has one input
    model_height_ = shape[2];
    model_width_ = shape[3];
    opinfo_.device_id = device_id;

    post_processer = std::make_shared<DbnetPostProcesser>(elf_path, device_id);
  }

  void SetThreshold(float threshold) {
    thresh_ = threshold;
    opinfo_.thresh = threshold;
  }
  void SetBoxThreshold(float threshold) { box_thresh_ = threshold; }
  void SetBoxUnclipRatio(float threshold) { unclip_ratio_ = threshold; }
  void SetPolygonScoreUsage(bool usage) { use_polygon_score_ = usage; }
  void SetDilationUsage(bool usage) { use_dilation_ = usage; }
  void SetVisualizeUsage(bool usage) { visualize_ = usage; }

  void SetMaxContourNumber(uint32_t max_contour_num) {
    opinfo_.max_contour_num = max_contour_num;
  }
  static void SetTestDataPath(const std::string &path) {
    test_data_path_ = path;
  }

  std::vector<vsx::Image> GetTestData(
      uint32_t bsize, uint32_t dtype, const Context &context,
      const std::vector<vsx::TShape> &input_shapes) {
    const auto &input_shape = input_shapes[0];
    CHECK(input_shape.ndim() >= 2);
    vsx::Image image;
    if (test_data_path_.empty()) {
      int width, height;
      height = input_shape[input_shape.ndim() - 2];
      width = input_shape[input_shape.ndim() - 1];
      image = vsx::Image(vsx::BGR_INTERLEAVE, width, height, context);
    } else {
      LOG(INFO) << "USE REAL DATA: " << test_data_path_;
      CHECK(vsx::MakeVsxImage(test_data_path_, image, vsx::BGR_INTERLEAVE) == 0)
          << "Failed to read: " << test_data_path_;
      if (context.dev_type != vsx::Context::kCPU) {
        auto image_dev = vsx::Image(vsx::BGR_INTERLEAVE, image.Width(),
                                    image.Height(), context);
        image_dev.CopyFrom(image);
      }
    }

    std::vector<vsx::Image> images;
    images.reserve(bsize);
    for (uint32_t i = 0; i < bsize; ++i) {
      images.push_back(image);
    }
    return images;
  }

 private:
  vsx::Tensor save_boxes_2_tensor(
      const std::vector<float> &scores,
      const std::vector<std::vector<std::vector<int>>> &boxes) {
    int64_t box_num = boxes.size();
    vsx::Tensor tsr(vsx::TShape({box_num, 9}), vsx::Context::CPU(),
                    vsx::TypeFlag::kFloat32);

    // NOTE: 要确保内存连续
    float *dataArray = tsr.MutableData<float>();
    size_t currentIndex = 0;
    for (size_t i = 0; i < boxes.size(); i++) {
      const auto &vec2D = boxes[i];
      dataArray[currentIndex++] = scores[i];
      for (const auto &vec1D : vec2D) {
        for (const int &value : vec1D) {
          dataArray[currentIndex++] = value;
        }
      }
    }
    return tsr;
  }

  std::vector<vsx::Tensor> ProcessImpl(const std::vector<vsx::Image> &images) {
    auto outputs = stream_->RunSync(images);
    // NOTE: this vector receive fp16 tensor
    auto batch_size = outputs.size();
    std::vector<vsx::Tensor> results;
    results.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      auto &output = outputs[i][0];

      if (output.GetContext().dev_type != vsx::Context::kVACC) {
        output = output.Clone(vsx::Context::VACC(device_id_));
      }

      // vsx::TShape output_shape;
      // model_->GetOutputShapeByIndex(0, output_shape);
      // std::cout << "model output shape: " << output_shape << std::endl;

      // output = output.Reshape(output_shape);

      std::vector<int> srcimg_hw{images[i].Height(), images[i].Width()};
      std::vector<float> ratio_hw{this->model_height_ * 1.0f / srcimg_hw[0],
                                  this->model_width_ * 1.0f / srcimg_hw[1]};
      DBresult db_res = post_processer->Process(
          output, srcimg_hw, ratio_hw, this->thresh_, this->use_dilation_,
          this->box_thresh_, this->unclip_ratio_, this->use_polygon_score_,
          this->opinfo_, this->visualize_);
      if (!db_res.boxes.empty())
        results.push_back(save_boxes_2_tensor(db_res.scores, db_res.boxes));
      else
        results.push_back(
            vsx::Tensor({0}, vsx::Context::CPU(), vsx::TypeFlag::kFloat32));
    }
    return results;
  }

 private:
  // for post process
  double thresh_ = 0.3;
  double box_thresh_ = 0.5;
  double unclip_ratio_ = 2.0;
  bool use_polygon_score_ = true;
  bool use_dilation_ = false;
  bool visualize_ = false;
  // model info
  int model_height_, model_width_;

  std::shared_ptr<DbnetPostProcesser> post_processer;
  // test data path
  static std::string test_data_path_;
  static std::string elf_path_;

  FindContourOpInfo opinfo_{
      "/opt/vastai/vaststreamx/data/elf/find_contours_ext_op", 0.3, 0, 1024};
};

std::string DBnetDetector::test_data_path_ = "";

}  // namespace vsx