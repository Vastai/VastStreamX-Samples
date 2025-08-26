
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

#include "common/model_cv2.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

namespace vsx {

class Mask2Former : public ModelCV2 {
 public:
  Mask2Former(const std::string &model_prefix, const std::string &vdsp_config,
              uint32_t batch_size = 1, uint32_t device_id = 0,
              float threshold = 0.2, const std::string &hw_config = "")
      : ModelCV2(model_prefix, vdsp_config, batch_size, device_id, hw_config),
        threshold_(threshold) {
    model_->GetInputShapeByIndex(0, input_shape_);
  }

 protected:
  std::vector<std::vector<vsx::Tensor>> ProcessImpl(
      const std::vector<vsx::Image> &images) {
    auto outputs = stream_->RunSync(images);
    std::vector<std::vector<vsx::Tensor>> results;
    results.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++) {
      const auto &output = outputs[i];
      std::vector<vsx::Tensor> tensor_host;
      for (const auto &out : output) {
        tensor_host.push_back(out.Clone());
      }
      results.push_back(
          PostProcess(tensor_host, images[i].Width(), images[i].Height()));
    }
    return results;
  }

  std::vector<vsx::Tensor> BilinearInterpolation(
      const vsx::Tensor &input_tensor, int new_width, int new_height) {
    int src_width = 256, src_height = 256;
    int tensor_count = input_tensor.GetSize() / (src_width * src_height);
    // transpose input_tensor
    vsx::Tensor trans_tensor({input_tensor.Shape()[1], input_tensor.Shape()[0]},
                             vsx::Context::CPU(), vsx::TypeFlag::kFloat32);
    const float *srcdata = input_tensor.Data<float>();
    float *dstdata = trans_tensor.MutableData<float>();
    int width = input_tensor.Shape()[1];
    int height = input_tensor.Shape()[0];
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        dstdata[w * height + h] = srcdata[h * width + w];
      }
    }
    // resize with bilinear interpolation
    std::vector<vsx::Tensor> results;
    for (int i = 0; i < tensor_count; i++) {
      vsx::Tensor tensor({new_height, new_width}, vsx::Context::CPU(),
                         vsx::TypeFlag::kFloat32);
      float *src_data =
          trans_tensor.MutableData<float>() + i * src_width * src_height;
      float *dst_data = tensor.MutableData<float>();
      cv::Mat src_mat(src_height, src_width, CV_32F, src_data);
      cv::Mat resize_mat(new_height, new_width, CV_32F, dst_data);
      cv::resize(src_mat, resize_mat, cv::Size(new_width, new_height), 0, 0,
                 cv::INTER_LINEAR);
      results.push_back(tensor);
    }
    return results;
  }

  std::vector<vsx::Tensor> PostProcess(
      const std::vector<vsx::Tensor> &fp16_tensors, int image_width,
      int image_height) {
    auto fp32_tensors = vsx::ConvertTensorFromFp16ToFp32(fp16_tensors);
    std::vector<vsx::Tensor> results;

    auto &cls_tensor = fp32_tensors[0];
    int width = cls_tensor.Shape()[1];
    int height = cls_tensor.Shape()[0];
    const float *cls_data = cls_tensor.Data<float>();

    std::vector<std::pair<int, float>> indices_scores_vec(height * (width - 1));
    int index = 0;
    for (int h = 0; h < height; h++) {
      float max_value = -100000.0;
      for (int w = 0; w < width; w++) {
        if (max_value < cls_data[w]) max_value = cls_data[w];
      }
      double sum_exp = 0.0;
      std::vector<double> exp_values(width);
      for (int w = 0; w < width; w++) {
        exp_values[w] = std::exp(cls_data[w] - max_value);
        sum_exp += exp_values[w];
      }
      for (auto &value : exp_values) {
        value /= sum_exp;
      }
      for (int w = 0; w < width - 1; w++) {
        indices_scores_vec[index] =
            std::make_pair(index, static_cast<float>(exp_values[w]));
        index++;
      }
      cls_data += width;
    }

    int num_classes = 80;
    int topk = 100;

    std::nth_element(
        indices_scores_vec.begin(), indices_scores_vec.begin() + topk - 1,
        indices_scores_vec.end(),
        [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
          return a.second > b.second;
        });
    std::sort(
        indices_scores_vec.begin(), indices_scores_vec.begin() + topk,
        [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
          return a.second > b.second;
        });

    int tk = 0;
    for (int i = 0; i < topk; i++) {
      if (indices_scores_vec[i].second < threshold_) break;
      tk++;
    }
    if (tk == 0) return results;
    topk = tk;

    std::vector<int> topk_indices(topk);
    std::vector<int> class_indices(topk);
    for (int i = 0; i < topk; ++i) {
      topk_indices[i] = indices_scores_vec[i].first / num_classes;
      class_indices[i] = indices_scores_vec[i].first % num_classes;
    }

    auto seg_result =
        BilinearInterpolation(fp32_tensors[1], image_width, image_height);

    std::vector<vsx::Tensor> mask_pred;
    for (auto &index : topk_indices) {
      mask_pred.push_back(seg_result[index]);
    }

    std::vector<float> pred_masks_sum;
    for (auto &pred : mask_pred) {
      float sum = 0.0;
      auto tensor = pred.Clone();
      float *data = tensor.MutableData<float>();
      for (size_t s = 0; s < tensor.GetSize(); s++) {
        if (data[s] > 0) {
          data[s] = 1.0f;
          sum += 1.0f;
        } else {
          data[s] = 0.0f;
        }
      }
      pred_masks_sum.push_back(sum + 1e-6);
    }

    std::vector<float> mask_scores_per_image;
    for (size_t s = 0; s < mask_pred.size(); s++) {
      float sum = 0.0;
      const float *data = mask_pred[s].Data<float>();
      for (size_t i = 0; i < mask_pred[s].GetSize(); i++) {
        if (data[i] > 0) sum += vsx::sigmoid(data[i]);
      }
      mask_scores_per_image.push_back(sum / pred_masks_sum[s]);
    }

    std::vector<std::pair<float, int>> scores_indices(topk);
    for (size_t s = 0; s < mask_scores_per_image.size(); s++) {
      scores_indices[s].first =
          indices_scores_vec[s].second * mask_scores_per_image[s];
      scores_indices[s].second = static_cast<int>(s);
    }

    std::sort(scores_indices.begin(), scores_indices.end(),
              [](const std::pair<float, int> &a,
                 const std::pair<float, int> &b) { return a.first > b.first; });

    int object_count = 0;
    for (auto &score_index : scores_indices) {
      if (score_index.first < threshold_) break;
      object_count++;
    }

    vsx::Tensor classes({object_count}, vsx::Context::CPU(),
                        vsx::TypeFlag::kFloat32);
    vsx::Tensor scores({object_count}, vsx::Context::CPU(),
                       vsx::TypeFlag::kFloat32);
    for (int i = 0; i < object_count; i++) {
      classes.MutableData<float>()[i] = class_indices[scores_indices[i].second];
      scores.MutableData<float>()[i] = scores_indices[i].first;
    }

    vsx::Tensor boxes({object_count, 4}, vsx::Context::CPU(),
                      vsx::TypeFlag::kFloat32);
    vsx::Tensor masks({object_count, image_height, image_width},
                      vsx::Context::CPU(), vsx::TypeFlag::kUint8);

    for (int i = 0; i < object_count; i++) {
      uint8_t *masks_data =
          masks.MutableData<uint8_t>() + i * image_height * image_width;
      int t = scores_indices[i].second;
      float *src_data = mask_pred[t].MutableData<float>();
      float *box_data = boxes.MutableData<float>() + i * 4;
      int w0 = 10000, h0 = 10000, w1 = -1, h1 = -1;
      for (int h = 0; h < image_height; h++) {
        for (int w = 0; w < image_width; w++) {
          if (src_data[h * image_width + w] <= 0) {
            masks_data[h * image_width + w] = 0;
          } else {
            masks_data[h * image_width + w] = 1;
            if (w0 > w) w0 = w;
            if (h0 > h) h0 = h;
            if (w1 < w) w1 = w;
            if (h1 < h) h1 = h;
          }
        }
      }
      box_data[0] = w0;
      box_data[1] = h0;
      box_data[2] = w1;
      box_data[3] = h1;
    }

    vsx::Tensor num({2}, vsx::Context::CPU(), vsx::TypeFlag::kUint32);
    num.MutableData<uint32_t>()[0] = object_count;

    results.push_back(std::move(classes));
    results.push_back(std::move(scores));
    results.push_back(std::move(boxes));
    results.push_back(std::move(masks));
    results.push_back(std::move(num));
    return results;
  }

 private:
  float threshold_ = 0.2;
  vsx::TShape input_shape_;
};

}  // namespace vsx