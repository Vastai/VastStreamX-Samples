
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <omp.h>

#include "vaststreamx/vaststreamx.h"

inline vsx::Tensor GetL2Norm(const vsx::Tensor& input) {
  auto result = input.Clone();
  float* in_data = input.MutableData<float>();
  float* res_data = result.MutableData<float>();

  int height = input.Shape()[0], width = input.Shape()[1];
  for (int h = 0; h < height; h++) {
    float sum = 0;
    for (int w = 0; w < width; w++) {
      int p = h * width + w;
      float x0 = std::abs(in_data[p]);
      sum += x0 * x0;
    }
    float d_sqrt = std::sqrt(sum);
    for (int w = 0; w < width; w++) {
      int p = h * width + w;
      res_data[p] = in_data[p] / d_sqrt;
    }
  }
  return result;
}

inline vsx::Tensor GetRes(const vsx::Tensor& data_a, const vsx::Tensor& data_b,
                          float multiply_const, float add_const) {
  int a_width = data_a.GetSize() / 512;
  int a_height = 512;
  int b_width = 512;
  int b_height = data_b.Shape()[0];

  auto result = vsx::Tensor(
      vsx::TShape({1, b_height, data_a.Shape()[2], data_a.Shape()[3]}),
      vsx::Context::CPU(), vsx::TypeFlag::kFloat32);

  float* a_data = data_a.MutableData<float>();
  float* b_data = data_b.MutableData<float>();
  float* res_data = result.MutableData<float>();

#pragma omp parallel for
  for (int aw = 0; aw < a_width; aw++) {
    for (int bh = 0; bh < b_height; bh++) {
      float sum = 0;
      for (int bw = 0; bw < b_width; bw++) {
        sum += a_data[bw * a_width + aw] * b_data[bh * b_width + bw];
      }
      sum *= multiply_const;
      sum += add_const;
      res_data[bh * a_width + aw] = sum;
    }
  }
  return result;
}

inline std::vector<vsx::Tensor> GetScoresBatch(
    const std::vector<vsx::Tensor>& img_features,
    const vsx::Tensor& text_feature) {
  const auto& data_3489 = img_features[0];
  const auto& data_3566 = img_features[1];
  const auto& data_3643 = img_features[2];

  auto start = std::chrono::high_resolution_clock::now();
  auto res_l2_b = GetL2Norm(text_feature);
  auto end = std::chrono::high_resolution_clock::now();
  auto cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();
  std::cout << "GetL2Norm cost: " << cost << " ms.\n";

  start = std::chrono::high_resolution_clock::now();
  auto res_3489 =
      GetRes(data_3489, res_l2_b, 1.7583335638046265, -12.202274322509766);
  end = std::chrono::high_resolution_clock::now();
  cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
             .count();
  std::cout << "GetRes1 cost: " << cost << " ms.\n";
  start = std::chrono::high_resolution_clock::now();
  auto res_3566 =
      GetRes(data_3566, res_l2_b, 1.7852298021316528, -10.711109161376953);
  end = std::chrono::high_resolution_clock::now();
  cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
             .count();
  std::cout << "GetRes2 cost: " << cost << " ms.\n";
  start = std::chrono::high_resolution_clock::now();
  auto res_3643 =
      GetRes(data_3643, res_l2_b, 2.0692732334136963, -9.184325218200684);
  end = std::chrono::high_resolution_clock::now();
  cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
             .count();
  std::cout << "GetRes3 cost: " << cost << " ms.\n";
  std::vector<vsx::Tensor> outputs{res_3489, res_3566, res_3643};
  return outputs;
}

inline vsx::Tensor SingleLevelGridPriors(const vsx::TShape& featmat_size,
                                         int stride, float offset = 0.5) {
  int feat_h = featmat_size[featmat_size.ndim() - 2];
  int feat_w = featmat_size[featmat_size.ndim() - 1];
  int stride_w = stride, stride_h = stride;

  std::vector<float> shift_x;
  shift_x.reserve(feat_w);
  for (int i = 0; i < feat_w; i++) {
    shift_x.push_back((i + offset) * stride_w);
  }
  std::vector<float> shift_y;
  shift_y.reserve(feat_h);
  for (int i = 0; i < feat_h; i++) {
    shift_y.push_back((i + offset) * stride_h);
  }

  vsx::Tensor result({feat_w * feat_h, 2}, vsx::Context::CPU(),
                     vsx::TypeFlag::kFloat32);
  float* data = result.MutableData<float>();
  for (int h = 0; h < feat_h; h++) {
    for (int w = 0; w < feat_w; w++) {
      data[0] = shift_x[w];
      data[1] = shift_y[h];
      data += 2;
    }
  }

  return result;
}

inline std::vector<vsx::Tensor> GetGridPriors(
    const std::vector<vsx::TShape>& featmap_sizes,
    const std::vector<int>& strides) {
  std::vector<vsx::Tensor> result;
  result.reserve(featmap_sizes.size());
  for (size_t i = 0; i < featmap_sizes.size(); i++) {
    result.push_back(SingleLevelGridPriors(featmap_sizes[i], strides[i]));
  }
  return std::move(result);
}

inline vsx::Tensor GetBboxCoderDecode(const vsx::Tensor& points,
                                      const vsx::Tensor& pred_bboxes,
                                      const vsx::Tensor& stride) {
  auto new_pred_bboxes = pred_bboxes.Clone();
  auto shape = new_pred_bboxes.Shape();
  int dim1 = shape[shape.ndim() - 1];  // 4
  int dim2 = shape[shape.ndim() - 2];
  float* data = new_pred_bboxes.MutableData<float>();
  const int* stride_data = stride.Data<int>();
  for (int d2 = 0; d2 < dim2; d2++) {
    for (int d1 = 0; d1 < dim1; d1++) {
      data[0] = data[0] * stride_data[0];
      data++;
    }
    stride_data++;
  }

  auto result = vsx::Tensor(pred_bboxes.Shape(), vsx::Context::CPU(),
                            vsx::TypeFlag::kFloat32);
  const float* poind_data = points.Data<float>();
  const float* pred_data = new_pred_bboxes.Data<float>();
  float* res_data = result.MutableData<float>();
  for (int d2 = 0; d2 < dim2; d2++) {
    res_data[0] = poind_data[0] - pred_data[0];
    res_data[1] = poind_data[1] - pred_data[1];
    res_data[2] = poind_data[0] + pred_data[2];
    res_data[3] = poind_data[1] + pred_data[3];
    poind_data += 2;
    pred_data += 4;
    res_data += 4;
  }

  return result;
}

inline std::vector<vsx::Tensor> GetPostProcessGenGrids(
    const std::vector<vsx::Tensor>& cls_scores,
    const std::vector<vsx::Tensor>& bbox_preds, int num_classes = 1203,
    const std::vector<int>& featmap_strides = {8, 16, 32}) {
  int num_base_peiors = 1;
  std::vector<vsx::TShape> featmap_sizes;
  for (auto& tensor : cls_scores) {
    featmap_sizes.push_back({tensor.Shape()[2], tensor.Shape()[3]});
  }
  auto mlvl_priors = GetGridPriors(featmap_sizes, featmap_strides);
  int len = 0;
  for (auto& mlvl : mlvl_priors) {
    len += mlvl.Shape()[0];
  }

  vsx::Tensor flatten_priors({len, 2}, vsx::Context::CPU(),
                             vsx::TypeFlag::kFloat32);
  float* dst = flatten_priors.MutableData<float>();
  for (auto& mlvl : mlvl_priors) {
    auto size = mlvl.GetSize();
    memcpy(dst, mlvl.Data<float>(), size * sizeof(float));
    dst += size;
  }

  std::vector<int> lens;
  lens.reserve(featmap_sizes.size());
  int sum_len = 0;
  for (auto feat_size : featmap_sizes) {
    int l = feat_size[0] * feat_size[1] * num_base_peiors;
    sum_len += l;
    lens.push_back(l);
  }

  vsx::Tensor flatten_stride({sum_len}, vsx::Context::CPU(),
                             vsx::TypeFlag::kInt32);
  int* flatten_stride_data = flatten_stride.MutableData<int>();
  for (size_t s = 0; s < lens.size(); s++) {
    for (size_t i = 0; i < lens[s]; i++) {
      flatten_stride_data[0] = featmap_strides[s];
      flatten_stride_data++;
    }
  }

  sum_len = 0;
  std::vector<vsx::Tensor> trans_cls_scores;
  trans_cls_scores.reserve(cls_scores.size());
  for (auto& score : cls_scores) {
    int dim = score.GetSize() / num_classes;
    vsx::Tensor flatten_score({dim, num_classes}, vsx::Context::CPU(),
                              vsx::TypeFlag::kFloat32);
    float* dst = flatten_score.MutableData<float>();
    const float* src = score.Data<float>();
    for (int d = 0; d < dim; d++) {
      for (int n = 0; n < num_classes; n++) {
        dst[d * num_classes + n] = vsx::sigmoid(src[n * dim + d]);
      }
    }
    trans_cls_scores.push_back(flatten_score);
    sum_len += dim;
  }

  auto flatten_cls_scores = vsx::Tensor(
      {sum_len, num_classes}, vsx::Context::CPU(), vsx::TypeFlag::kFloat32);
  dst = flatten_cls_scores.MutableData<float>();
  for (auto& score : trans_cls_scores) {
    const float* src = score.Data<float>();
    size_t size = score.GetSize();
    memcpy(dst, src, size * sizeof(float));
    dst += size;
  }

  sum_len = 0;
  std::vector<vsx::Tensor> trans_bbox_preds;
  trans_bbox_preds.reserve(bbox_preds.size());
  for (auto& box : bbox_preds) {
    int dim0 = 4;
    int dim1 = box.GetSize() / 4;
    vsx::Tensor flatten_box({dim1, dim0}, vsx::Context::CPU(),
                            vsx::TypeFlag::kFloat32);
    float* dst = flatten_box.MutableData<float>();
    const float* src = box.Data<float>();
    for (int d0 = 0; d0 < dim0; d0++) {
      for (int d1 = 0; d1 < dim1; d1++) {
        dst[d1 * dim0 + d0] = src[d0 * dim1 + d1];
      }
    }
    trans_bbox_preds.push_back(flatten_box);
    sum_len += dim1;
  }
  auto flatten_bbox_preds =
      vsx::Tensor({sum_len, 4}, vsx::Context::CPU(), vsx::TypeFlag::kFloat32);
  dst = flatten_bbox_preds.MutableData<float>();
  for (auto& bbox : trans_bbox_preds) {
    const float* src = bbox.Data<float>();
    size_t size = bbox.GetSize();
    memcpy(dst, src, size * sizeof(float));
    dst += size;
  }

  auto flatten_decoded_bboxes =
      GetBboxCoderDecode(flatten_priors, flatten_bbox_preds, flatten_stride);

  std::vector<vsx::Tensor> result{flatten_cls_scores, flatten_decoded_bboxes};
  return std::move(result);
}

inline std::vector<vsx::Tensor> FilterScoresAndTopK(const vsx::Tensor& scores,
                                                    float score_thresh,
                                                    int topk) {
  auto shape = scores.Shape();
  int height = shape[shape.ndim() - 2];
  int width = shape[shape.ndim() - 1];
  std::vector<std::tuple<float, int, int>> valid_scores;
  const float* data = scores.Data<float>();
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      if (data[0] > score_thresh) {
        valid_scores.push_back(std::make_tuple(data[0], h, w));
      }
      data++;
    }
  }
  int num_topk = topk < valid_scores.size() ? topk : valid_scores.size();

  std::partial_sort(
      valid_scores.begin(), valid_scores.begin() + num_topk, valid_scores.end(),
      [](std::tuple<float, int, int>& a, std::tuple<float, int, int>& b) {
        return std::get<0>(a) > std::get<0>(b);
      });
  vsx::Tensor score_result({num_topk}, vsx::Context::CPU(),
                           vsx::TypeFlag::kFloat32);
  vsx::Tensor labels({num_topk}, vsx::Context::CPU(), vsx::TypeFlag::kInt32);
  vsx::Tensor keep_idxs({num_topk}, vsx::Context::CPU(), vsx::TypeFlag::kInt32);

  float* sco_data = score_result.MutableData<float>();
  int* lab_data = labels.MutableData<int>();
  int* idx_data = keep_idxs.MutableData<int>();

  for (int s = 0; s < num_topk; s++) {
    sco_data[s] = std::get<0>(valid_scores[s]);
    idx_data[s] = std::get<1>(valid_scores[s]);
    lab_data[s] = std::get<2>(valid_scores[s]);
  }

  std::vector<vsx::Tensor> result{score_result, labels, keep_idxs};
  return std::move(result);
}
inline float IOU(const std::vector<float>& box1,
                 const std::vector<float>& box2) {
  // x1 y1 x2 y2
  float x_int1 = std::max(box1[0], box2[0]);
  float y_int1 = std::max(box1[1], box2[1]);
  float x_int2 = std::min(box1[2], box2[2]);
  float y_int2 = std::min(box1[3], box2[3]);
  float inter_area =
      std::max(0.0f, x_int2 - x_int1) * std::max(0.0f, y_int2 - y_int1);

  float box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
  float box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
  float iou = inter_area / (box1_area + box2_area - inter_area);
  return iou;
}

inline std::vector<vsx::Tensor> nms(const vsx::Tensor& bboxes,
                                    const vsx::Tensor& labels,
                                    const vsx::Tensor& scores,
                                    float iou_thresh) {
  std::vector<vsx::Tensor> result;
  result.reserve(3);
  const int* label_data = labels.Data<int>();
  int label_count = labels.GetSize();
  if (label_count == 0) {
    return result;
  }

  const float* box_data = bboxes.Data<float>();
  const float* score_data = scores.Data<float>();

  std::vector<int> keep_objects;

  for (int i = label_count - 1; i >= 0; i--) {
    bool keep = true;
    for (int j = i - 1; j >= 0; j--) {
      if (label_data[i] == label_data[j]) {
        auto iou = IOU({box_data[i * 4 + 0], box_data[i * 4 + 1],
                        box_data[i * 4 + 2], box_data[i * 4 + 3]},
                       {box_data[j * 4 + 0], box_data[j * 4 + 1],
                        box_data[j * 4 + 2], box_data[j * 4 + 3]});
        if (iou >= iou_thresh) {
          keep = false;
          break;
        }
      }
    }
    if (keep) {
      keep_objects.push_back(i);
    }
  }

  int result_count = keep_objects.size();
  vsx::Tensor result_box({result_count, 4}, vsx::Context::CPU(),
                         vsx::TypeFlag::kFloat32);
  vsx::Tensor result_labels({result_count}, vsx::Context::CPU(),
                            vsx::TypeFlag::kInt32);
  vsx::Tensor result_scores({result_count}, vsx::Context::CPU(),
                            vsx::TypeFlag::kFloat32);

  float* rb_data = result_box.MutableData<float>();
  float* rs_data = result_scores.MutableData<float>();
  int* rl_data = result_labels.MutableData<int>();

  for (int i = result_count - 1; i >= 0; i--) {
    rb_data[0] = box_data[keep_objects[i] * 4 + 0];
    rb_data[1] = box_data[keep_objects[i] * 4 + 1];
    rb_data[2] = box_data[keep_objects[i] * 4 + 2];
    rb_data[3] = box_data[keep_objects[i] * 4 + 3];
    rs_data[0] = score_data[keep_objects[i]];
    rl_data[0] = label_data[keep_objects[i]];

    rb_data += 4;
    rs_data++;
    rl_data++;
  }

  result.push_back(result_labels);
  result.push_back(result_scores);
  result.push_back(result_box);
  return result;
}
inline std::vector<vsx::Tensor> GetPostProcess(
    const std::vector<vsx::Tensor>& cls_scores,
    const std::vector<vsx::Tensor>& bbox_preds, float scale_factor,
    const std::vector<float>& pad_parmas, float score_thresh, int nms_pre,
    float iou_thresh, int max_per_image, int nms_threads) {
  auto flattens = GetPostProcessGenGrids(cls_scores, bbox_preds);
  auto scores_and_topk =
      FilterScoresAndTopK(flattens[0], score_thresh, nms_pre);

  vsx::Tensor& flatten_bboxes = flattens[1];
  auto keep_idxs = scores_and_topk[2];
  size_t idx_count = keep_idxs.GetSize();
  vsx::Tensor bboxes({static_cast<int>(idx_count), 4}, vsx::Context::CPU(),
                     vsx::TypeFlag::kFloat32);
  const int* idx_data = scores_and_topk[2].Data<int>();
  float* box_data = bboxes.MutableData<float>();
  const float* flatten_data = flatten_bboxes.Data<float>();

  for (int i = 0; i < idx_count; i++) {
    int idx = idx_data[i];
    box_data[0] = (flatten_data[idx * 4 + 0] - pad_parmas[2]) / scale_factor;
    box_data[1] = (flatten_data[idx * 4 + 1] - pad_parmas[0]) / scale_factor;
    box_data[2] = (flatten_data[idx * 4 + 2] - pad_parmas[2]) / scale_factor;
    box_data[3] = (flatten_data[idx * 4 + 3] - pad_parmas[0]) / scale_factor;

    box_data += 4;
  }

  auto& scores = scores_and_topk[0];
  auto& labels = scores_and_topk[1];
  auto result = nms(bboxes, labels, scores, iou_thresh, nms_threads);

  return result;
}
