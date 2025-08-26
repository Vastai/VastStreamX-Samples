
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <omp.h>
#include <torch/torch.h>

#include <mutex>
#include <thread>

#include "glog/logging.h"
#include "vaststreamx/vaststreamx.h"

template <typename T>
inline void print_torch_tensor(const torch::Tensor& tensor, int depth = 0,
                               bool first_line = false, bool new_line = true) {
  if (depth == 0) {
    std::cout << "Torch Tensor shape: [";
    for (auto s : tensor.sizes()) {
      std::cout << s << ",";
    }
    std::cout << "]\n";
  }
  if (tensor.numel() == 0) {
    std::cout << "Empty tensor" << std::endl;
    return;
  }
  if (!first_line) {
    for (int i = 0; i < depth; i++) std::cout << " ";
  }
  std::cout << "[";
  if (tensor.dim() <= 1) {
    for (int i = 0; i < tensor.numel(); i++) {
      if (i >= 3 && tensor.numel() - i > 3) {
        if (i == 3) std::cout << "... ";
      } else {
        std::cout << tensor[i].item<T>() << " ";
      }
    }
    std::cout << "]";
    if (new_line) std::cout << "\n";
  } else {
    int dims = tensor.sizes()[0];
    for (int i = 0; i < dims; i++) {
      if (i >= 3 && dims - i > 3) {
        if (i == 3) {
          for (int i = 0; i < depth; i++) std::cout << " ";
          std::cout << "... \n";
        }
      } else {
        print_torch_tensor<T>(tensor[i], depth + 1, i == 0, (dims - i != 1));
      }
    }
    std::cout << "]";
    if (new_line) std::cout << "\n\n";
  }
}

inline torch::Tensor VsxTensor2TorchTensor(const vsx::Tensor& vsx_tensor) {
  torch::Dtype dtype;
  switch (vsx_tensor.DType()) {
    case vsx::TypeFlag::kFloat16:
      dtype = torch::kFloat16;
      break;
    case vsx::TypeFlag::kFloat32:
      dtype = torch::kFloat32;
      break;
    default:
      CHECK(false) << "Unsupport vsx data type: " << vsx_tensor.DType();
  }

  std::vector<int64_t> shape;
  shape.reserve(vsx_tensor.Shape().ndim());
  for (int i = 0; i < vsx_tensor.Shape().ndim(); i++) {
    shape.push_back(vsx_tensor.Shape()[i]);
  }
  return torch::from_blob(vsx_tensor.MutableData<float>(), shape, dtype)
      .clone();
}

inline vsx::Tensor TorchTensor2VsxTensor(const torch::Tensor& torch_tensor) {
  std::vector<int64_t> shape;
  for (auto s : torch_tensor.sizes()) {
    shape.push_back(s);
  }

  vsx::TypeFlag dtype;
  int ele_size;

  if (torch_tensor.dtype() == torch::kFloat32) {
    dtype = vsx::TypeFlag::kFloat32;
    ele_size = sizeof(float);
  } else if (torch_tensor.dtype() == torch::kInt32) {
    dtype = vsx::TypeFlag::kInt32;
    ele_size = sizeof(float);
  } else {
    CHECK(false) << "Unsupport torch data type: "
                 << torch::toString(torch_tensor.dtype());
  }

  vsx::Tensor vsx_tensor(vsx::TShape(shape), vsx::Context::CPU(), dtype);
  memcpy(vsx_tensor.MutableData<void>(), torch_tensor.data_ptr(),
         torch_tensor.numel() * ele_size);  // 支持int32类型
  return vsx_tensor;
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
                                    const vsx::Tensor& scores, float iou_thresh,
                                    int topk, int nms_threads) {
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

  int num_threads = nms_threads;
  std::mutex mutex_lock;

  std::vector<std::thread> threads;
  for (int n = 0; n < num_threads; n++) {
    auto handle = std::thread(
        [&](int thread_id) {
          std::vector<int> keep_ids;
          for (int i = label_count - 1; i >= 0; i--) {
            if (label_data[i] % num_threads != thread_id) continue;
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
              keep_ids.push_back(i);
            }
          }
          mutex_lock.lock();
          keep_objects.insert(keep_objects.end(), keep_ids.begin(),
                              keep_ids.end());
          mutex_lock.unlock();
        },
        n);
    threads.push_back(std::move(handle));
  }
  for (auto& handle : threads) {
    handle.join();
  }
  int object_count = static_cast<int>(keep_objects.size());
  int result_count = object_count > topk ? topk : object_count;

  std::partial_sort(keep_objects.begin(), keep_objects.begin() + result_count,
                    keep_objects.end(), [](int a, int b) { return a < b; });

  vsx::Tensor result_box({result_count, 4}, vsx::Context::CPU(),
                         vsx::TypeFlag::kFloat32);
  vsx::Tensor result_labels({result_count}, vsx::Context::CPU(),
                            vsx::TypeFlag::kInt32);
  vsx::Tensor result_scores({result_count}, vsx::Context::CPU(),
                            vsx::TypeFlag::kFloat32);

  float* rb_data = result_box.MutableData<float>();
  float* rs_data = result_scores.MutableData<float>();
  int* rl_data = result_labels.MutableData<int>();

  for (int i = 0; i < result_count; i++) {
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

inline torch::Tensor GetL2Norm(const torch::Tensor& data, int axis = 1) {
  auto x0 = torch::abs(data);
  return data / x0.pow(2).sum(axis, true).sqrt();
}

inline torch::Tensor GetRes(const torch::Tensor& data_a,
                            const torch::Tensor& data_b, int h, int w,
                            float multiply_const, float add_const) {
  return data_a.permute({0, 2, 3, 1})
      .reshape({-1, 512})
      .matmul(data_b)
      .reshape({1, h, w, -1})
      .permute({0, 3, 1, 2})
      .mul(multiply_const)
      .add(add_const);
}

inline std::vector<torch::Tensor> GetScoresBatch(
    const std::vector<vsx::Tensor>& vsx_img_features,
    const vsx::Tensor& vsx_text_feature) {
  auto data_3489 = VsxTensor2TorchTensor(vsx_img_features[0]);
  auto data_3566 = VsxTensor2TorchTensor(vsx_img_features[1]);
  auto data_3643 = VsxTensor2TorchTensor(vsx_img_features[2]);

  auto text_feature = VsxTensor2TorchTensor(vsx_text_feature);

  auto res_l2_b = GetL2Norm(text_feature).transpose(1, 0);
  torch::Tensor res_3489, res_3566, res_3643;

  auto thread_3489 = std::thread([&]() {
    res_3489 = GetRes(data_3489, res_l2_b, 160, 160, 1.7583335638046265,
                      -12.202274322509766);
  });

  auto thread_3566 = std::thread([&]() {
    res_3566 = GetRes(data_3566, res_l2_b, 80, 80, 1.7852298021316528,
                      -10.711109161376953);
  });

  auto thread_3643 = std::thread([&]() {
    res_3643 = GetRes(data_3643, res_l2_b, 40, 40, 2.0692732334136963,
                      -9.184325218200684);
  });

  thread_3489.join();
  thread_3566.join();
  thread_3643.join();

  return {res_3489, res_3566, res_3643};
}

inline torch::Tensor SingleLevelGridPriors(
    const std::vector<int64_t>& featmat_size, int stride, float offset = 0.5) {
  int feat_h = featmat_size[featmat_size.size() - 2];
  int feat_w = featmat_size[featmat_size.size() - 1];
  int stride_w = stride, stride_h = stride;
  auto shift_x =
      (torch::arange(0, feat_w, torch::kFloat32) + offset) * stride_w;
  auto shift_y =
      (torch::arange(0, feat_h, torch::kFloat32) + offset) * stride_h;

  auto grid = torch::meshgrid({shift_y, shift_x}, "ij");
  auto shift_yy = grid[0].reshape(-1);
  auto shift_xx = grid[1].reshape(-1);
  auto shifts = torch::stack({shift_xx, shift_yy}, -1);
  return shifts;
}
inline std::vector<torch::Tensor> GetGridPriors(
    const std::vector<std::vector<int64_t>>& featmap_sizes,
    const std::vector<int>& strides) {
  std::vector<torch::Tensor> result;
  result.reserve(featmap_sizes.size());
  for (size_t i = 0; i < featmap_sizes.size(); i++) {
    result.push_back(SingleLevelGridPriors(featmap_sizes[i], strides[i]));
  }
  return result;
}

inline torch::Tensor GetBboxCoderDecode(const torch::Tensor& points,
                                        const torch::Tensor& pred_bboxes,
                                        const torch::Tensor& stride) {
  auto distance = pred_bboxes * stride.unsqueeze(0).unsqueeze(-1);
  auto x1 = points.select(-1, 0) - distance.select(-1, 0);
  auto y1 = points.select(-1, 1) - distance.select(-1, 1);
  auto x2 = points.select(-1, 0) + distance.select(-1, 2);
  auto y2 = points.select(-1, 1) + distance.select(-1, 3);
  auto bboxes = torch::stack({x1, y1, x2, y2}, -1);
  return bboxes;
}
inline std::vector<torch::Tensor> GetPostProcessGenGrids(
    const std::vector<torch::Tensor>& cls_scores,
    const std::vector<torch::Tensor>& bbox_preds, int num_classes = 1203,
    const std::vector<int>& featmap_strides = {8, 16, 32}) {
  int num_base_peiors = 1;
  std::vector<std::vector<int64_t>> featmap_sizes;
  for (auto& tensor : cls_scores) {
    featmap_sizes.push_back(
        {tensor.sizes()[tensor.dim() - 2], tensor.sizes()[tensor.dim() - 1]});
  }
  auto mlvl_priors = GetGridPriors(featmap_sizes, featmap_strides);
  auto flatten_priors = torch::cat(mlvl_priors);

  std::vector<torch::Tensor> mlvl_strides;
  mlvl_strides.reserve(featmap_strides.size());

  for (size_t s = 0; s < featmap_strides.size(); s++) {
    auto stride = flatten_priors.new_full(
        featmap_sizes[s][0] * featmap_sizes[s][1] * num_base_peiors,
        featmap_strides[s]);
    mlvl_strides.push_back(stride);
  }
  auto flatten_stride = torch::cat(mlvl_strides);

  std::vector<torch::Tensor> v_flatten_cls_scores;
  v_flatten_cls_scores.reserve(cls_scores.size());
  for (auto& cls_score : cls_scores) {
    v_flatten_cls_scores.emplace_back(
        cls_score.permute({0, 2, 3, 1}).reshape({1, -1, num_classes}));
  }

  std::vector<torch::Tensor> v_flatten_bbox_preds;
  v_flatten_bbox_preds.reserve(bbox_preds.size());
  for (auto& bbox_pred : bbox_preds) {
    v_flatten_bbox_preds.emplace_back(
        bbox_pred.permute({0, 2, 3, 1}).reshape({1, -1, 4}));
  }

  auto flatten_cls_scores =
      torch::cat(v_flatten_cls_scores, 1).sigmoid().squeeze();
  auto flatten_bbox_preds = torch::cat(v_flatten_bbox_preds, 1);

  auto flatten_decoded_bboxes =
      GetBboxCoderDecode(flatten_priors, flatten_bbox_preds, flatten_stride)
          .squeeze();

  return {flatten_cls_scores, flatten_decoded_bboxes};
}

inline std::vector<torch::Tensor> FilterScoresAndTopK(
    const torch::Tensor& scores, float score_thresh, int topk) {
  auto valid_mask = scores > score_thresh;
  auto valid_scores = scores.masked_select(valid_mask);
  auto valid_idxs = torch::nonzero(valid_mask);
  int num_topk = std::min(topk, static_cast<int>(valid_idxs.sizes()[0]));
  auto scores_idxs = torch::sort(valid_scores, 0, true);

  auto sel_scores = std::get<0>(scores_idxs).narrow(0, 0, num_topk);
  auto sel_idx = std::get<1>(scores_idxs).narrow(0, 0, num_topk);
  auto topk_idxs = torch::index_select(valid_idxs, 0, sel_idx);
  auto keep_idxs_labels = topk_idxs.unbind(1);

  return {sel_scores, keep_idxs_labels[1].to(torch::kInt32),
          keep_idxs_labels[0].to(torch::kInt32)};
}

inline std::vector<vsx::Tensor> FilterScoresAndTopK(const vsx::Tensor& scores,
                                                    float score_thresh,
                                                    int topk) {
  auto shape = scores.Shape();
  int height = shape[shape.ndim() - 2];
  int width = shape[shape.ndim() - 1];
  std::vector<std::tuple<float, int, int>> valid_scores;
  valid_scores.reserve(1000);
  const float* data = scores.Data<float>();
  std::mutex vec_lock;
#pragma omp parallel for
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      if (data[h * width + w] > score_thresh) {
        vec_lock.lock();
        valid_scores.push_back(std::make_tuple(data[h * width + w], h, w));
        vec_lock.unlock();
      }
    }
  }
  int score_size = static_cast<int>(valid_scores.size());
  int num_topk = topk < score_size ? topk : score_size;

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

  return result;
}

inline std::vector<vsx::Tensor> GetPostProcess(
    const std::vector<torch::Tensor>& cls_scores,
    const std::vector<vsx::Tensor>& bbox_preds, float scale_factor,
    const std::vector<float>& pad_parmas, float score_thresh, int nms_pre,
    float iou_thresh, int max_per_image, int nms_threads) {
  std::vector<torch::Tensor> torch_bbox_preds;
  torch_bbox_preds.reserve(bbox_preds.size());
  for (auto& box : bbox_preds) {
    torch_bbox_preds.push_back(VsxTensor2TorchTensor(box));
  }

  auto flattens = GetPostProcessGenGrids(cls_scores, torch_bbox_preds);

#if 0
  // use libtorch
  auto scores_and_topk =
      FilterScoresAndTopK(flattens[0], score_thresh, nms_pre);

  auto keep_idxs = scores_and_topk[2];
  auto flatten_decoded_bboxes = flattens[1];

  auto bboxes = torch::index_select(flatten_decoded_bboxes, 0, keep_idxs);

  bboxes -= torch::tensor(
      {pad_parmas[2], pad_parmas[0], pad_parmas[2], pad_parmas[0]},
      torch::kFloat32);
  bboxes /= torch::tensor({scale_factor}, torch::kFloat32).repeat({1, 4});

  auto vsx_bboxes = TorchTensor2VsxTensor(bboxes);
  auto vsx_scores = TorchTensor2VsxTensor(scores_and_topk[0]);
  auto vsx_labels = TorchTensor2VsxTensor(scores_and_topk[1]);
  auto result = nms(vsx_bboxes, vsx_labels, vsx_scores, iou_thresh);

#else

  auto flattens_scores = TorchTensor2VsxTensor(flattens[0]);
  auto flatten_bboxes = TorchTensor2VsxTensor(flattens[1]);

  auto scores_and_topk =
      FilterScoresAndTopK(flattens_scores, score_thresh, nms_pre);

  auto keep_idxs = scores_and_topk[2];
  size_t idx_count = keep_idxs.GetSize();
  vsx::Tensor bboxes({static_cast<int>(idx_count), 4}, vsx::Context::CPU(),
                     vsx::TypeFlag::kFloat32);
  const int* idx_data = scores_and_topk[2].Data<int>();
  float* box_data = bboxes.MutableData<float>();
  const float* flatten_data = flatten_bboxes.Data<float>();
#pragma omp parallel for
  for (size_t i = 0; i < idx_count; i++) {
    int idx = idx_data[i];
    float* box = box_data + i * 4;
    box[0] = (flatten_data[idx * 4 + 0] - pad_parmas[2]) / scale_factor;
    box[1] = (flatten_data[idx * 4 + 1] - pad_parmas[0]) / scale_factor;
    box[2] = (flatten_data[idx * 4 + 2] - pad_parmas[2]) / scale_factor;
    box[3] = (flatten_data[idx * 4 + 3] - pad_parmas[0]) / scale_factor;
  }

  auto& scores = scores_and_topk[0];
  auto& labels = scores_and_topk[1];

  auto result =
      nms(bboxes, labels, scores, iou_thresh, max_per_image, nms_threads);
#endif

  return result;
}