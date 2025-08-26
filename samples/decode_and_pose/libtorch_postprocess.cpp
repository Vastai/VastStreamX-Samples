
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "libtorch_postprocess.h"

#include <torch/torch.h>
#include <torchvision/ops/nms.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"
#include "vaststreamx/vaststreamx.h"

#define CALCULATE_EXECUTION_TIME(label, statement)                             \
  do {                                                                         \
    auto start_time = std::chrono::high_resolution_clock::now();               \
    statement;                                                                 \
    auto end_time = std::chrono::high_resolution_clock::now();                 \
    auto s = std::chrono::duration_cast<std::chrono::microseconds>(end_time -  \
                                                                   start_time) \
                 .count();                                                     \
    std::cout << label << " - Execution time: " << s << " microseconds"        \
              << std::endl;                                                    \
  } while (0)

namespace ti = torch::indexing;

static void SAVE_TENSOR(std::string filename, torch::Tensor tsr) {
  std::ofstream file(filename);
  file << tsr << std::endl;
  file.close();
  return;
}

static void SAVE_TENSOR_3D(std::string path_prefix, torch::Tensor tsr) {
  for (int i = 0; i < tsr.size(0); ++i) {
    std::ofstream file(path_prefix + std::to_string(i) + ".txt");
    file << tsr[i] << std::endl;
    file.close();
  }
  return;
}

static void SAVE_TENSOR_BIN(std::string path_prefix, torch::Tensor tsr) {
  std::ofstream ofs(path_prefix, std::ios_base::binary);
  auto tmp = tsr.contiguous();
  ofs.write((char *)tmp.data_ptr(), tmp.nbytes());
  ofs.close();
  return;
}

static void VsxTensorToTorchTensor(const vsx::Tensor &vsxTensor,
                                   torch::Tensor &torchTensor) {
  auto torch_type = torch::kFloat32;
  switch (vsxTensor.GetDType()) {
    case vsx::TypeFlag::kUint8:
      torch_type = torch::kUInt8;
      break;
    case vsx::TypeFlag::kInt8:
      torch_type = torch::kInt8;
      break;
    case vsx::TypeFlag::kUint16:
      LOG(FATAL) << "LibTorch unsupport type of kUint16";
      break;
    case vsx::TypeFlag::kUint32:
      LOG(FATAL) << "LibTorch unsupport type of kUint32";
      break;
    case vsx::TypeFlag::kBfloat16:
      LOG(FATAL) << "LibTorch unsupport type of kBfloat16";
      break;
    case vsx::TypeFlag::kInt16:
      torch_type = torch::kInt16;
      break;
    case vsx::TypeFlag::kInt32:
      torch_type = torch::kInt32;
      break;
    case vsx::TypeFlag::kFloat16:
      torch_type = torch::kFloat16;
      break;
    case vsx::TypeFlag::kFloat32:
      torch_type = torch::kFloat32;
      break;
    default:
      break;
  }

  std::vector<int64_t> shape;
  shape.reserve(vsxTensor.Shape().ndim());

  for (size_t i = 0; i < vsxTensor.Shape().ndim(); i++) {
    shape.push_back(vsxTensor.Shape()[i]);
  }

  char *src_data = vsxTensor.MutableData<char>();
  torchTensor = torch::from_blob(src_data, shape,
                                 torch::TensorOptions().dtype(torch_type))
                    .to(torch_type);
}
static void TorchTensorToVsxTensor(torch::Tensor &torh,
                                   vsx::Tensor &vsxTensor) {
  auto vsx_type = vsx::TypeFlag::kUint8;
  auto torchTensor = torh.contiguous();
  assert(torchTensor.is_contiguous());  // fix
  switch (torchTensor.scalar_type()) {
    case torch::kUInt8:
      vsx_type = vsx::TypeFlag::kUint8;
      break;
    case torch::kInt8:
      vsx_type = vsx::TypeFlag::kInt8;
      break;
    case torch::kInt16:
      vsx_type = vsx::TypeFlag::kInt16;
      break;
    case torch::kInt32:
      vsx_type = vsx::TypeFlag::kInt32;
      break;
    case torch::kInt64:
      LOG(FATAL) << "Vsx unsupport type of kInt64";
      break;
    case torch::kFloat16:
      vsx_type = vsx::TypeFlag::kFloat16;
      break;
    case torch::kFloat32:
      vsx_type = vsx::TypeFlag::kFloat32;
      break;
    case torch::kFloat64:
      LOG(FATAL) << "Vsx unsupport type of kFloat64";
      break;
  }

  std::vector<int64_t> shape;
  shape.reserve(torchTensor.sizes().size());
  for (size_t i = 0; i < torchTensor.sizes().size(); i++) {
    shape.push_back(torchTensor.sizes()[i]);
  }

  vsxTensor = vsx::Tensor(vsx::TShape{shape}, vsx::Context::CPU(), vsx_type);
  size_t men_len =
      torchTensor.numel() *
      torchTensor
          .element_size();  // bug , you must make sure the data is contignous
  char *src = (char *)torchTensor.data_ptr();
  char *dst = vsxTensor.MutableData<char>();
  memcpy(dst, src, men_len);
}

static void SAVE_TENSOR_NUMPY(std::string path_prefix, torch::Tensor tsr) {
  vsx::Tensor tmp;
  TorchTensorToVsxTensor(tsr, tmp);
  vsx::SaveTensor(path_prefix, tmp);
  return;
}

static std::vector<torch::Tensor> make_anchors(
    std::vector<torch::Tensor> feats, std::vector<int> strides,
    float grid_cell_offset = 0.5) {  // double  -> float
  std::vector<torch::Tensor> anchor_points, stride_tensor;
  assert(feats.size() != 0);
  auto dtype = feats[0].dtype();
  auto device = feats[0].device();
  for (int i = 0; i < strides.size(); i++) {
    int h = feats[i].size(2);
    int w = feats[i].size(3);
    auto sx = torch::arange(w, dtype) + grid_cell_offset;
    auto sy = torch::arange(h, dtype) + grid_cell_offset;
    auto meshgrid = torch::meshgrid({sy, sx});
    sx = meshgrid[1];
    sy = meshgrid[0];

    anchor_points.push_back(torch::stack({sx, sy}, -1).view({-1, 2}));
    stride_tensor.push_back(torch::full({h * w, 1}, strides[i], dtype));
  }
  return {torch::cat(anchor_points), torch::cat(stride_tensor)};
}

static torch::Tensor dist2bbox(torch::Tensor &distance,
                               torch::Tensor &anchor_points, bool xywh = true,
                               int dim = -1) {
  auto chunk = distance.chunk(2, dim);
  torch::Tensor lt = chunk[0];
  torch::Tensor rb = chunk[1];
  torch::Tensor x1y1 = anchor_points - lt;
  torch::Tensor x2y2 = anchor_points + rb;
  if (xywh) {
    torch::Tensor c_xy = (x1y1 + x2y2) / 2;
    torch::Tensor wh = x2y2 - x1y1;
    return torch::cat({c_xy, wh}, dim);
  } else {
    return torch::cat({x1y1, x2y2}, dim);
  }
}

class DFL {
 public:
  DFL(int c1 = 16) {
    conv_ = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(c1, 1, 1).bias(false).stride(1).padding(0));
    torch::Tensor x = torch::arange(c1, torch::kFloat);
    conv_->weight.data().copy_(x.view({1, c1, 1, 1}));
    c1_ = c1;
  }
  torch::Tensor Forward(const torch::Tensor &x) {
    auto x_shape = x.sizes();

    auto b = x_shape[0];
    auto c = x_shape[1];
    auto a = x_shape[2];

    auto y = x.view({b, 4, c1_, a}).transpose(2, 1).softmax(1);

    auto output = conv_->forward(y);

    return output.view({b, 4, a});
  }

 private:
  torch::nn::Conv2d conv_ = torch::nn::Conv2d(
      torch::nn::Conv2dOptions(16, 1, 1).bias(false).stride(1).padding(0));
  int c1_ = 16;
};

static torch::Tensor kpts_decode(torch::Tensor &kpts, torch::Tensor &anchors,
                                 torch::Tensor &strides) {
  uint32_t ndim = 3;
  auto y = kpts.clone();
  y.index({ti::Slice(), ti::Slice(2, ti::None, 3)}).sigmoid_();
  y.index({ti::Slice(), ti::Slice(0, ti::None, ndim)}) =
      (y.index({ti::Slice(), ti::Slice(0, ti::None, ndim)}) * 2.0 +
       (anchors[0] - 0.5)) *
      strides;
  y.index({ti::Slice(), ti::Slice(1, ti::None, ndim)}) =
      (y.index({ti::Slice(), ti::Slice(1, ti::None, ndim)}) * 2.0 +
       (anchors[1] - 0.5)) *
      strides;
  return y;
}

static torch::Tensor xywh2xyxy(torch::Tensor &x) {
  auto y = x.clone();
  y.index({torch::indexing::Slice(), 0}) =
      x.index({torch::indexing::Slice(), 0}) -
      x.index({torch::indexing::Slice(), 2}) / 2;
  y.index({torch::indexing::Slice(), 1}) =
      x.index({torch::indexing::Slice(), 1}) -
      x.index({torch::indexing::Slice(), 3}) / 2;
  y.index({torch::indexing::Slice(), 2}) =
      x.index({torch::indexing::Slice(), 0}) +
      x.index({torch::indexing::Slice(), 2}) / 2;
  y.index({torch::indexing::Slice(), 3}) =
      x.index({torch::indexing::Slice(), 1}) +
      x.index({torch::indexing::Slice(), 3}) / 2;

  return y;
}

static torch::Tensor UsingNumIndex1D(torch::Tensor &source,
                                     torch::Tensor &num_index) {
  // 遍历
  int cnt = num_index.size(0);
  for (int i = 0; i < cnt; ++i) {
    source[i] = source[num_index[i]];  // 要求升序
  }
  return source.slice(0, 0, cnt, 1).clone();
}

static std::vector<torch::Tensor> non_max_suppression(
    torch::Tensor &prediction, float conf_thres = 0.001, float iou_thres = 0.65,
    torch::IntArrayRef classes = torch::IntArrayRef(), bool agnostic = false,
    bool multi_label = false, torch::IntArrayRef labels = torch::IntArrayRef(),
    int max_det = 300, int nc = 1, float max_time_img = 0.05,
    int max_nms = 30000, int max_wh = 7680) {
  int bs = prediction.sizes()[0];
  nc = nc > 0 ? nc : (prediction.sizes()[1] - 4);
  int nm = prediction.sizes()[1] - nc - 4;
  int mi = 4 + nc;
  torch::Tensor xc =
      prediction.index({ti::Slice(), ti::Slice(4, mi)}).amax(1) > conf_thres;
  std::vector<torch::Tensor> output(bs, torch::zeros({0, 6 + nm}));

  for (int xi = 0; xi < prediction.size(0); xi++) {
    auto x = prediction[xi];
    x = x.transpose(0, -1).index({xc[xi]});

    if (x.size(0) == 0) {
      continue;
    }

    auto x_list = x.split_with_sizes({4, nc, nm}, 1);
    auto box = xywh2xyxy(x_list[0]);
    auto cls = x_list[1];
    auto mask = x_list[2];

    auto conf_j = cls.max(1, true);
    auto conf = std::get<0>(conf_j);
    auto j = std::get<1>(conf_j);
    conf = conf.squeeze(1);
    j = j.squeeze(1);
    torch::Tensor cond = (conf > conf_thres).nonzero().squeeze();
    x = torch::cat(
        {box.index_select(0, cond), conf.index_select(0, cond).unsqueeze(1),
         j.index_select(0, cond).to(torch::kFloat).unsqueeze(1),
         mask.index_select(0, cond)},
        1);

    int n = x.size(0);

    if (n == 0) continue;

    torch::Tensor column = x.index({torch::indexing::Slice(), 4});

    // 对列进行降序排序
    torch::Tensor sorted_indices = torch::argsort(column, 0, true);

    // 获取前 max_nms 个排序后的索引
    torch::Tensor selected_indices =
        sorted_indices.index({torch::indexing::Slice(ti::None, max_nms)});

    // 根据索引选择张量的子集
    torch::Tensor selected_x = x.index({selected_indices});

    // 更新 x
    x = selected_x;

    auto c = x.index({"...", ti::Slice({5, 6})}) * (agnostic ? 0 : max_wh);
    auto boxes = (x.index({ti::Slice(), ti::Slice(0, 4)}) + c).to(torch::kCPU);
    auto scores = x.index({ti::Slice(), 4});
    auto keep = vision::ops::nms(boxes, scores, iou_thres);
    keep = keep.to(torch::kCPU)
               .index({ti::Slice(0, std::min(max_det, (int)keep.size(0)))});
    output[xi] = UsingNumIndex1D(x, keep);
  }

  return output;
}

static torch::Tensor crop_mask(torch::Tensor &masks, torch::Tensor &boxes) {
  auto size = masks.sizes();
  auto n = size[0], h = size[1], w = size[2];
  auto box = torch::chunk(boxes.unsqueeze(2), 4, 1);
  auto x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];
  auto r = torch::arange(w, c10::TensorOptions().dtype(x1.dtype()));
  auto c = torch::arange(h, c10::TensorOptions().dtype(x1.dtype()));
  r = at::reshape(r, {1, 1, w});
  c = at::reshape(c, {1, h, 1});
  return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2));
}

static torch::Tensor process_mask_upsample(torch::Tensor &protos,
                                           torch::Tensor &masks_in,
                                           torch::Tensor &bboxes,
                                           const std::vector<int64_t> &shape) {
  auto c = protos.size(0);
  auto mh = protos.size(1);
  auto mw = protos.size(2);
  auto masks = (torch::mm(masks_in, protos.to(torch::kFloat32).view({c, -1})))
                   .sigmoid()
                   .view({-1, mh, mw});

  namespace F = torch::nn::functional;
  masks = F::interpolate(masks.unsqueeze(0), F::InterpolateFuncOptions()
                                                 .size(shape)
                                                 .mode(torch::kBilinear)
                                                 .align_corners(false))[0];
  masks = crop_mask(masks, bboxes);
  return masks.gt_(0.5);
}

void clip_coords(torch::Tensor &boxes, const std::vector<int64_t> &shape) {
  boxes.index({"...", 0}).clamp_(0, shape[1]);
  boxes.index({"...", 1}).clamp_(0, shape[0]);
}

static torch::Tensor scale_coords(const std::vector<int64_t> &img1_shape,
                                  torch::Tensor &coords,
                                  const std::vector<int64_t> &img0_shape) {
  auto gain = std::min(1.0 * img1_shape[0] / img0_shape[0],
                       1.0 * img1_shape[1] / img0_shape[1]);
  auto pad0 = (img1_shape[1] - img0_shape[1] * gain) / 2.0;
  auto pad1 = (img1_shape[0] - img0_shape[0] * gain) / 2.0;
  coords.index({"...", 0}) = coords.index({"...", 0}) - pad0;
  coords.index({"...", 1}) = coords.index({"...", 1}) - pad1;
  coords.index({"...", 2}) = coords.index({"...", 2}) - pad0;
  coords.index({"...", 3}) = coords.index({"...", 3}) - pad1;
  coords.index({"...", ti::Slice({ti::None, 4})}) =
      coords.index({"...", ti::Slice({ti::None, 4})}) / gain;

  clip_coords(coords, img0_shape);

  return coords;
}

static torch::Tensor scale_coords_kpts(
    const std::vector<int64_t> &img1_shape, torch::Tensor &coords,
    const std::vector<int64_t> &img0_shape,
    const std::vector<std::vector<double>> &ratio_pad, bool normalize = false) {
  double gain = 0;
  std::vector<double> pad;
  if (ratio_pad.empty()) {
    gain = std::min(img1_shape[0] / img0_shape[0],
                    img1_shape[1] / img0_shape[1]);  // gain  = old / new
    pad = {(img1_shape[1] - img0_shape[1] * gain) / 2,
           (img1_shape[0] - img0_shape[0] * gain) / 2};  // wh padding
  } else {
    gain = ratio_pad[0][0];
    pad = ratio_pad[1];
  }

  coords.index({"...", 0}) = coords.index({"...", 0}) - (double)pad[0];
  coords.index({"...", 1}) = coords.index({"...", 1}) - (double)pad[1];
  coords.index({"...", 0}) = coords.index({"...", 0}) / gain;
  coords.index({"...", 1}) = coords.index({"...", 1}) / gain;
  clip_coords(coords, img0_shape);
  if (normalize) {
    coords.index({"...", 0}) = coords.index({"...", 0}) / img0_shape[1];
    coords.index({"...", 1}) = coords.index({"...", 1}) / img0_shape[0];
  }

  return coords;
}

static torch::Tensor scale_image(
    torch::Tensor &masks, const std::vector<int64_t> &img0_shape,
    const std::vector<std::vector<float>> &ratio_pad) {
  auto img1_shape = masks.sizes();
  if (img0_shape[0] == img1_shape[0] &&
      img0_shape[1] == img1_shape[1]) {  // (h,w)
    return masks;
  }
  float gain, pad0, pad1;
  if (ratio_pad.size() == 0) {
    gain = std::min(1.0 * img1_shape[0] / img0_shape[0],
                    1.0 * img1_shape[1] / img0_shape[1]);
    pad0 = (img1_shape[1] - img0_shape[1] * gain) / 2.0;
    pad1 = (img1_shape[0] - img0_shape[0] * gain) / 2.0;
  } else {
    gain = ratio_pad[0][0];
    pad0 = ratio_pad[1][0];
    pad1 = ratio_pad[1][1];
  }

  int top = static_cast<int>(pad1);
  int left = static_cast<int>(pad0);
  int bottom = static_cast<int>(img1_shape[0] - pad1);
  int right = static_cast<int>(img1_shape[1] - pad0);

  if (masks.size(0) < 2) {
    std::cout << ("len of masks shape should be 2 or 3, but got " +
                  std::to_string(masks.size(0)));
    assert(masks.size(0) >= 2);
  }

  auto new_masks = masks
                       .index({torch::indexing::Slice(top, bottom),
                               torch::indexing::Slice(left, right)})
                       .contiguous();
  auto new_width = img0_shape[1];
  auto new_height = img0_shape[0];
  cv::Mat input_mat(new_masks.size(0), new_masks.size(1),
                    CV_MAKETYPE(CV_8U, new_masks.size(2)),
                    new_masks.data_ptr<uint8_t>());
  cv::Mat resized_mat;
  cv::resize(input_mat, resized_mat, cv::Size(new_width, new_height));
  torch::Tensor resized_tensor =
      torch::from_blob(resized_mat.data,
                       {new_height, new_width, resized_mat.channels()},
                       torch::kUInt8)
          .clone();  // some thing wrong

  return resized_tensor;
}

std::vector<PoseResult> post_process(
    const std::vector<vsx::Tensor> &infer_output,
    const std::vector<int64_t> &model_size, uint32_t num_output,
    const std::vector<vsx::TShape> &output_shape,
    const vsx::TShape &image_shape, float conf_thres, float iou_thres) {
  const uint32_t height = image_shape[0];
  const uint32_t width = image_shape[1];
  std::vector<torch::Tensor> stream_ouput;
  uint32_t ttt = 0;
  for (auto tensor : infer_output) {
    torch::Tensor t;
    VsxTensorToTorchTensor(tensor, t);
    stream_ouput.push_back(t);
  }

  auto dfl = DFL(16);
  for (uint32_t i = 0; i < num_output; ++i) {
    std::vector<int64_t> shape;
    for (auto j : output_shape[i]) {
      shape.push_back(j);
    }

    stream_ouput[i] = torch::Tensor(stream_ouput[i].reshape(shape));
  }

  std::vector<torch::Tensor> output;
  for (int i = 0; i < 3; ++i) {
    auto x = torch::cat({stream_ouput[i * 2], stream_ouput[i * 2 + 1]}, 1);
    output.push_back(x);
  }

  auto tensors = make_anchors(output, {8, 16, 32}, 0.5);
  auto anchors = tensors[0].transpose(0, 1);
  auto strides = tensors[1].transpose(0, 1);
  tensors.clear();
  uint32_t i = 0;
  for (auto t : output) {
    tensors.push_back(t.view({1, 65, -1}));
  }
  auto x_cat = torch::cat(tensors, 2);
  torch::Tensor box = x_cat.index({ti::Slice(), ti::Slice(0, 16 * 4)});
  torch::Tensor cls = x_cat.index({ti::Slice(), ti::Slice(16 * 4, ti::None)});
  auto box_dfl = dfl.Forward(box);
  auto tmp = anchors.unsqueeze(0);
  auto dbox = dist2bbox(box_dfl, tmp, true, 1) * strides;
  auto ty = torch::cat({dbox, cls.sigmoid()}, 1);
  auto &det_out0 = ty;
  auto &det_out1 = output;

  // keypoints cat
  std::vector<torch::Tensor> tkpt;
  for (size_t i = 6; i < 9; i++) {
    tkpt.push_back(stream_ouput[i].view({1, 51, -1}));
  }
  auto kpt = torch::cat(tkpt, -1);
  auto pkpt = kpts_decode(kpt, anchors, strides);
  auto kpt_out = torch::cat({det_out0, pkpt}, 1);
  auto pred = non_max_suppression(kpt_out, conf_thres, iou_thres)[0];
  auto npr = pred.size(0);
  auto predn = pred.clone();

  if (npr > 0) {
    auto pred_kpts =
        predn.index({"...", ti::Slice({6, ti::None})}).view({npr, 17, -1});
    auto scale_r = (double)model_size[0] / std::max(height, width);
    auto pad_h = (model_size[0] - height * scale_r) / 2;
    auto pad_w = (model_size[0] - width * scale_r) / 2;
    scale_coords_kpts(model_size, pred_kpts, {height, width},
                      {{scale_r, scale_r}, {pad_w, pad_h}});
    predn.index({"...", ti::Slice(6, ti::None)}) =
        pred_kpts.clone().reshape({npr, -1});
    auto tmp = pred.index({"...", ti::Slice(ti::None, 4)});
    predn.index({"...", ti::Slice(ti::None, 4)}) =
        scale_coords(model_size, tmp, {height, width, 3}).round();
  }

  std::vector<PoseResult> objs;
  for (int i = 0; i < predn.sizes()[0]; i++) {
    PoseResult obj;
    auto id = predn[i][5].item().toInt();
    obj.classId = id;
    auto score = predn[i][4].item().toFloat();
    obj.score = score;
    // if (score < 0.1)
    //     continue;

    obj.x = predn[i][0].item().toInt();
    obj.y = predn[i][1].item().toInt();
    obj.w = predn[i][2].item().toInt() - obj.x;
    obj.h = predn[i][3].item().toInt() - obj.y;
    for (int k = 0; k < 17; k++) {
      auto kpsX = predn[i][6 + 3 * k].item().toFloat();
      auto kpsY = predn[i][6 + 3 * k + 1].item().toFloat();
      auto kpsS = predn[i][6 + 3 * k + 2].item().toFloat();
      obj.keyPoints.push_back(kpsX);
      obj.keyPoints.push_back(kpsY);
      obj.keyPoints.push_back(kpsS > 0.5f ? 2.0f : 1.0f);
    }
    objs.push_back(obj);
  }

  return objs;
}
