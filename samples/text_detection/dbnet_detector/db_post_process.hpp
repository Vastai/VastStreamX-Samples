// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <math.h>

#include <iostream>
#include <map>
#include <vector>

#include "clipper.h"  //  clipper.hpp -> clipper.h
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"
#include "vaststreamx/vaststreamx.h"

namespace vsx {

struct DBresult {
  std::vector<float> scores;
  std::vector<std::vector<std::vector<int>>> boxes;
};

struct FindContourOpInfo {
  std::string elf_path;
  float thresh;
  uint32_t device_id;
  uint32_t max_contour_num;
};

template <class T>
T clamp(T x, T min, T max) {
  if (x > max) return max;
  if (x < min) return min;
  return x;
}

inline std::vector<std::vector<float>> Mat2Vector(cv::Mat mat) {
  std::vector<std::vector<float>> img_vec;
  std::vector<float> tmp;

  for (int i = 0; i < mat.rows; ++i) {
    tmp.clear();
    for (int j = 0; j < mat.cols; ++j) {
      tmp.push_back(mat.at<float>(i, j));
    }
    img_vec.push_back(tmp);
  }
  return img_vec;
}

inline void GetContourArea(std::vector<std::vector<float>> box,
                           float unclip_ratio, float &distance) {
  int pts_num = 4;
  float area = 0.0f;
  float dist = 0.0f;
  for (int i = 0; i < pts_num; i++) {
    area += box[i][0] * box[(i + 1) % pts_num][1] -
            box[i][1] * box[(i + 1) % pts_num][0];
    dist += sqrtf((box[i][0] - box[(i + 1) % pts_num][0]) *
                      (box[i][0] - box[(i + 1) % pts_num][0]) +
                  (box[i][1] - box[(i + 1) % pts_num][1]) *
                      (box[i][1] - box[(i + 1) % pts_num][1]));
  }
  area = fabs(float(area / 2.0));

  distance = area * unclip_ratio / dist;
}

inline cv::RotatedRect Unclip(std::vector<std::vector<float>> box,
                              float unclip_ratio) {
  float distance = 1.0;

  GetContourArea(box, unclip_ratio, distance);

  ClipperLib::ClipperOffset offset;
  ClipperLib::Path p;
  p << ClipperLib::IntPoint(static_cast<int>(box[0][0]),
                            static_cast<int>(box[0][1]))
    << ClipperLib::IntPoint(static_cast<int>(box[1][0]),
                            static_cast<int>(box[1][1]))
    << ClipperLib::IntPoint(static_cast<int>(box[2][0]),
                            static_cast<int>(box[2][1]))
    << ClipperLib::IntPoint(static_cast<int>(box[3][0]),
                            static_cast<int>(box[3][1]));
  offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

  ClipperLib::Paths soln;
  offset.Execute(soln, distance);
  std::vector<cv::Point2f> points;

  for (size_t j = 0; j < soln.size(); j++) {
    for (size_t i = 0; i < soln[soln.size() - 1].size(); i++) {
      points.emplace_back(soln[j][i].X, soln[j][i].Y);
    }
  }
  cv::RotatedRect res = cv::minAreaRect(points);

  return res;
}

inline bool XsortFp32(std::vector<float> a, std::vector<float> b) {
  if (a[0] != b[0]) return a[0] < b[0];
  return false;
}

inline bool XsortInt(std::vector<int> a, std::vector<int> b) {
  if (a[0] != b[0]) return a[0] < b[0];
  return false;
}

inline std::vector<std::vector<int>> OrderPointsClockwise(
    std::vector<std::vector<int>> pts) {
  std::vector<std::vector<int>> box = pts;
  std::sort(box.begin(), box.end(), XsortInt);

  std::vector<std::vector<int>> leftmost = {box[0], box[1]};
  std::vector<std::vector<int>> rightmost = {box[2], box[3]};

  if (leftmost[0][1] > leftmost[1][1]) std::swap(leftmost[0], leftmost[1]);

  if (rightmost[0][1] > rightmost[1][1]) std::swap(rightmost[0], rightmost[1]);

  std::vector<std::vector<int>> rect = {leftmost[0], rightmost[0], rightmost[1],
                                        leftmost[1]};
  return rect;
}

inline std::vector<std::vector<float>> GetMiniBoxes(cv::RotatedRect box,
                                                    float &ssid) {
  ssid = std::min(box.size.width, box.size.height);

  cv::Mat points;
  cv::boxPoints(box, points);

  auto array = Mat2Vector(points);
  std::sort(array.begin(), array.end(), XsortFp32);

  std::vector<float> idx1 = array[0], idx2 = array[1], idx3 = array[2],
                     idx4 = array[3];
  if (array[3][1] <= array[2][1]) {
    idx2 = array[3];
    idx3 = array[2];
  } else {
    idx2 = array[2];
    idx3 = array[3];
  }
  if (array[1][1] <= array[0][1]) {
    idx1 = array[1];
    idx4 = array[0];
  } else {
    idx1 = array[0];
    idx4 = array[1];
  }

  array[0] = idx1;
  array[1] = idx2;
  array[2] = idx3;
  array[3] = idx4;

  return array;
}

inline float BoxScoreFast(std::vector<std::vector<float>> box_array,
                          cv::Mat pred) {
  auto array = box_array;
  int width = pred.cols;
  int height = pred.rows;

  float box_x[4] = {array[0][0], array[1][0], array[2][0], array[3][0]};
  float box_y[4] = {array[0][1], array[1][1], array[2][1], array[3][1]};

  int xmin =
      clamp(static_cast<int>(::floorf(*(std::min_element(box_x, box_x + 4)))),
            0, width - 1);
  int xmax =
      clamp(static_cast<int>(::ceilf(*(std::max_element(box_x, box_x + 4)))), 0,
            width - 1);
  int ymin =
      clamp(static_cast<int>(::floorf(*(std::min_element(box_y, box_y + 4)))),
            0, height - 1);
  int ymax =
      clamp(static_cast<int>(::ceilf(*(std::max_element(box_y, box_y + 4)))), 0,
            height - 1);

  cv::Mat mask;
  mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  cv::Point root_point[4];
  root_point[0] = cv::Point(static_cast<int>(array[0][0]) - xmin,
                            static_cast<int>(array[0][1]) - ymin);
  root_point[1] = cv::Point(static_cast<int>(array[1][0]) - xmin,
                            static_cast<int>(array[1][1]) - ymin);
  root_point[2] = cv::Point(static_cast<int>(array[2][0]) - xmin,
                            static_cast<int>(array[2][1]) - ymin);
  root_point[3] = cv::Point(static_cast<int>(array[3][0]) - xmin,
                            static_cast<int>(array[3][1]) - ymin);
  const cv::Point *ppt[1] = {root_point};
  int npt[] = {4};
  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

  cv::Mat croppedImg;
  pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
      .copyTo(croppedImg);

  auto score = cv::mean(croppedImg, mask)[0];
  return score;
}

inline float PolygonScoreAcc(std::vector<cv::Point> contour, cv::Mat pred) {
  int width = pred.cols;
  int height = pred.rows;
  std::vector<float> box_x;
  std::vector<float> box_y;
  for (size_t i = 0; i < contour.size(); ++i) {
    box_x.push_back(contour[i].x);
    box_y.push_back(contour[i].y);
  }

  int xmin =
      clamp(int(std::floor(*(std::min_element(box_x.begin(), box_x.end())))), 0,
            width - 1);
  int xmax =
      clamp(int(std::ceil(*(std::max_element(box_x.begin(), box_x.end())))), 0,
            width - 1);
  int ymin =
      clamp(int(std::floor(*(std::min_element(box_y.begin(), box_y.end())))), 0,
            height - 1);
  int ymax =
      clamp(int(std::ceil(*(std::max_element(box_y.begin(), box_y.end())))), 0,
            height - 1);

  cv::Mat mask;
  mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  cv::Point *rook_point = new cv::Point[contour.size()];

  for (size_t i = 0; i < contour.size(); ++i) {
    rook_point[i] = cv::Point(int(box_x[i]) - xmin, int(box_y[i]) - ymin);
  }
  const cv::Point *ppt[1] = {rook_point};
  int npt[] = {int(contour.size())};

  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

  cv::Mat croppedImg;
  pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
      .copyTo(croppedImg);
  float score = cv::mean(croppedImg, mask)[0];
  delete[] rook_point;
  return score;
}

// TAG: to display result
inline void DisplayContours(
    const std::vector<std::vector<cv::Point>> &contours) {
  // 输出每个轮廓的点
  for (size_t i = 0; i < contours.size(); ++i) {
    std::cout << "Contour " << i << ": ";
    for (size_t j = 0; j < contours[i].size(); ++j) {
      std::cout << "(" << contours[i][j].x << "," << contours[i][j].y << ") ";
    }
    std::cout << std::endl;
  }
  return;
}

// NOTE: use op map
struct customized_param_t {
  int32_t width;
  int32_t height;
  int32_t pitch;
  float thresh;
};

uint32_t getInputCount(const char *op_name) { return 1; }
uint32_t getOutputCount(const char *op_name) { return 4; }

vsx::CustomOperatorCallback callback{
    getInputCount, nullptr, getOutputCount, nullptr, nullptr, nullptr, 0, 0};

inline void BoxesFromBitmap(const cv::Mat pred,
                            const std::vector<std::vector<cv::Point>> &contours,
                            std::vector<std::vector<std::vector<int>>> &boxes,
                            std::vector<float> &scores, float box_thresh,
                            float unclip_ratio, bool det_use_polygon_score) {
  const int min_size = 3;
  const int max_candidates = 1000;

  // TODO: convert this
  int width = pred.cols;
  int height = pred.rows;

  int num_contours =
      contours.size() >= max_candidates ? max_candidates : contours.size();
  for (int i = 0; i < num_contours; i++) {
    float ssid;
    if (contours[i].size() <= 2) continue;

    cv::RotatedRect box = cv::minAreaRect(contours[i]);
    auto array = GetMiniBoxes(box, ssid);

    auto box_for_unclip = array;
    // end get_mini_box

    if (ssid < min_size) {
      continue;
    }

    float score;
    if (det_use_polygon_score) {
      score = PolygonScoreAcc(contours[i], pred);
    } else {
      score = BoxScoreFast(array, pred);
    }
    // end box_score_fast
    if (score < box_thresh) continue;

    // start for unclip
    cv::RotatedRect points = Unclip(box_for_unclip, unclip_ratio);
    if (points.size.height < 1.001 && points.size.width < 1.001) continue;
    // end for unclip

    cv::RotatedRect clipbox = points;
    auto cliparray = GetMiniBoxes(clipbox, ssid);

    if (ssid < min_size + 2) continue;

    int dest_width = pred.cols;
    int dest_height = pred.rows;
    std::vector<std::vector<int>> intcliparray;

    for (int num_pt = 0; num_pt < 4; num_pt++) {
      std::vector<int> a{
          static_cast<int>(clamp(
              roundf(cliparray[num_pt][0] / float(width) * float(dest_width)),
              float(0), float(dest_width))),
          static_cast<int>(clamp(
              roundf(cliparray[num_pt][1] / float(height) * float(dest_height)),
              float(0), float(dest_height)))};
      intcliparray.push_back(a);
    }
    boxes.push_back(intcliparray);
    scores.push_back(score);

  }  // end for
  // return boxes;
}

inline void FilterTagDetRes(std::vector<std::vector<std::vector<int>>> &boxes,
                            float ratio_h, float ratio_w, int oriimg_h,
                            int oriimg_w) {
  std::vector<std::vector<std::vector<int>>> root_points;
  for (int n = 0; n < static_cast<int>(boxes.size()); n++) {
    boxes[n] = OrderPointsClockwise(boxes[n]);
    for (int m = 0; m < static_cast<int>(boxes[0].size()); m++) {
      boxes[n][m][0] /= ratio_w;
      boxes[n][m][1] /= ratio_h;

      boxes[n][m][0] =
          static_cast<int>(std::min(std::max(boxes[n][m][0], 0), oriimg_w - 1));
      boxes[n][m][1] =
          static_cast<int>(std::min(std::max(boxes[n][m][1], 0), oriimg_h - 1));
    }
  }
}

inline void DisplayBoxAndScores(
    std::vector<float> &scores,
    std::vector<std::vector<std::vector<int>>> &filter_boxes) {
  assert(scores.size() == filter_boxes.size());
  for (size_t k = 0; k < scores.size(); ++k) {
    std::cout << "index:" << k << ", score:" << scores[k] << ",bbox:" << "[";
    for (auto &vec : filter_boxes[k]) {
      std::cout << "[" << vec[0] << " " << vec[1] << "] ";
    }
    std::cout << "]\n";
  }
}

class DbnetPostProcesser {
 public:
  DbnetPostProcesser(const std::string &elf_path, uint32_t device_id = 0,
                     const std::string &op_name = "find_contours")
      : device_id_(device_id) {
    CHECK(vsx::SetDevice(device_id) == 0)
        << "Failed to set device id: " << device_id;
    find_contours_op_ =
        std::make_shared<vsx::CustomOperator>(op_name.c_str(), elf_path);
    find_contours_op_->SetCallback(callback);
  }

  DBresult Process(vsx::Tensor &fp16_tensor, std::vector<int> &srcimg_hw,
                   std::vector<float> &ratio_hw, double threshold,
                   bool det_db_use_dilate, float box_thresh, float unclip_ratio,
                   bool det_use_polygon_score,
                   const vsx::FindContourOpInfo &OpInfo, bool display) {
    auto contours =
        RunfindContoursOp(srcimg_hw, fp16_tensor, OpInfo.max_contour_num,
                          OpInfo.thresh, OpInfo.device_id);

    // NOTE: convert to fp32 host
    auto fp16_tensor_host = fp16_tensor.Clone();
    auto fp32_tensor = vsx::ConvertTensorFromFp16ToFp32(fp16_tensor_host);

    // Get output and post process
    float *pred = fp32_tensor.MutableData<float>();
    auto shape_out = fp32_tensor.Shape();

    auto height = shape_out[shape_out.ndim() - 2];
    auto width = shape_out[shape_out.ndim() - 1];

    cv::Mat pred_map(height, width, CV_32F, reinterpret_cast<float *>(pred));

    DBresult result;
    // TODO: not pass bit_map , just pass contours
    BoxesFromBitmap(pred_map, contours, result.boxes, result.scores, box_thresh,
                    unclip_ratio, det_use_polygon_score);
    FilterTagDetRes(result.boxes, ratio_hw[0], ratio_hw[1], srcimg_hw[0],
                    srcimg_hw[1]);

    if (display) DisplayBoxAndScores(result.scores, result.boxes);

    return result;
  }
  std::vector<std::vector<cv::Point>> RunfindContoursOp(
      std::vector<int> &srcimg_hw, const Tensor &in_tensor,
      uint32_t max_contour_num, float thresh = 0.3, uint32_t device = 0) {
    std::vector<Tensor> input{in_tensor};

    // NOTE: make sure this values 's live life
    auto shape = in_tensor.Shape();

    auto width = shape[shape.ndim() - 1];
    auto height = shape[shape.ndim() - 2];
    customized_param_t cfg;
    cfg.width = width;
    cfg.height = height;
    cfg.pitch = width;
    cfg.thresh = thresh;

    // out_tensor0: to save point array, assume that there are 10 point in a
    // countour.
    Tensor out_tensor0({4, height, width}, Context::VACC(device),
                       TypeFlag::kUint8);
    // out_tensor1: to save contour array.
    Tensor out_tensor1({1, 20 * max_contour_num}, Context::VACC(device),
                       TypeFlag::kUint8);
    // out_tensor1: to save contour number.
    Tensor out_tensor2({1}, Context::VACC(device), TypeFlag::kUint32);
    Tensor vdsp_buffer({4, height, width + 2}, Context::VACC(device),
                       TypeFlag::kUint8);

    std::vector<Tensor> output{out_tensor0, out_tensor1, out_tensor2,
                               vdsp_buffer};

    assert(in_tensor.GetContext().dev_type == vsx::Context::kVACC);

    // this struct contain 4 byte
    // #pragma pack(1)
    struct point {
      int16_t x;
      int16_t y;
    };

    // this struct contain 20 byte
    // #pragma pack(1)
    struct contour {
      // not used
      int32_t id;
      int32_t parent_id;
      int32_t contour_type_;
      // used
      int32_t points_idx;
      int32_t num_points;
    };

    find_contours_op_->RunSync(input, output, &cfg, sizeof(customized_param_t));

    auto out_tensor0_cpu = output[0].Clone();
    auto out_tensor1_cpu = output[1].Clone();
    auto out_tensor2_cpu = output[2].Clone();

    // 获取轨迹的个数
    uint32_t det_contour_num = *(out_tensor2_cpu.MutableData<uint32_t>());

    if (det_contour_num > max_contour_num) {
      LOG(ERROR)
          << "Deteced contour num is greater than configed max_contour_num";
      return {};
    }
    // 获取轨迹
    std::vector<std::vector<cv::Point>> contours;
    contours.reserve(det_contour_num);
    auto contour_ptr = out_tensor1_cpu.MutableData<contour>();
    auto point_ptr = out_tensor0_cpu.MutableData<point>();
    uint64_t point_id_st = 0;
    for (uint32_t contour_id = 0; contour_id < det_contour_num; ++contour_id) {
      contour tmp_contour = contour_ptr[contour_id];
      // TODO: show points
      // TODO: do memory boundary check
      std::vector<cv::Point> contour;
      int points_num = tmp_contour.num_points;
      // LOG(INFO) << "contour " << contour_id << " has " << points_num
      //           << " points";
      for (int point_id = 0; point_id < points_num; ++point_id) {
        auto point = point_ptr[point_id_st + point_id];
        // LOG(INFO) << "( " << point.x << " , " << point.y << ")";
        contour.emplace_back(point.x, point.y);
      }
      point_id_st += points_num;
      contours.emplace_back(std::move(contour));
    }

    return contours;
  }

 private:
  uint32_t device_id_ = 0;
  std::shared_ptr<vsx::CustomOperator> find_contours_op_;
};

}  // namespace vsx