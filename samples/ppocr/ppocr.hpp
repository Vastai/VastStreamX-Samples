
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include "common/doc_img_orient_cls.hpp"
#include "common/text_cls.hpp"
#include "common/text_det.hpp"
#include "common/text_rec.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"
#include "vdsp_ops/rotate_op.hpp"
#include "vdsp_ops/warp_perspective_op.hpp"

namespace vsx {
class PPOCR {
 public:
  PPOCR(
      // document image orientation classify
      const std::string& doc_roi_model, const std::string& doc_roi_config,
      const std::vector<std::vector<int>>& doc_ori_labels, bool use_doc_ori_cls,
      // text detection
      const std::string& det_model, const std::string& det_config,
      const std::string& det_box_type, const std::string& det_elf_file,
      float det_box_thresh,
      // textline orientation classify
      const std::string& text_ori_model, const std::string& text_ori_config,
      float text_ori_thresh, bool use_text_ori_cls,
      // text recognition
      const std::string& rec_model, const std::string& rec_config,
      const std::string& rec_label_file, float rec_drop_score,
      // vdsp_op
      const std::string& rotate_op_file,
      const std::string& warp_perspective_op_file,
      // common
      uint32_t batch_size = 1, uint32_t device_id = 0,
      const std::string& hw_config = "") {
    use_doc_ori_cls_ = use_doc_ori_cls;
    doc_ori_labels_ = doc_ori_labels;
    det_box_type_ = det_box_type;
    use_text_ori_cls_ = use_text_ori_cls;
    text_ori_thresh_ = text_ori_thresh;
    rec_drop_score_ = rec_drop_score;
    device_id_ = device_id;
    vsx::SetDevice(device_id_);

    if (use_doc_ori_cls) {
      doc_ori_cls_ = std::make_shared<vsx::DocImgOrientClassifier>(
          doc_roi_model, doc_roi_config, batch_size, device_id);
    }

    text_det_ = std::make_shared<vsx::TextDetector>(
        det_model, det_config, det_elf_file, batch_size, device_id);
    text_det_->SetBoxThreshold(det_box_thresh);

    text_rec_ = std::make_shared<vsx::TextRecognizer>(
        rec_model, rec_config, batch_size, device_id, rec_label_file,
        hw_config);

    if (use_text_ori_cls_) {
      text_ori_cls_ = std::make_shared<vsx::TextClassifier>(
          text_ori_model, text_ori_config, batch_size, device_id, hw_config);
    }

    rotate_op_ = std::make_shared<vsx::RotateOp>(rotate_op_file, device_id);
    warp_perspective_op_ = std::make_shared<vsx::WarpPerspectiveOp>(
        warp_perspective_op_file, device_id);
  }

  // 当前 Process仅支持rgb_planar格式输入
  std::vector<std::tuple<std::vector<float>, float, std::string>> Process(
      vsx::Image& origin_rgb_planar, int& return_rotate_angle,
      bool do_doc_ori_cls = true, bool do_textline_ori_cls = true) {
    auto vsx_image = origin_rgb_planar;
    return_rotate_angle = 0;
    if (do_doc_ori_cls && use_doc_ori_cls_) {
      auto tensor = doc_ori_cls_->Process(vsx_image);
      int index;
      float score;
      doc_ori_cls_->PostProcess(tensor, index, score);
      int angle = get_angle(index);
      if (angle != 0) {
        vsx_image = rotate_op_->Process(vsx_image, 360 - angle);
        return_rotate_angle = 360 - angle;
      }
    }
    // text_detection
    auto det_results = text_det_->Process(vsx_image);
    // parse text detection result
    std::vector<std::tuple<std::vector<float>, float, std::string>> results;
    std::vector<vsx::Image> crop_images;
    if (det_results.GetSize() == 0) {
      std::cout << "No text detected in image.\n";
    } else {
      int obj_count = det_results.Shape()[0];
      const float* det_res_data = det_results.Data<float>();
      for (int i = 0; i < obj_count; i++) {
        // float score = det_res_data[i * 9 + 0];
        std::vector<cv::Point2f> src_points{
            cv::Point2f(det_res_data[i * 9 + 1], det_res_data[i * 9 + 2]),
            cv::Point2f(det_res_data[i * 9 + 3], det_res_data[i * 9 + 4]),
            cv::Point2f(det_res_data[i * 9 + 5], det_res_data[i * 9 + 6]),
            cv::Point2f(det_res_data[i * 9 + 7], det_res_data[i * 9 + 8])};
        if (det_box_type_ == "quad") {
          auto crop_image = GetRotateCropImage(vsx_image, src_points);
          if (!crop_image.IsEmpty()) {
            crop_images.push_back(crop_image);
          }
        } else {
          auto crop_image = GetMinareaRectCropImage(vsx_image, src_points);
          if (!crop_image.IsEmpty()) {
            crop_images.push_back(crop_image);
          }
        }
      }
      if (use_text_ori_cls_) {
        auto cls_result = text_ori_cls_->Process(crop_images);
        for (size_t i = 0; i < cls_result.size(); i++) {
          const float* cls_data = cls_result[i].Data<float>();
          if (cls_data[1] > cls_data[0] && cls_data[1] > text_ori_thresh_) {
            crop_images[i] = rotate_op_->Process(crop_images[i], 180);
          }
        }
      }

      // text recognition
      auto rec_res = text_rec_->Process(crop_images);
      for (size_t i = 0; i < rec_res.size(); i++) {
        float score = vsx::GetScoreFromTensor(rec_res[i]);
        if (score >= rec_drop_score_) {
          std::vector<float> coor{
              det_res_data[i * 9 + 1], det_res_data[i * 9 + 2],
              det_res_data[i * 9 + 3], det_res_data[i * 9 + 4],
              det_res_data[i * 9 + 5], det_res_data[i * 9 + 6],
              det_res_data[i * 9 + 7], det_res_data[i * 9 + 8]};
          results.emplace_back(std::make_tuple(
              coor, score, vsx::GetStringFromTensor(rec_res[i])));
        }
      }
    }
    return results;
  }

 private:
  int get_angle(int index) {
    for (auto& label : doc_ori_labels_) {
      if (label[0] == index) {
        return label[1];
      }
    }
    std::cerr << "Error: Cann't find index: " << index << " in label file"
              << std::endl;
    return -10000;
  }
  vsx::Image GetRotateCropImage(const vsx::Image& vsx_image,
                                const std::vector<cv::Point2f>& points) {
    // Calculate width and height of the cropped image
    float width1 = cv::norm(points[0] - points[1]);
    float width2 = cv::norm(points[2] - points[3]);
    float height1 = cv::norm(points[0] - points[3]);
    float height2 = cv::norm(points[1] - points[2]);

    int crop_w = std::max(width1, width2);
    int crop_h = std::max(height1, height2);
    if (crop_h < 5 || crop_w < 5) {
      return vsx::Image();
    }

    std::vector<cv::Point2f> pts_std = {
        cv::Point2f(0, 0), cv::Point2f(crop_w, 0), cv::Point2f(crop_w, crop_h),
        cv::Point2f(0, crop_h)};

    cv::Mat M = cv::getPerspectiveTransform(points, pts_std);

    std::vector<double> matrix;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        matrix.push_back(M.at<double>(i, j));
      }
    }

    auto crop_image =
        warp_perspective_op_->Process(vsx_image, matrix, crop_w, crop_h);

    // Check if the aspect ratio requires rotation
    if (static_cast<float>(crop_h) / crop_w >= 1.5) {
      crop_image = rotate_op_->Process(crop_image, 90);
    }
    return crop_image;
  }

  vsx::Image GetMinareaRectCropImage(const vsx::Image& vsx_image,
                                     const std::vector<cv::Point2f>& points) {
    cv::RotatedRect bounding_box = cv::minAreaRect(points);

    std::vector<cv::Point2f> box_points;
    cv::boxPoints(bounding_box,
                  box_points);  // Get the four corners of the rotated box

    // Sort points by x-coordinate to maintain order
    std::sort(
        box_points.begin(), box_points.end(),
        [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });

    int index_a = 0, index_b = 1, index_c = 2, index_d = 3;

    // Re-order points based on y-coordinate to get correct rectangle corners
    if (box_points[1].y > box_points[0].y) {
      index_a = 0;
      index_d = 1;
    } else {
      index_a = 1;
      index_d = 0;
    }
    if (box_points[3].y > box_points[2].y) {
      index_b = 2;
      index_c = 3;
    } else {
      index_b = 3;
      index_c = 2;
    }

    std::vector<cv::Point2f> box = {box_points[index_a], box_points[index_b],
                                    box_points[index_c], box_points[index_d]};

    return GetRotateCropImage(vsx_image, box);
  }

 private:
  std::string det_box_type_;
  bool use_doc_ori_cls_;
  bool use_text_ori_cls_;
  float text_ori_thresh_;
  float rec_drop_score_;
  std::vector<std::vector<int>> doc_ori_labels_;

  std::shared_ptr<vsx::DocImgOrientClassifier> doc_ori_cls_;
  std::shared_ptr<vsx::TextDetector> text_det_;
  std::shared_ptr<vsx::TextClassifier> text_ori_cls_;
  std::shared_ptr<vsx::TextRecognizer> text_rec_;

  std::shared_ptr<vsx::RotateOp> rotate_op_;
  std::shared_ptr<vsx::WarpPerspectiveOp> warp_perspective_op_;

  uint32_t device_id_;
};
}  // namespace vsx
