
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include "common/text_cls.hpp"
#include "common/text_det.hpp"
#include "common/text_rec.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"
namespace vsx {
class OCR_e2e {
 public:
  OCR_e2e(const std::string& det_model, const std::string& det_config,
          const std::string& det_box_type, const std::string& det_elf_file,
          const std::string& cls_model, const std::string& cls_config,
          const std::vector<uint32_t>& cls_labels, float cls_thresh,
          const std::string& rec_model, const std::string& rec_config,
          const std::string& rec_label_file, float rec_drop_score,
          bool use_angle_cls, uint32_t batch_size = 1, uint32_t device_id = 0,
          const std::string& hw_config = "")
      : det_box_type_(det_box_type),
        use_angle_cls_(use_angle_cls),
        cls_thresh_(cls_thresh),
        rec_drop_score_(rec_drop_score),
        text_det_(det_model, det_config, det_elf_file, batch_size, device_id),
        text_cls_(cls_model, cls_config, cls_labels, batch_size, device_id,
                  hw_config),
        text_rec_(rec_model, rec_config, batch_size, device_id, rec_label_file,
                  hw_config) {
    device_id_ = device_id;
  }
  std::vector<std::tuple<std::vector<float>, float, std::string>> Process(
      cv::Mat& image, bool do_cls = true) {
    auto img_format = text_det_.GetFusionOpIimageFormat();
    vsx::Image vsx_img;
    vsx::MakeVsxImage(image, vsx_img, img_format);
    auto det_results = text_det_.Process(vsx_img);
    std::vector<std::tuple<std::vector<float>, float, std::string>> results;
    if (det_results.GetSize() == 0) {
      std::cout << "No object detected in image.\n";
    } else {
      std::vector<vsx::Image> vacc_crop_imgs;
      int obj_count = det_results.Shape()[0];
      std::vector<cv::Mat> crop_imgs(obj_count);
      const float* det_res_data = det_results.Data<float>();
      for (int i = 0; i < obj_count; i++) {
        // float score = det_res_data[i * 9 + 0];
        std::vector<cv::Point2f> src_points{
            cv::Point2f(det_res_data[i * 9 + 1], det_res_data[i * 9 + 2]),
            cv::Point2f(det_res_data[i * 9 + 3], det_res_data[i * 9 + 4]),
            cv::Point2f(det_res_data[i * 9 + 5], det_res_data[i * 9 + 6]),
            cv::Point2f(det_res_data[i * 9 + 7], det_res_data[i * 9 + 8])};
        if (det_box_type_ == "quad") {
          GetRotateCropImage(image, src_points, crop_imgs[i]);
        } else {
          GetMinareaRectCropImage(image, src_points, crop_imgs[i]);
        }
        vsx::Image vsx_cpu_image;
        vsx::MakeVsxImage(crop_imgs[i], vsx_cpu_image, vsx::RGB_PLANAR);
        vacc_crop_imgs.push_back(
            vsx_cpu_image.Clone(vsx::Context::VACC(device_id_)));
      }
      if (use_angle_cls_) {
        auto cls_result = text_cls_.Process(vacc_crop_imgs);
        for (size_t i = 0; i < cls_result.size(); i++) {
          const float* cls_data = cls_result[i].Data<float>();
          if (cls_data[1] > cls_data[0] && cls_data[1] > cls_thresh_) {
            cv::rotate(crop_imgs[i], crop_imgs[i], cv::ROTATE_180);
            vsx::Image vsx_image;
            vsx::MakeVsxImage(crop_imgs[i], vsx_image, vsx::BGR_INTERLEAVE);
            vacc_crop_imgs[i] = vsx_image.Clone(vsx::Context::VACC(device_id_));
          }
        }
      }
      auto rec_res = text_rec_.Process(vacc_crop_imgs);
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
  void GetRotateCropImage(const cv::Mat& image,
                          const std::vector<cv::Point2f>& points,
                          cv::Mat& dst_img) {
    // Calculate width and height of the cropped image
    float width1 = cv::norm(points[0] - points[1]);
    float width2 = cv::norm(points[2] - points[3]);
    float height1 = cv::norm(points[0] - points[3]);
    float height2 = cv::norm(points[1] - points[2]);

    int crop_w = std::max(width1, width2);
    int crop_h = std::max(height1, height2);

    std::vector<cv::Point2f> pts_std = {
        cv::Point2f(0, 0), cv::Point2f(crop_w, 0), cv::Point2f(crop_w, crop_h),
        cv::Point2f(0, crop_h)};

    cv::Mat M = cv::getPerspectiveTransform(points, pts_std);

    cv::warpPerspective(image, dst_img, M, cv::Size(crop_w, crop_h),
                        cv::INTER_CUBIC, cv::BORDER_REPLICATE);

    // Check if the aspect ratio requires rotation
    if (static_cast<float>(dst_img.rows) / dst_img.cols >= 1.5) {
      cv::rotate(dst_img, dst_img, cv::ROTATE_90_CLOCKWISE);
    }
  }

  void GetMinareaRectCropImage(const cv::Mat& image,
                               const std::vector<cv::Point2f>& points,
                               cv::Mat& dst_img) {
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

    GetRotateCropImage(image, box, dst_img);
  }

 private:
  std::string det_box_type_;
  bool use_angle_cls_;
  float cls_thresh_;
  float rec_drop_score_;
  vsx::TextDetector text_det_;
  vsx::TextClassifier text_cls_;
  vsx::TextRecognizer text_rec_;
  uint32_t device_id_;
};
}  // namespace vsx
