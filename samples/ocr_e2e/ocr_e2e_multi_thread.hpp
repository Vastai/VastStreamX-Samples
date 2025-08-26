
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#include "common/readerwritercircularbuffer.h"
#include "common/text_cls.hpp"
#include "common/text_det.hpp"
#include "common/text_rec.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

namespace vsx {

using moodycamel::BlockingReaderWriterCircularBuffer;
typedef std::tuple<cv::Mat, vsx::Tensor> DetPostInputType;
typedef std::tuple<vsx::Tensor, std::vector<cv::Mat>, std::vector<vsx::Image>>
    ClsInputType;
typedef std::tuple<vsx::Tensor, std::vector<vsx::Image>> RecInputType;
typedef std::tuple<std::vector<float>, float, std::string> TextObject;

enum StopFlag {
  INIT_VALUE = 0,
  INPUT_STOP,
  DET_STOP,
  DET_POST_STOP,
  CLS_STOP,
  REC_STOP
};

class OCR_e2e_Async {
 public:
  OCR_e2e_Async(const std::string& det_model, const std::string& det_config,
                const std::string& det_box_type,
                const std::string& det_elf_file, const std::string& cls_model,
                const std::string& cls_config,
                const std::vector<uint32_t>& cls_labels, float cls_thresh,
                const std::string& rec_model, const std::string& rec_config,
                const std::string& rec_label_file, float rec_drop_score,
                bool use_angle_cls, uint32_t batch_size = 1,
                uint32_t device_id = 0, const std::string& hw_config = "",
                size_t queue_size = 5)
      : det_box_type_(det_box_type),
        use_angle_cls_(use_angle_cls),
        cls_thresh_(cls_thresh),
        rec_drop_score_(rec_drop_score),
        text_det_(det_model, det_config, det_elf_file, batch_size, device_id),
        text_cls_(cls_model, cls_config, cls_labels, batch_size, device_id,
                  hw_config),
        text_rec_(rec_model, rec_config, batch_size, device_id, rec_label_file,
                  hw_config),
        det_inputs_(queue_size),
        det_post_inputs_(queue_size),
        cls_inputs_(queue_size),
        rec_inputs_(queue_size),
        rec_outputs_(10),
        stop_flag_(0) {
    vsx::GetDevice(device_id_);
    det_input_format_ = text_det_.GetFusionOpIimageFormat();

    det_ticks_.reserve(1024);
    det_tocks_.reserve(1024);
    cls_ticks_.reserve(1024);
    cls_tocks_.reserve(1024);
    rec_ticks_.reserve(1024);
    rec_tocks_.reserve(1024);

    det_thread_ = std::thread(&OCR_e2e_Async::DetectThread, this);
    det_post_thread_ = std::thread(&OCR_e2e_Async::DetectPostThread, this);
    cls_thread_ = std::thread(&OCR_e2e_Async::ClassifyThread, this);
    rec_thread_ = std::thread(&OCR_e2e_Async::RecognizeThread, this);
  }
  ~OCR_e2e_Async() {
    if (det_thread_.joinable()) det_thread_.join();
    if (det_post_thread_.joinable()) det_post_thread_.join();
    if (cls_thread_.joinable()) cls_thread_.join();
    if (rec_thread_.joinable()) rec_thread_.join();
  }
  int ProcessAsync(cv::Mat& image, bool do_cls = true) {
    det_inputs_.wait_enqueue(image);
    return 0;
  }
  bool GetOutput(std::vector<TextObject>& objs) {
    while (true) {
      if (stop_flag_ != static_cast<int>(StopFlag::REC_STOP)) {
        if (rec_outputs_.wait_dequeue_timed(objs, 10 * 1000)) {
          return true;
        }
      } else {
        return false;
      }
    }
  }
  void Stop() { stop_flag_ = static_cast<int>(StopFlag::INPUT_STOP); }

 private:
  void DetectThread() {
    while (true) {
      cv::Mat cv_mat;
      if (det_inputs_.wait_dequeue_timed(cv_mat, 10 * 1000)) {
        det_ticks_.push_back(std::chrono::high_resolution_clock::now());
        vsx::Image vsx_img;
        vsx::MakeVsxImage(cv_mat, vsx_img, det_input_format_);
        auto det_results = text_det_.Process(vsx_img);
        auto det_post_input = std::make_tuple<cv::Mat, vsx::Tensor>(
            std::move(cv_mat), std::move(det_results));
        det_post_inputs_.wait_enqueue(std::move(det_post_input));
        det_tocks_.push_back(std::chrono::high_resolution_clock::now());
      } else if (stop_flag_ == static_cast<int>(StopFlag::INPUT_STOP)) {
        stop_flag_ = static_cast<int>(StopFlag::DET_STOP);
        break;
      }
    }
  }
  void DetectPostThread() {
    int queue_size = 10;
    BlockingReaderWriterCircularBuffer<std::future<ClsInputType>> queue_futs(
        queue_size);
    bool stopped = false;
    std::atomic<int> left(0);
    std::thread cunsume_thread([&] {
      while (!stopped || left > 0) {
        std::future<ClsInputType> fut;
        if (queue_futs.wait_dequeue_timed(fut, 1000)) {
          auto result = fut.get();
          cls_inputs_.wait_enqueue(std::move(result));
          --left;
        }
      }
    });

    while (true) {
      DetPostInputType post_input;
      if (det_post_inputs_.wait_dequeue_timed(post_input, 10 * 1000)) {
        auto fut = std::async(
            std::launch::async,
            [&](DetPostInputType&& post_in) {
              auto cv_mat = std::get<0>(post_in);
              auto det_results = std::get<1>(post_in);
              std::vector<vsx::Image> vsx_crop_imgs;
              int obj_count = det_results.Shape()[0];
              std::vector<cv::Mat> crop_imgs(obj_count);
              const float* det_res_data = det_results.Data<float>();
              for (int i = 0; i < obj_count; i++) {
                std::vector<cv::Point2f> src_points{
                    cv::Point2f(det_res_data[i * 9 + 1],
                                det_res_data[i * 9 + 2]),
                    cv::Point2f(det_res_data[i * 9 + 3],
                                det_res_data[i * 9 + 4]),
                    cv::Point2f(det_res_data[i * 9 + 5],
                                det_res_data[i * 9 + 6]),
                    cv::Point2f(det_res_data[i * 9 + 7],
                                det_res_data[i * 9 + 8])};
                if (det_box_type_ == "quad") {
                  GetRotateCropImage(cv_mat, src_points, crop_imgs[i]);
                } else {
                  GetMinareaRectCropImage(cv_mat, src_points, crop_imgs[i]);
                }
                vsx::Image vsx_cpu_image;
                vsx::MakeVsxImage(crop_imgs[i], vsx_cpu_image, vsx::RGB_PLANAR);
                vsx_crop_imgs.push_back(vsx_cpu_image);
              }
              auto result =
                  std::make_tuple(std::move(det_results), std::move(crop_imgs),
                                  std::move(vsx_crop_imgs));
              return result;
            },
            std::move(post_input));
        ++left;
        queue_futs.wait_enqueue(std::move(fut));
      } else if (stop_flag_ == static_cast<int>(StopFlag::DET_STOP)) {
        stopped = true;
        cunsume_thread.join();
        stop_flag_ = static_cast<int>(StopFlag::DET_POST_STOP);
        break;
      }
    }
  }
  void ClassifyThread() {
    while (true) {
      ClsInputType cls_input;
      if (cls_inputs_.wait_dequeue_timed(cls_input, 10 * 1000)) {
        cls_ticks_.push_back(std::chrono::high_resolution_clock::now());
        auto det_results = std::get<0>(cls_input);
        auto crop_imgs = std::get<1>(cls_input);
        auto vsx_crop_imgs = std::get<2>(cls_input);
        int obj_count = det_results.Shape()[0];

        // run cls
        if (use_angle_cls_ && obj_count) {
          auto cls_result = text_cls_.Process(vsx_crop_imgs);
          for (size_t i = 0; i < cls_result.size(); i++) {
            const float* cls_data = cls_result[i].Data<float>();
            if (cls_data[1] > cls_data[0] && cls_data[1] > cls_thresh_) {
              cv::rotate(crop_imgs[i], crop_imgs[i], cv::ROTATE_180);
              vsx::Image vsx_image;
              vsx::MakeVsxImage(crop_imgs[i], vsx_image, vsx::BGR_INTERLEAVE);
              vsx_crop_imgs[i] = vsx_image;
            }
          }
        }
        // set rec input
        auto rec_input =
            std::make_tuple(std::move(det_results), std::move(vsx_crop_imgs));
        rec_inputs_.wait_enqueue(rec_input);
        cls_tocks_.push_back(std::chrono::high_resolution_clock::now());
      } else if (stop_flag_ == static_cast<int>(StopFlag::DET_POST_STOP)) {
        stop_flag_ = static_cast<int>(StopFlag::CLS_STOP);
        break;
      }
    }
  }
  void RecognizeThread() {
    while (true) {
      RecInputType rec_input;
      if (rec_inputs_.wait_dequeue_timed(rec_input, 10 * 1000)) {
        rec_ticks_.push_back(std::chrono::high_resolution_clock::now());
        vsx::Tensor det_result = std::get<0>(rec_input);
        std::vector<vsx::Image> vsx_crop_imgs = std::get<1>(rec_input);
        std::vector<TextObject> results;
        const float* det_res_data = det_result.Data<float>();
        if (vsx_crop_imgs.size() > 0) {
          auto rec_res = text_rec_.Process(vsx_crop_imgs);
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
        // set output
        rec_outputs_.wait_enqueue(std::move(results));
        rec_tocks_.push_back(std::chrono::high_resolution_clock::now());
      } else if (stop_flag_ == static_cast<int>(StopFlag::CLS_STOP)) {
        stop_flag_ = static_cast<int>(StopFlag::REC_STOP);
        break;
      }
    }
  }

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

 public:
  std::vector<time_point> det_ticks_;
  std::vector<time_point> det_tocks_;

  std::vector<time_point> cls_ticks_;
  std::vector<time_point> cls_tocks_;

  std::vector<time_point> rec_ticks_;
  std::vector<time_point> rec_tocks_;

 private:
  std::string det_box_type_;
  bool use_angle_cls_;
  float cls_thresh_;
  float rec_drop_score_;
  vsx::TextDetector text_det_;
  vsx::TextClassifier text_cls_;
  vsx::TextRecognizer text_rec_;
  uint32_t device_id_;

  BlockingReaderWriterCircularBuffer<cv::Mat> det_inputs_;
  BlockingReaderWriterCircularBuffer<DetPostInputType> det_post_inputs_;
  BlockingReaderWriterCircularBuffer<ClsInputType> cls_inputs_;
  BlockingReaderWriterCircularBuffer<RecInputType> rec_inputs_;
  BlockingReaderWriterCircularBuffer<std::vector<TextObject>> rec_outputs_;

  std::thread det_thread_, det_post_thread_, cls_thread_, rec_thread_;
  std::atomic<int> stop_flag_;
  vsx::ImageFormat det_input_format_;
};
}  // namespace vsx