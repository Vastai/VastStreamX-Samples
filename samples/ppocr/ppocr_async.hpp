
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

#include "common/doc_img_orient_cls_async.hpp"
#include "common/readerwritercircularbuffer.h"
#include "common/text_cls.hpp"
#include "common/text_det_async.hpp"
#include "common/text_rec_async.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"
#include "vdsp_ops/rotate_op.hpp"
#include "vdsp_ops/warp_perspective_op.hpp"
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

namespace vsx {

using moodycamel::BlockingReaderWriterCircularBuffer;
typedef std::tuple<vsx::Image, vsx::Tensor> DetPostInputType;
typedef std::tuple<vsx::Tensor, std::vector<vsx::Image>> TextOriClsInputType;
typedef std::tuple<vsx::Tensor, std::vector<vsx::Image>> RecInputType;
typedef std::tuple<std::vector<float>, float, std::string> TextObject;

enum StopFlag {
  INIT_VALUE = 0,
  INPUT_STOP,
  DOC_ORI_STOP,
  DET_STOP,
  DET_POST_STOP,
  CLS_STOP,
  REC_STOP
};

class PPOCR_Async {
 public:
  PPOCR_Async(
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
      const std::string& hw_config = "", size_t queue_size = 1)
      : doc_ori_labels_(doc_ori_labels),
        image_rotated_angles_(queue_size + 10),
        doc_ori_inputs_(queue_size),
        det_inputs_(queue_size),
        det_post_inputs_(queue_size),
        text_ori_inputs_(queue_size),
        rec_inputs_(queue_size),
        rec_outputs_(queue_size + 10) {
    det_box_type_ = det_box_type;
    use_doc_ori_cls_ = use_doc_ori_cls;
    use_text_ori_cls_ = use_text_ori_cls;
    text_ori_thresh_ = text_ori_thresh;
    rec_drop_score_ = rec_drop_score;
    queue_size_ = queue_size;

    device_id_ = device_id;
    vsx::SetDevice(device_id_);

    if (use_doc_ori_cls_) {
      doc_ori_cls_ = std::make_shared<vsx::DocImgOrientClassifierAsync>(
          doc_roi_model, doc_roi_config, batch_size, device_id);
    }

    text_det_ = std::make_shared<vsx::TextDetectorAsync>(
        det_model, det_config, det_elf_file, batch_size, device_id);
    text_det_->SetBoxThreshold(det_box_thresh);

    if (use_text_ori_cls_) {
      text_ori_cls_ = std::make_shared<vsx::TextClassifier>(
          text_ori_model, text_ori_config, batch_size, device_id, hw_config);
    }
    text_rec_ = std::make_shared<vsx::TextRecognizerAsync>(
        rec_model, rec_config, batch_size, device_id, rec_label_file,
        hw_config);

    rotate_op_ = std::make_shared<vsx::RotateOp>(rotate_op_file, device_id);
    warp_perspective_op_ = std::make_shared<vsx::WarpPerspectiveOp>(
        warp_perspective_op_file, device_id);

    stop_flag_ = static_cast<int>(StopFlag::INIT_VALUE);

    doc_ori_ticks_.reserve(1024);
    doc_ori_tocks_.reserve(1024);
    det_ticks_.reserve(1024);
    det_tocks_.reserve(1024);
    text_ori_ticks_.reserve(1024);
    text_ori_tocks_.reserve(1024);
    rec_ticks_.reserve(1024);
    rec_tocks_.reserve(1024);

    doc_ori_cls_thread_ =
        std::thread(&PPOCR_Async::DocImgOrientClsThread, this);
    det_thread_ = std::thread(&PPOCR_Async::DetectThread, this);
    det_post_thread_ = std::thread(&PPOCR_Async::DetectPostThread, this);
    text_ori_cls_thread_ =
        std::thread(&PPOCR_Async::TextLineOrientClsThread, this);
    rec_thread_ = std::thread(&PPOCR_Async::RecognizeThread, this);
  }
  ~PPOCR_Async() {
    if (doc_ori_cls_thread_.joinable()) doc_ori_cls_thread_.join();
    if (det_thread_.joinable()) det_thread_.join();
    if (det_post_thread_.joinable()) det_post_thread_.join();
    if (text_ori_cls_thread_.joinable()) text_ori_cls_thread_.join();
    if (rec_thread_.joinable()) rec_thread_.join();
  }
  int ProcessAsync(vsx::Image& vsx_image) {
    doc_ori_inputs_.wait_enqueue(vsx_image);
    return 0;
  }
  bool GetOutput(std::vector<TextObject>& objs, int& rotated_angle) {
    while (true) {
      if (stop_flag_ != static_cast<int>(StopFlag::REC_STOP)) {
        if (rec_outputs_.wait_dequeue_timed(objs, 10 * 1000)) {
          image_rotated_angles_.wait_dequeue(rotated_angle);
          return true;
        }
      } else {
        return false;
      }
    }
  }
  void Stop() { stop_flag_ = static_cast<int>(StopFlag::INPUT_STOP); }

 private:
  void DocImgOrientClsThread() {
    vsx::SetDevice(device_id_);
    BlockingReaderWriterCircularBuffer<vsx::Image> input_images(queue_size_);
    std::thread get_output_thread;
    if (use_doc_ori_cls_) {
      get_output_thread = std::thread([&]() {
        vsx::SetDevice(device_id_);
        while (true) {
          std::vector<vsx::Tensor> outs;
          if (doc_ori_cls_->GetOutput(outs)) {
            vsx::Image vsx_image;
            input_images.wait_dequeue(vsx_image);
            int index;
            float score;
            doc_ori_cls_->PostProcess(outs[0], index, score);
            int angle = get_angle(index);
            int origin_rotated_angle = 0;
            if (angle != 0) {
              origin_rotated_angle = 360 - angle;
              vsx_image = rotate_op_->Process(vsx_image, origin_rotated_angle);
            }
            image_rotated_angles_.wait_enqueue(origin_rotated_angle);
            det_inputs_.wait_enqueue(std::move(vsx_image));
            doc_ori_tocks_.push_back(std::chrono::high_resolution_clock::now());
          } else {
            break;
          }
        }
      });
    }
    while (true) {
      vsx::Image vsx_img;
      if (doc_ori_inputs_.wait_dequeue_timed(vsx_img, 10 * 1000)) {
        if (use_doc_ori_cls_) {
          doc_ori_ticks_.push_back(std::chrono::high_resolution_clock::now());
          input_images.wait_enqueue(vsx_img);
          doc_ori_cls_->ProcessAsync(vsx_img);
        } else {
          image_rotated_angles_.wait_enqueue(0);
          det_inputs_.wait_enqueue(std::move(vsx_img));
        }
      } else if (stop_flag_ == static_cast<int>(StopFlag::INPUT_STOP)) {
        if (use_doc_ori_cls_) {
          doc_ori_cls_->CloseInput();
          doc_ori_cls_->WaitUntilDone();
          get_output_thread.join();
        }
        stop_flag_ = static_cast<int>(StopFlag::DOC_ORI_STOP);
        break;
      }
    }
  }

  void DetectThread() {
    vsx::SetDevice(device_id_);

    BlockingReaderWriterCircularBuffer<vsx::Image> input_images(queue_size_);
    std::thread output_thread = std::thread([&]() {
      vsx::SetDevice(device_id_);
      while (true) {
        std::vector<vsx::Tensor> det_results;
        if (text_det_->GetOutput(det_results)) {
          vsx::Image vsx_image;
          input_images.wait_dequeue(vsx_image);
          auto det_post_input =
              std::make_tuple(std::move(vsx_image), std::move(det_results[0]));
          det_post_inputs_.wait_enqueue(std::move(det_post_input));
          det_tocks_.push_back(std::chrono::high_resolution_clock::now());
        } else {
          break;
        }
      }
    });

    while (true) {
      vsx::Image vsx_img;
      if (det_inputs_.wait_dequeue_timed(vsx_img, 10 * 1000)) {
        det_ticks_.push_back(std::chrono::high_resolution_clock::now());
        text_det_->ProcessAsync(vsx_img);
        input_images.wait_enqueue(vsx_img);
      } else if (stop_flag_ == static_cast<int>(StopFlag::DOC_ORI_STOP)) {
        text_det_->CloseInput();
        text_det_->WaitUntilDone();
        output_thread.join();
        stop_flag_ = static_cast<int>(StopFlag::DET_STOP);
        break;
      }
    }
  }
  void DetectPostThread() {
    vsx::SetDevice(device_id_);
    BlockingReaderWriterCircularBuffer<std::future<TextOriClsInputType>>
        queue_futs(queue_size_ + 10);
    bool stopped = false;
    std::atomic<int> left(0);
    std::thread cunsume_thread([&] {
      vsx::SetDevice(device_id_);
      while (!stopped || left > 0) {
        std::future<TextOriClsInputType> fut;
        if (queue_futs.wait_dequeue_timed(fut, 1000)) {
          auto result = fut.get();
          text_ori_inputs_.wait_enqueue(std::move(result));
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
              auto vsx_image = std::get<0>(post_in);
              auto det_results = std::get<1>(post_in);
              int obj_count = det_results.Shape()[0];
              const float* det_res_data = det_results.Data<float>();
              std::vector<vsx::Image> crop_images;
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
                  auto crop_image = GetRotateCropImage(vsx_image, src_points);
                  if (!crop_image.IsEmpty()) {
                    crop_images.push_back(crop_image);
                  }
                } else {
                  auto crop_image =
                      GetMinareaRectCropImage(vsx_image, src_points);
                  if (!crop_image.IsEmpty()) {
                    crop_images.push_back(crop_image);
                  }
                }
              }
              auto result = std::make_tuple(std::move(det_results),
                                            std::move(crop_images));
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

  void TextLineOrientClsThread() {
    vsx::SetDevice(device_id_);
    while (true) {
      TextOriClsInputType text_ori_input;
      if (text_ori_inputs_.wait_dequeue_timed(text_ori_input, 10 * 1000)) {
        text_ori_ticks_.push_back(std::chrono::high_resolution_clock::now());
        auto det_results = std::get<0>(text_ori_input);
        auto crop_images = std::get<1>(text_ori_input);
        int obj_count = det_results.Shape()[0];
        // run cls
        if (use_text_ori_cls_ && obj_count) {
          auto cls_result = text_ori_cls_->Process(crop_images);
          for (size_t i = 0; i < cls_result.size(); i++) {
            const float* cls_data = cls_result[i].Data<float>();
            if (cls_data[1] > cls_data[0] && cls_data[1] > text_ori_thresh_) {
              crop_images[i] = rotate_op_->Process(crop_images[i], 180);
            }
          }
        }
        // set rec input
        auto rec_input =
            std::make_tuple(std::move(det_results), std::move(crop_images));
        rec_inputs_.wait_enqueue(rec_input);
        text_ori_tocks_.push_back(std::chrono::high_resolution_clock::now());
      } else if (stop_flag_ == static_cast<int>(StopFlag::DET_POST_STOP)) {
        stop_flag_ = static_cast<int>(StopFlag::CLS_STOP);
        break;
      }
    }
  }
  void RecognizeThread() {
    vsx::SetDevice(device_id_);
    BlockingReaderWriterCircularBuffer<vsx::Tensor> det_results(50);
    BlockingReaderWriterCircularBuffer<bool> infer_flags(50);

    std::thread output_thread = std::thread([&]() {
      vsx::SetDevice(device_id_);
      while (true) {
        bool flag;
        infer_flags.wait_dequeue(flag);
        std::vector<TextObject> results;
        if (!flag) {
          rec_outputs_.wait_enqueue(std::move(results));
          rec_tocks_.push_back(std::chrono::high_resolution_clock::now());
        } else {
          std::vector<vsx::Tensor> rec_res;
          if (text_rec_->GetOutput(rec_res)) {
            vsx::Tensor det_result;
            det_results.wait_dequeue(det_result);
            const float* det_res_data = det_result.Data<float>();
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
            // set output
            rec_outputs_.wait_enqueue(std::move(results));
            rec_tocks_.push_back(std::chrono::high_resolution_clock::now());
          } else {
            break;
          }
        }
      }
    });
    while (true) {
      RecInputType rec_input;
      if (rec_inputs_.wait_dequeue_timed(rec_input, 10 * 1000)) {
        rec_ticks_.push_back(std::chrono::high_resolution_clock::now());
        vsx::Tensor det_result = std::get<0>(rec_input);
        auto crop_imgs = std::get<1>(rec_input);
        std::vector<TextObject> results;
        if (crop_imgs.size() > 0) {
          det_results.wait_enqueue(std::move(det_result));
          infer_flags.wait_enqueue(true);
          text_rec_->ProcessAsync(crop_imgs);
        } else {
          infer_flags.wait_enqueue(false);
        }
      } else if (stop_flag_ == static_cast<int>(StopFlag::CLS_STOP)) {
        infer_flags.wait_enqueue(true);
        text_rec_->CloseInput();
        output_thread.join();
        text_rec_->WaitUntilDone();
        stop_flag_ = static_cast<int>(StopFlag::REC_STOP);
        break;
      }
    }
  }
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

 public:
  std::vector<time_point> doc_ori_ticks_;
  std::vector<time_point> doc_ori_tocks_;

  std::vector<time_point> det_ticks_;
  std::vector<time_point> det_tocks_;

  std::vector<time_point> text_ori_ticks_;
  std::vector<time_point> text_ori_tocks_;

  std::vector<time_point> rec_ticks_;
  std::vector<time_point> rec_tocks_;

 private:
  std::string det_box_type_;
  bool use_doc_ori_cls_;
  bool use_text_ori_cls_;
  float text_ori_thresh_;
  float rec_drop_score_;
  std::vector<std::vector<int>> doc_ori_labels_;

  std::shared_ptr<vsx::DocImgOrientClassifierAsync> doc_ori_cls_;
  std::shared_ptr<vsx::TextDetectorAsync> text_det_;
  std::shared_ptr<vsx::TextClassifier> text_ori_cls_;
  std::shared_ptr<vsx::TextRecognizerAsync> text_rec_;

  std::shared_ptr<vsx::RotateOp> rotate_op_;
  std::shared_ptr<vsx::WarpPerspectiveOp> warp_perspective_op_;

  uint32_t device_id_;

  BlockingReaderWriterCircularBuffer<int> image_rotated_angles_;
  BlockingReaderWriterCircularBuffer<vsx::Image> doc_ori_inputs_;
  BlockingReaderWriterCircularBuffer<vsx::Image> det_inputs_;
  BlockingReaderWriterCircularBuffer<DetPostInputType> det_post_inputs_;
  BlockingReaderWriterCircularBuffer<TextOriClsInputType> text_ori_inputs_;
  BlockingReaderWriterCircularBuffer<RecInputType> rec_inputs_;
  BlockingReaderWriterCircularBuffer<std::vector<TextObject>> rec_outputs_;

  std::thread doc_ori_cls_thread_, det_thread_, det_post_thread_,
      text_ori_cls_thread_, rec_thread_;
  std::atomic<int> stop_flag_;

  size_t queue_size_;
};
}  // namespace vsx