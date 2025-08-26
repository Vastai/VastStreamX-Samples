
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <chrono>
#include <fstream>
#include <thread>

#include "common/cmdline.hpp"
#include "common/detector.hpp"
#include "common/file_system.hpp"
#include "common/utils.hpp"
#include "libtorch_postprocess.h"
#include "model_cv2.hpp"
#include "opencv2/opencv.hpp"
#include "vaststreamx/media/video_capture.h"

const std::vector<int32_t> COCO_INDEX{
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

const std::vector<cv::Scalar> KPS_COLORS = {
    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},
    {0, 255, 0},    {255, 128, 0},  {255, 128, 0},  {255, 128, 0},
    {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {51, 153, 255},
    {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255},
    {51, 153, 255}};

const std::vector<std::vector<unsigned int>> SKELETON = {
    {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13},
    {6, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},  {2, 3},  {1, 2},
    {1, 3},   {2, 4},   {3, 5},   {4, 6},   {5, 7}};

const std::vector<cv::Scalar> LIMB_COLORS = {
    {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255},
    {255, 51, 255}, {255, 51, 255}, {255, 51, 255}, {255, 128, 0},
    {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},
    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},
    {0, 255, 0},    {0, 255, 0},    {0, 255, 0}};

void DrawPose(cv::Mat &im, const std::vector<PoseResult> &result) {
  for (auto &obj : result) {
    cv::rectangle(im, cv::Rect{(int)obj.x, (int)obj.y, (int)obj.w, (int)obj.h},
                  cv::Scalar(255, 0, 0), 1);
    constexpr uint32_t steps = 3;
    constexpr uint32_t radius = 5;
    const auto &kpts = obj.keyPoints;
    for (size_t i = 0; i < kpts.size() / steps; i++) {
      cv::Scalar color{KPS_COLORS[i]};
      int32_t x_coord = kpts[steps * i];
      int32_t y_coord = kpts[steps * i + 1];
      if (x_coord % 640 != 0 && y_coord % 640 != 0) {
        if (steps == 3) {
          auto conf = kpts[steps * i + 2];
          if (conf < 0.5) continue;
        }
        cv::circle(im, cv::Point(x_coord, y_coord), radius, color, -1);
      }
    }

    for (size_t i = 0; i < SKELETON.size(); i++) {
      cv::Scalar color{LIMB_COLORS[i]};
      auto &sk = SKELETON[i];
      auto pos1 = cv::Point(int(kpts[(sk[0] - 1) * steps]),
                            int(kpts[(sk[0] - 1) * steps + 1]));
      auto pos2 = cv::Point(int(kpts[(sk[1] - 1) * steps]),
                            int(kpts[(sk[1] - 1) * steps + 1]));

      if (steps == 3) {
        auto conf1 = kpts[(sk[0] - 1) * steps + 2];
        auto conf2 = kpts[(sk[1] - 1) * steps + 2];
        if (conf1 < 0.5 || conf2 < 0.5) continue;
      }

      if (pos1.x % 640 == 0 or pos1.y % 640 == 0 or pos1.x < 0 or pos1.y < 0)
        continue;
      if (pos2.x % 640 == 0 or pos2.y % 640 == 0 or pos2.x < 0 or pos2.y < 0)
        continue;
      cv::line(im, pos1, pos2, color, 2);
    }
  }
}

cmdline::parser ArgumentParser(int argc, char **argv) {
  cmdline::parser args;
  args.add<std::string>(
      "model_prefix", 'm', "model prefix of the model suite files", false,
      "/opt/vastai/vaststreamx/data/models/"
      "yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "../data/configs/yolo_div255_yuv_nv12.json ");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<float>("threshold", 't', "threshold for detection", false, 0.1);
  args.add<std::string>("uri", '\0', "uri to decode", false,
                        "../data/videos/test.mp4");
  args.add<std::string>("output_path", '\0', "output path", false, "");
  args.add<uint32_t>("num_channels", '\0', "number of channles to decode",
                     false, 1);
  args.parse_check(argc, argv);
  return args;
}

void process(std::string uri, const std::string model_prefix,
             std::string vdsp_params, uint32_t device_id, float threshold,
             const std::string output_path, uint32_t index) {
  uint32_t batch_size = 1;
  vsx::ModelCV2 body_model(model_prefix, vdsp_params, batch_size, device_id);
  // vsx::Detector detector(model_prefix, vdsp_params, batch_size, device_id);
  // detector.SetThreshold(threshold);
  vsx::SetDevice(device_id);
  vsx::VideoCapture video_capture(uri, vsx::FULLSPEED_MODE, device_id);
  uint32_t idx = 0;
  auto tick = std::chrono::high_resolution_clock::now();
  while (1) {
    vsx::Image image;
    bool flag = video_capture.read(image);
    if (!flag) {
      std::cout << "cap.read() returns 0\n";
      break;
    }

    std::vector<vsx::Tensor> results_body = body_model.Process(image);
    std::vector<vsx::Tensor> results_host;
    results_host.reserve(results_body.size());
    for (size_t i = 0; i < results_body.size(); ++i) {
      auto tensorfp32 = vsx::ConvertTensorFromFp16ToFp32(results_body[i]);
      results_host.push_back(tensorfp32);
    }

    vsx::TShape input_shape;
    body_model.GetInputShapeByIndex(0, input_shape);
    std::vector<vsx::TShape> output_shapes;
    body_model.GetOutputShapes(output_shapes);
    uint32_t output_cnt = 0;
    body_model.GetOutputCount(output_cnt);
    vsx::TShape image_shape =
        vsx::Tuple{(int64_t)image.Height(), (int64_t)image.Width(), (int64_t)3};
    std::cerr << "threshold: " << threshold << std::endl;

    auto post_process_result =
        post_process(results_host, {input_shape[2], input_shape[3]}, output_cnt,
                     output_shapes, image_shape, threshold);
    PoseResultInfo pose_result;
    pose_result.poses = post_process_result;
    pose_result.posesNums = post_process_result.size();
    pose_result.channelId = index;
    pose_result.frameId = idx++;
    std::cerr << "pose_result.posesNums: " << pose_result.posesNums
              << std::endl;

    // auto result = detector.Process(image);
    if (!output_path.empty()) {
      // auto res_shape = result.Shape();
      // const float* res_data = result.Data<float>();

      vsx::Image cpu_image(image.Format(), image.Width(), image.Height(),
                           vsx::Context::CPU(0), image.WidthPitch(),
                           image.HeightPitch(), image.GetDType());
      cpu_image.CopyFrom(image);

      vsx::Image unpitch_image(image.Format(), image.Width(), image.Height(),
                               vsx::Context::CPU());
      unpitch_image.CopyFrom(cpu_image);
      uint8_t *yuv_nv12 = unpitch_image.MutableData<uint8_t>();
      int width = unpitch_image.WidthPitch() > unpitch_image.Width()
                      ? unpitch_image.WidthPitch()
                      : unpitch_image.Width();
      int height = unpitch_image.HeightPitch() > unpitch_image.Height()
                       ? unpitch_image.HeightPitch()
                       : unpitch_image.Height();

      cv::Mat img_rgb;
      // nv12 -> bgr_interleave
      cv::Mat nv12_opencv(height * 3 / 2, width, CV_8UC1, yuv_nv12, width);
      cv::cvtColor(nv12_opencv, img_rgb, cv::COLOR_YUV2BGR_NV12);

      DrawPose(img_rgb, post_process_result);
      std::string ouput_file =
          output_path + "/img_" + std::to_string(idx) + ".png";
      cv::imwrite(ouput_file, img_rgb);

      // for (int j = 0; j < res_shape[0]; j++) {
      //   if (res_data[0] < 0) break;
      //   cv::Rect2f rect{res_data[2], res_data[3], res_data[4], res_data[5]};
      //   cv::rectangle(img_rgb, rect, cv::Scalar(0, 255, 0), 2);
      //   res_data += vsx::kDetectionOffset;
      // }
      // std::string ouput_file =
      //     output_path + "/img_" + std::to_string(idx++) + ".png";
      // cv::imwrite(ouput_file, img_rgb);

      auto tock = std::chrono::high_resolution_clock::now();
      auto cost =
          std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick)
              .count();
      std::cout << idx << "th Decode+AI @ " << (idx * 1000.0 / cost)
                << " fps\n";
    }
  }
}

int main(int argc, char **argv) {
  auto args = ArgumentParser(argc, argv);
  std::vector<std::shared_ptr<std::thread>> vec_of_threads;
  int num_channels = args.get<uint32_t>("num_channels");

  // uint32_t batch_size = 1;
  std::string uri = args.get<std::string>("uri");
  std::string model_prefix = args.get<std::string>("model_prefix");
  std::string vdsp_params = args.get<std::string>("vdsp_params");
  uint32_t device_id = args.get<uint32_t>("device_id");
  float threshold = args.get<float>("threshold");
  std::string output_path = args.get<std::string>("output_path");

  for (int i = 0; i < num_channels; i++) {
    std::shared_ptr<std::thread> t =
        std::make_shared<std::thread>(process, uri, model_prefix, vdsp_params,
                                      device_id, threshold, output_path, i);
    vec_of_threads.emplace_back(t);
  }
  for (const std::shared_ptr<std::thread> &t : vec_of_threads) {
    t->join();
  }
  return 0;
}