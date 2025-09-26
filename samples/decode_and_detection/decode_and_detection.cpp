
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
#include "opencv2/opencv.hpp"
#include "vaststreamx/media/video_capture.h"
#include "yolov5_det.h"
cmdline::parser ArgumentParser(int argc, char** argv) {
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
  args.add<uint32_t>("drop_per_frames", '\0', "drop one frame per frames",
                     false, 0);
  args.parse_check(argc, argv);
  return args;
}

void process(std::string uri, const std::string model_prefix,
             std::string vdsp_params, uint32_t device_id, float threshold,
             const std::string output_path, uint32_t index,uint64_t drop_per_frames=0) {
  uint32_t batch_size = 1;
  vsx::Detector detector(model_prefix, vdsp_params, batch_size, device_id);
  detector.SetThreshold(threshold);
  vsx::SetDevice(device_id);
  vsx::VideoCapture video_capture(uri, vsx::FULLSPEED_MODE, device_id);
  uint32_t idx = 0;
  auto tick = std::chrono::high_resolution_clock::now();
  uint64_t frame_count = 0;
  while (1) {
    vsx::Image image;
    bool flag = video_capture.read(image);
    if (!flag) {
      std::cout << "cap.read() returns 0\n";
      break;
    }
    // drop frame per frames
    frame_count++;
    if (frame_count == drop_per_frames) {
      std::cout << "drop frame\n";
      frame_count = 0;
      continue;
    }
    
    auto result = detector.Process(image);
    if (!output_path.empty()) {
      auto res_shape = result.Shape();
      const float* res_data = result.Data<float>();

      vsx::Image cpu_image(image.Format(), image.Width(), image.Height(),
                           vsx::Context::CPU(0), image.WidthPitch(),
                           image.HeightPitch(), image.GetDType());
      cpu_image.CopyFrom(image);

      vsx::Image unpitch_image(image.Format(), image.Width(), image.Height(),
                               vsx::Context::CPU());
      unpitch_image.CopyFrom(cpu_image);
      uint8_t* yuv_nv12 = unpitch_image.MutableData<uint8_t>();
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
      DetecResultInfo det_result;
      det_result.channelId = index;
      det_result.frameId = idx++;
      for (int j = 0; j < res_shape[0]; j++) {
        if (res_data[0] < 0) break;
        ObjectData object;
        object.classId = res_data[0];
        object.score = res_data[1];
        object.x = res_data[2];
        object.y = res_data[3];
        object.width = res_data[4];
        object.height = res_data[5];
        det_result.objects.emplace_back(object);
        det_result.obj_nums++;
        std::cerr << "classId: " << res_data[0] << ", score: " << res_data[1]
                  << "\n";
        std::cerr << "bounding_box: xmin:" << res_data[2]
                  << ", ymin:" << res_data[3] << ", width:" << res_data[4]
                  << ", height:" << res_data[5] << "\n";
        cv::Rect2f rect{res_data[2], res_data[3], res_data[4], res_data[5]};
        cv::rectangle(img_rgb, rect, cv::Scalar(0, 255, 0), 2);
        res_data += vsx::kDetectionOffset;
      }
      std::string ouput_file =
          output_path + "/img_" + std::to_string(idx) + ".png";
      cv::imwrite(ouput_file, img_rgb);

      auto tock = std::chrono::high_resolution_clock::now();
      auto cost =
          std::chrono::duration_cast<std::chrono::milliseconds>(tock - tick)
              .count();
      std::cout << index << "th Decode+AI @ " << (idx * 1000.0 / cost)
                << " fps\n";
    }
  }
}

int main(int argc, char** argv) {
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
  uint32_t drop_per_frames = args.get<uint32_t>("drop_per_frames");

  for (int i = 0; i < num_channels; i++) {
    std::shared_ptr<std::thread> t =
        std::make_shared<std::thread>(process, uri, model_prefix, vdsp_params,
                                      device_id, threshold, output_path, i,drop_per_frames);
    vec_of_threads.emplace_back(t);
  }
  for (const std::shared_ptr<std::thread>& t : vec_of_threads) {
    t->join();
  }
  return 0;
}