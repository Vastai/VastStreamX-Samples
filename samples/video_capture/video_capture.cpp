
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fstream>
#include <iostream>

#include "common/cmdline.hpp"
#include "common/file_system.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"
#include "vaststreamx/vaststreamx.h"

cmdline::parser ArgumentParser(int argc, char **argv) {
  cmdline::parser args;
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("input_uri", '\0', "input uri", false,
                        "../data/videos/test.mp4");
  args.add<uint32_t>("frame_count", '\0', "frame count to save", false, 0);
  args.add<std::string>("output_folder", '\0', "output image file", false,
                        "./output");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char *argv[]) {
  auto args = ArgumentParser(argc, argv);
  // initialize
  uint32_t device_id = args.get<uint32_t>("device_id");
  vsx::SetDevice(device_id);
  vsx::VideoCapture cap(args.get<std::string>("input_uri"), vsx::FULLSPEED_MODE,
                        device_id);

  auto frame_count = args.get<uint32_t>("frame_count");
  int count = 0;
  auto output_folder = args.get<std::string>("output_folder");
  while (count < frame_count) {
    vsx::Image frame;
    bool ret = cap.read(frame);  // frame format is nv12, memory is in device
    if (!ret) {
      break;
    }
    vsx::Image rgb_image_cpu;
    vsx::CvtColor(frame, rgb_image_cpu, vsx::ImageFormat::RGB_PLANAR,
                  vsx::ImageColorSpace::kCOLOR_SPACE_BT601, true);
    cv::Mat cv_mat;
    vsx::ConvertVsxImageToCvMatBgrPacked(rgb_image_cpu, cv_mat);
    cv::imwrite(output_folder + "/frame_" + std::to_string(count) + ".jpg",
                cv_mat);
    count += 1;
  }
  std::cout << "Close cap\n";
  cap.release();
  std::cout << "Read " << count << " frames.\n";

  return 0;
}
