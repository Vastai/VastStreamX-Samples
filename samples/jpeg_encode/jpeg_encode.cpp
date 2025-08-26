
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

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<uint32_t>("height", '\0', "image height", false, 354);
  args.add<uint32_t>("width", '\0', "image width ", false, 474);
  args.add<std::string>("input_file", '\0', "input image file", false,
                        "../data/images/cat_354x474_nv12.yuv");
  args.add<std::string>("output_file", '\0', "output image file", false,
                        "./jpeg_encode_result.jpg");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char* argv[]) {
  auto args = ArgumentParser(argc, argv);
  int width = args.get<uint32_t>("width");
  int height = args.get<uint32_t>("height");
  // init card
  vsx::SetDevice(args.get<uint32_t>("device_id"));

  vsx::JpegEncoder encoder;

  // prepare data
  std::shared_ptr<vsx::DataManager> data;
  vsx::ReadBinaryFile(args.get<std::string>("input_file"), data);

  // create cpu vsx::Image, width_pith,height_pitch use default,
  vsx::Image vsx_image(vsx::ImageFormat::YUV_NV12, width, height, 0, 0, data);

  // start encode
  encoder.SendImage(vsx_image);
  encoder.StopSendImage();

  // get encoded bytes
  std::shared_ptr<vsx::DataManager> encoded_bytes;
  encoder.RecvData(encoded_bytes);
  std::cout << "Encoded data bytes: " << encoded_bytes->GetDataSize() << "\n";

  // write encoded bytes to output file
  vsx::WriteBinaryFile(args.get<std::string>("output_file"),
                       encoded_bytes->GetDataPtr(),
                       encoded_bytes->GetDataSize());

  return 0;
}