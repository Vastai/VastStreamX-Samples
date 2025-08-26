
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
  args.add<std::string>("input_file", '\0', "input image file", false,
                        "../data/images/cat.jpg");
  args.add<std::string>("output_file", '\0', "output image file", false,
                        "./jpeg_decode_result.yuv");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char* argv[]) {
  auto args = ArgumentParser(argc, argv);
  // init card
  vsx::SetDevice(args.get<uint32_t>("device_id"));

  std::shared_ptr<vsx::DataManager> data;
  vsx::ReadBinaryFile(args.get<std::string>("input_file"), data);

  // decode jpeg data
  vsx::JpegDecoder decoder;
  decoder.SendData(data);
  decoder.StopSendData();

  // get decoded image
  vsx::Image output;
  decoder.RecvImage(output);

  // width_align: 64， height_align 4
  std::cout << "Output image width: " << output.Width()
            << ", height: " << output.Height()
            << ", width_pitch: " << output.WidthPitch()
            << ", height_pitch: " << output.HeightPitch() << std::endl;

  std::cout << "Decoded image format is: "
            << vsx::ImageFormatToString(output.Format()) << "\n";
  // copy data from device to cpu
  auto cpu_image = vsx::Image(output.Format(), output.Width(), output.Height(),
                              vsx::Context::CPU(), output.WidthPitch(),
                              output.HeightPitch());
  cpu_image.CopyFrom(output);

  // deltet pitch data
  auto unpitch_image =
      vsx::Image(output.Format(), output.Width(), output.Height());
  unpitch_image.CopyFrom(cpu_image);

  // write decoded image to output file
  vsx::WriteBinaryFile(args.get<std::string>("output_file"),
                       unpitch_image.Data<void>(),
                       unpitch_image.GetDataBytes());
  // save as bmp
  cv::Mat cv_mat;
  vsx::ConvertVsxImageToCvMatBgrPacked(unpitch_image, cv_mat);

  std::filesystem::path p(args.get<std::string>("output_file"));
  cv::imwrite(p.stem().string() + ".bmp", cv_mat);

  return 0;
}