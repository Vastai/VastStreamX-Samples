
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <chrono>
#include <thread>

#include "common/cmdline.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"
#include "vaststreamx/vaststreamx.h"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("input_file", '\0', "input_image", false,
                        "../data/images/dog.jpg");
  args.add<std::string>("output_file", '\0', "output image", false,
                        "flip_result.jpg");
  args.add<std::string>("flip_type", '\0', "flip type x or y", false, "x");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char* argv[]) {
  auto args = ArgumentParser(argc, argv);
  uint32_t device_id = args.get<uint32_t>("device_id");
  auto input_file = args.get<std::string>("input_file");
  auto flip_type_str = args.get<std::string>("flip_type");

  vsx::ImageFlipType flip_type;

  if (flip_type_str == "x" || flip_type_str == "X") {
    flip_type = vsx::ImageFlipType::kFLIP_TYPE_X_AXIS;
  } else if (flip_type_str == "y" || flip_type_str == "Y") {
    flip_type = vsx::ImageFlipType::kFLIP_TYPE_Y_AXIS;
  } else {
    std::cout << "Unsupport flip type: \"" << flip_type_str
              << "\", only support x or y\n";
    return -1;
  }
  vsx::SetDevice(device_id);

  vsx::Image image;
  if (vsx::MakeVsxImage(input_file, image, vsx::YUV_NV12) != 0) {
    std::cout << "Read input image failed: " << input_file << std::endl;
  }
  int iimage_width = image.Width();
  int iimage_height = image.Height();
  auto image_format = image.Format();

  auto op = vsx::BuildInOperator(vsx::BuildInOperatorType::kSINGLE_OP_FLIP);

  op.SetAttribute<vsx::AttrKey::kIimageWidth>(iimage_width);
  op.SetAttribute<vsx::AttrKey::kIimageHeight>(iimage_height);
  op.SetAttribute<vsx::AttrKey::kIimageWidthPitch>(iimage_width);
  op.SetAttribute<vsx::AttrKey::kIimageHeightPitch>(iimage_height);
  op.SetAttribute<vsx::AttrKey::kOimageWidth>(iimage_width);
  op.SetAttribute<vsx::AttrKey::kOimageHeight>(iimage_height);
  op.SetAttribute<vsx::AttrKey::kOimageWidthPitch>(iimage_width);
  op.SetAttribute<vsx::AttrKey::kOimageHeightPitch>(iimage_height);
  op.SetAttribute<vsx::AttrKey::kIimageFormat>(
      vsx::BuildInOperatorAttrImageType::kYUV_NV12);
  op.SetAttribute<vsx::AttrKey::kDirection>(flip_type);

  // execute

  // copy input to device memory
  auto input_vacc = vsx::Image(image_format, iimage_width, iimage_height,
                               vsx::Context::VACC(device_id));
  input_vacc.CopyFrom(image);

  // make device memory image for output
  auto output_vacc = vsx::Image(image_format, iimage_width, iimage_height,
                                vsx::Context::VACC(device_id));
  std::vector<vsx::Image> outputs{output_vacc};
  // buildin operator execute
  op.Execute({input_vacc}, outputs);
  // copy result to cpu memory
  vsx::Image output_cpu = vsx::Image(output_vacc.Format(), output_vacc.Width(),
                                     output_vacc.Height());
  output_cpu.CopyFrom(output_vacc);

  cv::Mat out_mat;
  vsx::ConvertVsxImageToCvMatBgrPacked(output_cpu, out_mat);
  cv::imwrite(args.get<std::string>("output_file"), out_mat);
  std::cout << "save result to : " << args.get<std::string>("output_file")
            << std::endl;
  return 0;
}
