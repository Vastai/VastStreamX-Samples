
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
                        "crop_result.jpg");
  args.add<std::string>("crop_rect", '\0', "output size [x,y,w,h]", false,
                        "[33,65,416,416]");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char* argv[]) {
  auto args = ArgumentParser(argc, argv);
  uint32_t device_id = args.get<uint32_t>("device_id");
  auto input_file = args.get<std::string>("input_file");
  std::vector<uint32_t> crop_rect =
      vsx::ParseVecUint(args.get<std::string>("crop_rect"));

  vsx::SetDevice(device_id);

  vsx::Image image;
  if (vsx::MakeVsxImage(input_file, image, vsx::BGR_INTERLEAVE) != 0) {
    std::cout << "Read input image failed: " << input_file << std::endl;
  }
  int iimage_width = image.Width();
  int iimage_height = image.Height();
  auto image_format = image.Format();
  int crop_x = crop_rect[0];
  int crop_y = crop_rect[1];
  int crop_w = crop_rect[2];
  int crop_h = crop_rect[3];

  auto op = vsx::BuildInOperator(vsx::BuildInOperatorType::kSINGLE_OP_CROP);

  op.SetAttribute<vsx::AttrKey::kIimageWidth>(iimage_width);
  op.SetAttribute<vsx::AttrKey::kIimageHeight>(iimage_height);
  op.SetAttribute<vsx::AttrKey::kIimageWidthPitch>(iimage_width);
  op.SetAttribute<vsx::AttrKey::kIimageHeightPitch>(iimage_height);
  op.SetAttribute<vsx::AttrKey::kOimageWidth>(crop_w);
  op.SetAttribute<vsx::AttrKey::kOimageHeight>(crop_h);
  op.SetAttribute<vsx::AttrKey::kOimageWidthPitch>(crop_w);
  op.SetAttribute<vsx::AttrKey::kOimageHeightPitch>(crop_h);
  op.SetAttribute<vsx::AttrKey::kIimageFormat>(
      vsx::BuildInOperatorAttrImageType::kBGR888);
  op.SetAttribute<vsx::AttrKey::kCropX>(crop_x);
  op.SetAttribute<vsx::AttrKey::kCropY>(crop_y);

  // execute

  // copy input to device memory
  auto input_vacc = vsx::Image(image_format, iimage_width, iimage_height,
                               vsx::Context::VACC(device_id));
  input_vacc.CopyFrom(image);

  // make device memory image for output
  auto output_vacc =
      vsx::Image(image_format, crop_w, crop_h, vsx::Context::VACC(device_id));
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
