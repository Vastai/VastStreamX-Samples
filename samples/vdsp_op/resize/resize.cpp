
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/cmdline.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"
#include "vaststreamx/vaststreamx.h"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("input_file", '\0', "input image", false,
                        "../data/images/dog.jpg");
  args.add<std::string>("output_size", '\0', "output size", false, "[256,256]");
  args.add<std::string>("output_file", '\0', "output image", false,
                        "resize_result.jpg");
  args.parse_check(argc, argv);
  return args;
}
int resize_bgr888_to_bgr888(cmdline::parser& args) {
  uint32_t device_id = args.get<uint32_t>("device_id");
  vsx::Image image;
  vsx::MakeVsxImage(args.get<std::string>("input_file"), image,
                    vsx::RGB_INTERLEAVE);

  std::vector<uint32_t> output_size =
      vsx::ParseVecUint(args.get<std::string>("output_size"));
  int output_width = output_size[0];
  int output_height = output_size[1];

  auto op = vsx::BuildInOperator(vsx::BuildInOperatorType::kSINGLE_OP_RESIZE);

  op.SetAttribute<vsx::AttrKey::kIimageWidth>(image.Width());
  op.SetAttribute<vsx::AttrKey::kIimageHeight>(image.Height());
  op.SetAttribute<vsx::AttrKey::kIimageWidthPitch>(image.Width());
  op.SetAttribute<vsx::AttrKey::kIimageHeightPitch>(image.Height());
  op.SetAttribute<vsx::AttrKey::kIimageFormat>(
      vsx::BuildInOperatorAttrImageType::kRGB888);
  op.SetAttribute<vsx::AttrKey::kOimageWidth>(output_width);
  op.SetAttribute<vsx::AttrKey::kOimageHeight>(output_height);
  op.SetAttribute<vsx::AttrKey::kOimageWidthPitch>(output_width);
  op.SetAttribute<vsx::AttrKey::kOimageHeightPitch>(output_height);
  op.SetAttribute<vsx::AttrKey::kOimageFormat>(
      vsx::BuildInOperatorAttrImageType::kRGB888);
  op.SetAttribute<vsx::AttrKey::kResizeType>(
      vsx::ImageResizeType::kRESIZE_TYPE_BILINEAR_CV);

  // copy input to device memory
  auto input_vacc = vsx::Image(image.Format(), image.Width(), image.Height(),
                               vsx::Context::VACC(device_id));
  input_vacc.CopyFrom(image);

  // make device memory image for output
  auto output_vacc = vsx::Image(image.Format(), output_width, output_height,
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
  std::cout << "Save result to " << args.get<std::string>("output_file")
            << std::endl;

  return 0;
}

int resize_rgb888_to_rgb_planar(cmdline::parser& args) {
  uint32_t device_id = args.get<uint32_t>("device_id");
  vsx::Image image;
  vsx::MakeVsxImage(args.get<std::string>("input_file"), image,
                    vsx::RGB_INTERLEAVE);

  std::vector<uint32_t> output_size =
      vsx::ParseVecUint(args.get<std::string>("output_size"));
  int output_width = output_size[0];
  int output_height = output_size[1];

  auto op = vsx::BuildInOperator(vsx::BuildInOperatorType::kSINGLE_OP_RESIZE);

  op.SetAttribute<vsx::AttrKey::kIimageWidth>(image.Width());
  op.SetAttribute<vsx::AttrKey::kIimageHeight>(image.Height());
  op.SetAttribute<vsx::AttrKey::kIimageWidthPitch>(image.Width());
  op.SetAttribute<vsx::AttrKey::kIimageHeightPitch>(image.Height());
  op.SetAttribute<vsx::AttrKey::kIimageFormat>(
      vsx::BuildInOperatorAttrImageType::kRGB888);
  op.SetAttribute<vsx::AttrKey::kOimageWidth>(output_width);
  op.SetAttribute<vsx::AttrKey::kOimageHeight>(output_height);
  op.SetAttribute<vsx::AttrKey::kOimageWidthPitch>(output_width);
  op.SetAttribute<vsx::AttrKey::kOimageHeightPitch>(output_height);
  op.SetAttribute<vsx::AttrKey::kOimageFormat>(
      vsx::BuildInOperatorAttrImageType::kRGB_PLANAR);
  op.SetAttribute<vsx::AttrKey::kResizeType>(
      vsx::ImageResizeType::kRESIZE_TYPE_BILINEAR_CV);

  // copy input to device memory
  auto input_vacc = vsx::Image(image.Format(), image.Width(), image.Height(),
                               vsx::Context::VACC(device_id));
  input_vacc.CopyFrom(image);

  // make device memory image for output
  auto output_vacc = vsx::Image(vsx::RGB_PLANAR, output_width, output_height,
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
  std::cout << "Save result to " << args.get<std::string>("output_file")
            << std::endl;
  return 0;
}

int main(int argc, char* argv[]) {
  auto args = ArgumentParser(argc, argv);
  vsx::SetDevice(args.get<uint32_t>("device_id"));

  resize_bgr888_to_bgr888(args);
  // resize_rgb888_to_rgb_planar(args);
  return 0;
}