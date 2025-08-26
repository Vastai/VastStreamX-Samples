
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
  args.add<std::string>("output_size1", '\0', "output size1 [w,h]", false,
                        "[256,256]");
  args.add<std::string>("output_size2", '\0', "output size2 [w,h]", false,
                        "[320,320]");
  args.add<std::string>("output_file1", '\0', "output image1", false,
                        "scale_result1.jpg");
  args.add<std::string>("output_file2", '\0', "output image2", false,
                        "scale_result2.jpg");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char* argv[]) {
  auto args = ArgumentParser(argc, argv);
  vsx::SetDevice(args.get<uint32_t>("device_id"));

  uint32_t device_id = args.get<uint32_t>("device_id");

  std::vector<uint32_t> output_size1 =
      vsx::ParseVecUint(args.get<std::string>("output_size1"));
  std::vector<uint32_t> output_size2 =
      vsx::ParseVecUint(args.get<std::string>("output_size2"));

  vsx::Image image;
  vsx::MakeVsxImage(args.get<std::string>("input_file"), image, vsx::YUV_NV12);

  int output1_width = output_size1[0];
  int output1_height = output_size1[1];

  int output2_width = output_size2[0];
  int output2_height = output_size2[1];

  auto op = vsx::BuildInOperator(vsx::BuildInOperatorType::kSINGLE_OP_SCALE);

  op.SetAttribute<vsx::AttrKey::kIimageWidth>(image.Width());
  op.SetAttribute<vsx::AttrKey::kIimageHeight>(image.Height());
  op.SetAttribute<vsx::AttrKey::kIimageWidthPitch>(image.Width());
  op.SetAttribute<vsx::AttrKey::kIimageHeightPitch>(image.Height());
  op.SetAttribute<vsx::AttrKey::kResizeType>(
      vsx::ImageResizeType::kRESIZE_TYPE_BILINEAR);

  op.SetAttribute<vsx::AttrKey::kOimageCnt>(2);

  op.SetAttribute<vsx::AttrKey::kOimageWidth>(output1_width, 0);
  op.SetAttribute<vsx::AttrKey::kOimageHeight>(output1_height, 0);
  op.SetAttribute<vsx::AttrKey::kOimageWidthPitch>(output1_width, 0);
  op.SetAttribute<vsx::AttrKey::kOimageHeightPitch>(output1_height, 0);

  op.SetAttribute<vsx::AttrKey::kOimageWidth>(output2_width, 1);
  op.SetAttribute<vsx::AttrKey::kOimageHeight>(output2_height, 1);
  op.SetAttribute<vsx::AttrKey::kOimageWidthPitch>(output2_width, 1);
  op.SetAttribute<vsx::AttrKey::kOimageHeightPitch>(output2_height, 1);

  // copy input to device memory
  auto input_vacc = vsx::Image(image.Format(), image.Width(), image.Height(),
                               vsx::Context::VACC(device_id));
  input_vacc.CopyFrom(image);

  // make device memory image for output
  auto output1_vacc = vsx::Image(image.Format(), output1_width, output1_height,
                                 vsx::Context::VACC(device_id));
  auto output2_vacc = vsx::Image(image.Format(), output2_width, output2_height,
                                 vsx::Context::VACC(device_id));
  std::vector<vsx::Image> outputs{output1_vacc, output2_vacc};

  // buildin operator execute
  op.Execute({input_vacc}, outputs);

  // copy result to cpu memory
  vsx::Image output1_cpu = vsx::Image(
      output1_vacc.Format(), output1_vacc.Width(), output1_vacc.Height());
  output1_cpu.CopyFrom(output1_vacc);
  vsx::Image output2_cpu = vsx::Image(
      output2_vacc.Format(), output2_vacc.Width(), output2_vacc.Height());
  output2_cpu.CopyFrom(output2_vacc);

  cv::Mat out_mat;
  vsx::ConvertVsxImageToCvMatBgrPacked(output1_cpu, out_mat);
  cv::imwrite(args.get<std::string>("output_file1"), out_mat);

  vsx::ConvertVsxImageToCvMatBgrPacked(output2_cpu, out_mat);
  cv::imwrite(args.get<std::string>("output_file2"), out_mat);

  std::cout << "Save result to " << args.get<std::string>("output_file1")
            << ", " << args.get<std::string>("output_file2") << std::endl;

  return 0;
}