
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
  args.add<std::string>("input_file", '\0', "input image", false,
                        "../data/images/dog.jpg");
  args.add<std::string>("output_file1", '\0', "output image", false,
                        "batch_crop_resize_result1.jpg");
  args.add<std::string>("output_file2", '\0', "output image", false,
                        "batch_crop_resize_result2.jpg");
  args.add<std::string>("output_size", '\0', "output size [w,h]", false,
                        "[512,512]");
  args.add<std::string>("crop_rect1", '\0', "crop rect1 [x,y,w,h]", false,
                        "[50,70,131,230]");
  args.add<std::string>("crop_rect2", '\0', "crop rect1 [x,y,w,h]", false,
                        "[60,90,150,211]");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char* argv[]) {
  auto args = ArgumentParser(argc, argv);
  uint32_t device_id = args.get<uint32_t>("device_id");
  auto input_file = args.get<std::string>("input_file");
  std::vector<uint32_t> crop_rect1 =
      vsx::ParseVecUint(args.get<std::string>("crop_rect1"));
  std::vector<uint32_t> crop_rect2 =
      vsx::ParseVecUint(args.get<std::string>("crop_rect2"));
  std::vector<uint32_t> output_size =
      vsx::ParseVecUint(args.get<std::string>("output_size"));

  vsx::SetDevice(device_id);

  vsx::Image image;
  if (vsx::MakeVsxImage(input_file, image, vsx::BGR_INTERLEAVE) != 0) {
    std::cout << "Read input image failed: " << input_file << std::endl;
  }

  int crop1_x = crop_rect1[0];
  int crop1_y = crop_rect1[1];
  int crop1_w = crop_rect1[2];
  int crop1_h = crop_rect1[3];

  int crop2_x = crop_rect2[0];
  int crop2_y = crop_rect2[1];
  int crop2_w = crop_rect2[2];
  int crop2_h = crop_rect2[3];

  int oimage_width = output_size[0];
  int oimage_height = output_size[1];

  int rects = 2;
  auto op = vsx::BuildInOperator(
      vsx::BuildInOperatorType::kSINGLE_OP_BATCH_CROP_RESIZE);

  // [4] set attr

  op.SetAttribute<vsx::AttrKey::kIimageFormat>(
      vsx::BuildInOperatorAttrImageType::kBGR888);
  op.SetAttribute<vsx::AttrKey::kIimageWidth>(image.Width());
  op.SetAttribute<vsx::AttrKey::kIimageHeight>(image.Height());
  op.SetAttribute<vsx::AttrKey::kIimageWidthPitch>(image.Width());
  op.SetAttribute<vsx::AttrKey::kIimageHeightPitch>(image.Height());
  op.SetAttribute<vsx::AttrKey::kOimageWidth>(oimage_width);
  op.SetAttribute<vsx::AttrKey::kOimageHeight>(oimage_height);
  op.SetAttribute<vsx::AttrKey::kOimageWidthPitch>(oimage_width);
  op.SetAttribute<vsx::AttrKey::kOimageHeightPitch>(oimage_height);
  op.SetAttribute<vsx::AttrKey::kResizeType>(
      vsx::ImageResizeType::kRESIZE_TYPE_BILINEAR_CV);

  op.SetAttribute<vsx::AttrKey::kCropNum>(rects);

  op.SetAttribute<vsx::AttrKey::kCropX>(crop1_x, 0);
  op.SetAttribute<vsx::AttrKey::kCropY>(crop1_y, 0);
  op.SetAttribute<vsx::AttrKey::kCropWidth>(crop1_w, 0);
  op.SetAttribute<vsx::AttrKey::kCropHeight>(crop1_h, 0);

  op.SetAttribute<vsx::AttrKey::kCropX>(crop2_x, 1);
  op.SetAttribute<vsx::AttrKey::kCropY>(crop2_y, 1);
  op.SetAttribute<vsx::AttrKey::kCropWidth>(crop2_w, 1);
  op.SetAttribute<vsx::AttrKey::kCropHeight>(crop2_h, 1);

  // execute

  // copy input to device memory
  auto input_vacc = vsx::Image(image.Format(), image.Width(), image.Height(),
                               vsx::Context::VACC(device_id));
  input_vacc.CopyFrom(image);

  // make device memory image for output
  auto output1_vacc = vsx::Image(image.Format(), oimage_width, oimage_height,
                                 vsx::Context::VACC(device_id));
  auto output2_vacc = vsx::Image(image.Format(), oimage_width, oimage_height,
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

  cv::Mat out1_mat, out2_mat;
  vsx::ConvertVsxImageToCvMatBgrPacked(output1_cpu, out1_mat);
  vsx::ConvertVsxImageToCvMatBgrPacked(output2_cpu, out2_mat);
  cv::imwrite(args.get<std::string>("output_file1"), out1_mat);
  cv::imwrite(args.get<std::string>("output_file2"), out2_mat);
  std::cout << "save result to : " << args.get<std::string>("output_file1")
            << ", " << args.get<std::string>("output_file2") << std::endl;
  return 0;
}
