
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
  args.add<std::string>("output_file", '\0', "output image", false,
                        "scale_result.jpg");
  args.add<std::string>("output_size", '\0', "output size [w,h]", false,
                        "[320,320]");
  args.add<std::string>(
      "matrix", '\0', "warpaffine matirx, [x0,x1,x2,y0,y1,y2]", false,
      "[0.7890625, -0.611328125, 56.0, 0.611328125, 0.7890625, -416.0]");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char* argv[]) {
  auto args = ArgumentParser(argc, argv);
  uint32_t device_id = args.get<uint32_t>("device_id");
  auto input_file = args.get<std::string>("input_file");
  std::vector<uint32_t> output_size =
      vsx::ParseVecUint(args.get<std::string>("output_size"));
  std::vector<float> matrix =
      vsx::ParseVecFloat(args.get<std::string>("matrix"));

  vsx::SetDevice(device_id);

  vsx::Image image;
  if (vsx::MakeVsxImage(input_file, image, vsx::YUV_NV12) != 0) {
    std::cout << "Read input image failed: " << input_file << std::endl;
  }
  int iimage_width = image.Width();
  int iimage_height = image.Height();
  int oimage_width = output_size[0];
  int oimage_height = output_size[1];

  auto image_format = image.Format();

  auto op =
      vsx::BuildInOperator(vsx::BuildInOperatorType::kSINGLE_OP_WARP_AFFINE);

  op.SetAttribute<vsx::AttrKey::kIimageFormat>(
      vsx::BuildInOperatorAttrImageType::kYUV_NV12);
  op.SetAttribute<vsx::AttrKey::kIimageWidth>(iimage_width);
  op.SetAttribute<vsx::AttrKey::kIimageHeight>(iimage_height);
  op.SetAttribute<vsx::AttrKey::kIimageWidthPitch>(iimage_width);
  op.SetAttribute<vsx::AttrKey::kIimageHeightPitch>(iimage_height);

  op.SetAttribute<vsx::AttrKey::kOimageWidth>(oimage_width);
  op.SetAttribute<vsx::AttrKey::kOimageHeight>(oimage_height);
  op.SetAttribute<vsx::AttrKey::kOimageWidthPitch>(oimage_width);
  op.SetAttribute<vsx::AttrKey::kOimageHeightPitch>(oimage_height);

  op.SetAttribute<vsx::AttrKey::kFlags>(
      vsx::ImageWarpAffineFlag::kWARP_AFFINE_FLAG_BILINEAR);

  op.SetAttribute<vsx::AttrKey::kBorderMode>(
      vsx::ImagePaddingType::kPADDING_TYPE_CONSTANT);
  op.SetAttribute<vsx::AttrKey::kBorderValue>({114, 114, 114});
  op.SetAttribute<vsx::AttrKey::kM>(matrix);

  // execute
  // copy input to device memory
  auto input_vacc = vsx::Image(image_format, iimage_width, iimage_height,
                               vsx::Context::VACC(device_id));
  input_vacc.CopyFrom(image);

  // make device memory image for output
  auto output_vacc = vsx::Image(image_format, oimage_width, oimage_width,
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