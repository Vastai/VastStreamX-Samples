
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
                        "cvtcolor_result.jpg");
  args.add<std::string>("cvtcolor_code", '\0', "cvtcolor code", false,
                        "bgr2rgb_interleave2planar");
  args.parse_check(argc, argv);
  return args;
}

vsx::BuildInOperatorAttrColorCvtCode GetCvtColorCode(
    const std::string& cvtcolor_code, vsx::ImageFormat& input_format,
    vsx::ImageFormat& output_format) {
  std::string code_upper = cvtcolor_code;
  std::transform(code_upper.begin(), code_upper.end(), code_upper.begin(),
                 [](unsigned char c) { return std::toupper(c); });

  if (code_upper == "YUV2RGB_NV12") {
    input_format = vsx::YUV_NV12;
    output_format = vsx::RGB_PLANAR;
    return vsx::kYUV2RGB_NV12;
  }
  if (code_upper == "YUV2BGR_NV12") {
    input_format = vsx::YUV_NV12;
    output_format = vsx::BGR_PLANAR;
    return vsx::kYUV2BGR_NV12;
  }

  if (code_upper == "BGR2RGB") {
    input_format = vsx::BGR_PLANAR;
    output_format = vsx::RGB_PLANAR;
    return vsx::kBGR2RGB;
  }
  if (code_upper == "RGB2BGR") {
    input_format = vsx::RGB_PLANAR;
    output_format = vsx::BGR_PLANAR;
    return vsx::kRGB2BGR;
  }
  if (code_upper == "BGR2RGB_INTERLEAVE2PLANAR") {
    input_format = vsx::BGR_INTERLEAVE;
    output_format = vsx::RGB_PLANAR;
    return vsx::kBGR2RGB_INTERLEAVE2PLANAR;
  }
  if (code_upper == "RGB2BGR_INTERLEAVE2PLANAR") {
    input_format = vsx::RGB_INTERLEAVE;
    output_format = vsx::BGR_PLANAR;
    return vsx::kRGB2BGR_INTERLEAVE2PLANAR;
  }
  if (code_upper == "BGR2BGR_INTERLEAVE2PLANAR") {
    input_format = vsx::BGR_INTERLEAVE;
    output_format = vsx::BGR_PLANAR;
    return vsx::kBGR2BGR_INTERLEAVE2PLANAR;
  }
  if (code_upper == "RGB2RGB_INTERLEAVE2PLANAR") {
    input_format = vsx::RGB_INTERLEAVE;
    output_format = vsx::RGB_PLANAR;
    return vsx::kRGB2RGB_INTERLEAVE2PLANAR;
  }
  if (code_upper == "YUV2GRAY_NV12") {
    input_format = vsx::YUV_NV12;
    output_format = vsx::GRAY;
    return vsx::kYUV2GRAY_NV12;
  }
  if (code_upper == "BGR2GRAY_INTERLEAVE") {
    input_format = vsx::BGR_INTERLEAVE;
    output_format = vsx::GRAY;
    return vsx::kBGR2GRAY_INTERLEAVE;
  }
  if (code_upper == "BGR2GRAY_PLANAR") {
    input_format = vsx::BGR_PLANAR;
    output_format = vsx::GRAY;
    return vsx::kBGR2GRAY_PLANAR;
  }
  if (code_upper == "RGB2GRAY_INTERLEAVE") {
    input_format = vsx::RGB_INTERLEAVE;
    output_format = vsx::GRAY;
    return vsx::kRGB2GRAY_INTERLEAVE;
  }
  if (code_upper == "RGB2GRAY_PLANAR") {
    input_format = vsx::RGB_PLANAR;
    output_format = vsx::GRAY;
    return vsx::kRGB2GRAY_PLANAR;
  }
  if (code_upper == "RGB2YUV_NV12_PLANAR") {
    input_format = vsx::RGB_PLANAR;
    output_format = vsx::YUV_NV12;
    return vsx::kRGB2YUV_NV12_PLANAR;
  }
  if (code_upper == "BGR2YUV_NV12_PLANAR") {
    input_format = vsx::BGR_PLANAR;
    output_format = vsx::YUV_NV12;
    return vsx::kBGR2YUV_NV12_PLANAR;
  }

  CHECK(false) << "Unrecognize cvtcolor code: " << cvtcolor_code << std::endl;
}

int main(int argc, char* argv[]) {
  auto args = ArgumentParser(argc, argv);
  auto device_id = args.get<uint32_t>("device_id");

  vsx::ImageFormat input_format, output_format;
  auto cvtcolor_code = GetCvtColorCode(args.get<std::string>("cvtcolor_code"),
                                       input_format, output_format);
  vsx::SetDevice(device_id);

  vsx::Image image;
  vsx::MakeVsxImage(args.get<std::string>("input_file"), image, input_format);
  auto op =
      vsx::BuildInOperator(vsx::BuildInOperatorType::kSINGLE_OP_CVT_COLOR);

  op.SetAttribute<vsx::AttrKey::kIimageWidth>(image.Width());
  op.SetAttribute<vsx::AttrKey::kIimageHeight>(image.Height());
  op.SetAttribute<vsx::AttrKey::kIimageWidthPitch>(image.Width());
  op.SetAttribute<vsx::AttrKey::kIimageHeightPitch>(image.Height());
  op.SetAttribute<vsx::AttrKey::kOimageWidth>(image.Width());
  op.SetAttribute<vsx::AttrKey::kOimageHeight>(image.Height());
  op.SetAttribute<vsx::AttrKey::kOimageWidthPitch>(image.Width());
  op.SetAttribute<vsx::AttrKey::kOimageHeightPitch>(image.Height());
  op.SetAttribute<vsx::AttrKey::kColorCvtCode>(cvtcolor_code);
  op.SetAttribute<vsx::AttrKey::kColorSpace>(vsx::kCOLOR_SPACE_BT601);

  // copy input to device memory
  auto input_vacc = vsx::Image(image.Format(), image.Width(), image.Height(),
                               vsx::Context::VACC(device_id));
  input_vacc.CopyFrom(image);

  // make device memory image for output
  auto output_vacc = vsx::Image(output_format, image.Width(), image.Height(),
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