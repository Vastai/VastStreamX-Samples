
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "brightness_op.hpp"
#include "common/cmdline.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("input_file", '\0', "input file", false,
                        "../data/images/dog.jpg");
  args.add<std::string>("output_file", '\0', "output file", false,
                        "brightness_op_result.jpg");
  args.add<std::string>("elf_file", '\0', "elf_file path", false,
                        "/opt/vastai/vaststreamx/data/elf/brightness");
  args.add<float>("scale", '\0', "brightness scale coefficient", false, 2.2);
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char* argv[]) {
  auto args = ArgumentParser(argc, argv);
  uint32_t device_id = args.get<uint32_t>("device_id");
  float scale = args.get<float>("scale");

  vsx::SetDevice(device_id);
  auto brighness_op =
      vsx::BrightnessOp("img_brightness_adjust",
                        args.get<std::string>("elf_file"), device_id, scale);
  vsx::Image input_image;
  vsx::MakeVsxImage(args.get<std::string>("input_file"), input_image,
                    vsx::YUV_NV12);
  auto output_vacc = brighness_op.Process(input_image);
  // copy result to cpu memory
  auto output_cpu = vsx::Image(output_vacc.Format(), output_vacc.Width(),
                               output_vacc.Height());
  output_cpu.CopyFrom(output_vacc);

  cv::Mat out_mat;
  vsx::ConvertVsxImageToCvMatBgrPacked(output_cpu, out_mat);
  cv::imwrite(args.get<std::string>("output_file"), out_mat);
  std::cout << "Write result to " << args.get<std::string>("output_file")
            << std::endl;
  return 0;
}