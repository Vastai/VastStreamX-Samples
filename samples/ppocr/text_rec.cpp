
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/text_rec.hpp"

#include "common/cmdline.hpp"
#include "common/file_system.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("model_prefix", 'm',
                        "model prefix of the model suite files", false,
                        "/opt/vastai/vaststreamx/data/models/"
                        "resnet34_vd-int8-max-1_3_32_100-vacc/mod");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "../data/configs/crnn_rgbplanar.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("label_file", '\0', "label file", false,
                        "../data/labels/key_37.txt");
  args.add<std::string>("input_file", '\0', "input image", false,
                        "../data/images/word_336.png");
  args.add<std::string>("dataset_filelist", '\0', "input dataset filelist",
                        false, "");
  args.add<std::string>("dataset_root", '\0', "input dataset root", false, "");
  args.add<std::string>("dataset_output_file", '\0', "dataset output file",
                        false, "");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  const int batch_size = 1;
  vsx::TextRecognizer text_rec(args.get<std::string>("model_prefix"),
                               args.get<std::string>("vdsp_params"), batch_size,
                               args.get<uint32_t>("device_id"),
                               args.get<std::string>("label_file"),
                               args.get<std::string>("hw_config"));
  auto image_format = text_rec.GetFusionOpIimageFormat();

  if (!args.get<std::string>("dataset_filelist").empty()) {
    std::vector<std::string> filelist =
        vsx::ReadFileList(args.get<std::string>("dataset_filelist"));
    auto dataset_root = args.get<std::string>("dataset_root");
    std::ofstream outfile(args.get<std::string>("dataset_output_file"),
                          std::ios::out);
    CHECK(outfile.is_open())
        << "Failed to open " << args.get<std::string>("dataset_output_file");

    for (size_t s = 0; s < filelist.size(); s++) {
      auto fullname = filelist[s];
      if (!dataset_root.empty()) fullname = dataset_root + "/" + fullname;
      std::cout << fullname << std::endl;
      vsx::Image vsx_image;
      vsx::MakeVsxImage(fullname, vsx_image, image_format);
      auto result = text_rec.Process(vsx_image);
      std::filesystem::path p(fullname);
      std::string basename = p.stem().string();
      auto result_str = vsx::GetStringFromTensor(result);
      outfile << basename << " " << result_str << std::endl;
    }
    outfile.close();

  } else {
    vsx::Image vsx_image;
    vsx::MakeVsxImage(args.get<std::string>("input_file"), vsx_image,
                      image_format);
    auto result = text_rec.Process(vsx_image);
    std::cout << "score: " << vsx::GetScoreFromTensor(result) << std::endl;
    std::cout << "text: " << vsx::GetStringFromTensor(result) << std::endl;
  }

  return 0;
}