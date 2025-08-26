
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <fstream>

#include "common/cmdline.hpp"
#include "common/file_system.hpp"
#include "common/utils.hpp"
#include "hih_aligner/hih_aligner.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("model_prefix", 'm',
                        "model prefix of the model suite files", false,
                        "/opt/vastai/vaststreamx/data/models/"
                        "hih_2s-int8-percentile-1_3_256_256-vacc/mod");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "../data/configs/hih_bgr888.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("input_file", '\0', "input file", false,
                        "../data/images/face.jpg");
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
  vsx::Hih face_aligner(args.get<std::string>("model_prefix"),
                        args.get<std::string>("vdsp_params"), batch_size,
                        args.get<uint32_t>("device_id"));
  auto image_format = face_aligner.GetFusionOpIimageFormat();

  if (args.get<std::string>("dataset_filelist").empty()) {
    auto cv_image = cv::imread(args.get<std::string>("input_file"));
    CHECK(!cv_image.empty())
        << "Failed to read image:" << args.get<std::string>("input_file")
        << std::endl;
    vsx::Image vsx_image;
    CHECK(vsx::MakeVsxImage(cv_image, vsx_image, image_format) == 0);
    auto result = face_aligner.Process(vsx_image);
    const float* array_data = result.Data<float>();
    std::cout << "Face alignment results: \n";
    for (size_t i = 0; i < result.GetSize(); i += 2) {
      std::cout << "(" << array_data[i] << ", " << array_data[i + 1] << ")"
                << std::endl;
    }
  } else {
    std::vector<std::string> filelist =
        vsx::ReadFileList(args.get<std::string>("dataset_filelist"));
    auto dataset_root = args.get<std::string>("dataset_root");
    std::ofstream outfile(args.get<std::string>("dataset_output_file"));
    CHECK(outfile.is_open())
        << "Failed to open: " << args.get<std::string>("dataset_output_file");
    for (auto file : filelist) {
      auto fullname = file;
      if (!dataset_root.empty()) fullname = dataset_root + "/" + file;
      auto cv_image = cv::imread(fullname);
      CHECK(!cv_image.empty())
          << "Failed to read image:" << fullname << std::endl;
      vsx::Image vsx_image;
      CHECK(vsx::MakeVsxImage(cv_image, vsx_image, image_format) == 0);
      auto result = face_aligner.Process(vsx_image);
      const float* array_data = result.Data<float>();
      outfile << file << " ";
      for (size_t i = 0; i < result.GetSize(); i++) {
        outfile << array_data[i] << " ";
      }
      outfile << std::endl;
    }
    outfile.close();
  }

  return 0;
}