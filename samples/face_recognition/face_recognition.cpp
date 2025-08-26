
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iomanip>
#include <sstream>

#include "common/classifier.hpp"
#include "common/cmdline.hpp"
#include "common/file_system.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>(
      "model_prefix", 'm', "model prefix of the model suite files", false,
      "/opt/vastai/vaststreamx/data/models/"
      "facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "../data/configs/facenet_bgr888.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("input_file", '\0', "input file", false,
                        "../data/images/face.jpg");
  args.add<std::string>("dataset_filelist", '\0', "input dataset image list",
                        false, "");
  args.add<std::string>("dataset_root", '\0', "input dataset root", false, "");
  args.add<std::string>("dataset_output_folder", '\0', "dataset output folder",
                        false, "");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  const int batch_size = 1;
  vsx::Classifier facenet(args.get<std::string>("model_prefix"),
                          args.get<std::string>("vdsp_params"), batch_size,
                          args.get<uint32_t>("device_id"));
  auto image_format = facenet.GetFusionOpIimageFormat();

  if (args.get<std::string>("dataset_filelist").empty()) {
    auto cv_image = cv::imread(args.get<std::string>("input_file"));
    CHECK(!cv_image.empty())
        << "Failed to read image:" << args.get<std::string>("input_file")
        << std::endl;
    vsx::Image vsx_image;
    CHECK(vsx::MakeVsxImage(cv_image, vsx_image, image_format) == 0);
    auto result = facenet.Process(vsx_image);
    size_t feature_dims = result.GetSize();
    const float* feature = result.Data<float>();
    std::cout << "Feature len: " << feature_dims << std::endl;
    std::cout << "Face feature:\n";
    for (size_t i = 0; i < feature_dims - 1; i++) {
      std::cout << feature[i] << ",";
    }
    std::cout << feature[feature_dims - 1] << std::endl;
  } else {
    std::vector<std::string> filelist =
        vsx::ReadFileList(args.get<std::string>("dataset_filelist"));
    for (size_t s = 0; s < filelist.size(); s++) {
      std::stringstream npz_file;
      npz_file << args.get<std::string>("dataset_output_folder") << "/output_"
               << std::setw(6) << std::setfill('0') << s << ".npz";
      std::cout << "npz file: " << npz_file.str() << std::endl;
      auto file = filelist[s];
      if (!args.get<std::string>("dataset_root").empty()) {
        file = args.get<std::string>("dataset_root") + "/" + filelist[s];
      }
      auto cv_image = cv::imread(file);
      CHECK(!cv_image.empty()) << "Failed to read image:" << file << std::endl;
      vsx::Image vsx_image;
      CHECK(vsx::MakeVsxImage(cv_image, vsx_image, image_format) == 0);
      auto out_tensor = facenet.Process(vsx_image);

      std::unordered_map<std::string, vsx::Tensor> output_map;
      output_map["output_0"] = out_tensor;
      vsx::SaveTensorMap(npz_file.str(), output_map);
    }
  }
  return 0;
}