
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/cmdline.hpp"
#include "common/detr_model.hpp"
#include "common/file_system.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("model_prefix", 'm',
                        "model prefix of the model suite files", false,
                        "/opt/vastai/vaststreamx/data/models/"
                        "detr_res50-fp16-none-1_3_1066_800-vacc/mod");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false, "");
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "./data/configs/detr_bgr888.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<float>("threshold", '\0', "threshold for detection", false, 0.5);
  args.add<std::string>("label_file", '\0', "label file", false,
                        "../data/labels/coco2id.txt");
  args.add<std::string>("input_file", '\0', "input file", false,
                        "../data/images/dog.jpg");
  args.add<std::string>("output_file", '\0', "output file", false,
                        "result.png");
  args.add<std::string>("dataset_filelist", '\0', "dataset filename list",
                        false, "");
  args.add<std::string>("dataset_root", '\0', "dataset root", false, "");
  args.add<std::string>("dataset_output_folder", '\0',
                        "dataset output folder path", false, "");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  auto labels = vsx::LoadLabels(args.get<std::string>("label_file"));
  const int batch_size = 1;
  vsx::DetrModel detector(args.get<std::string>("model_prefix"),
                          args.get<std::string>("vdsp_params"), batch_size,
                          args.get<uint32_t>("device_id"),
                          args.get<float>("threshold"),
                          args.get<std::string>("hw_config"));
  auto image_format = detector.GetFusionOpIimageFormat();

  if (args.get<std::string>("dataset_filelist").empty()) {
    auto cv_image = cv::imread(args.get<std::string>("input_file"));
    CHECK(!cv_image.empty())
        << "Failed to read image:" << args.get<std::string>("input_file")
        << std::endl;
    vsx::Image vsx_image;
    CHECK(vsx::MakeVsxImage(cv_image, vsx_image, image_format) == 0);
    auto result = detector.Process(vsx_image);
    auto res_shape = result.Shape();
    const float* res_data = result.Data<float>();
    std::cout << "Detection objects:\n";
    for (int i = 0; i < res_shape[0]; i++) {
      if (res_data[0] < 0) break;
      std::string class_name = labels[static_cast<int>(res_data[0])];
      float score = res_data[1];
      std::cout << "Object class: " << class_name << ", score: " << score
                << ", bbox: [" << res_data[2] << ", " << res_data[3] << ", "
                << res_data[4] << ", " << res_data[5] << "]\n";
      cv::Rect2f rect = {res_data[2], res_data[3], res_data[4], res_data[5]};
      cv::rectangle(cv_image, rect, cv::Scalar(0, 255, 0), 2);
      res_data += vsx::kDetectionOffset;
    }
    cv::imwrite(args.get<std::string>("output_file"), cv_image);
  } else {
    auto filelist =
        vsx::ReadFileList(args.get<std::string>("dataset_filelist"));
    auto dataset_root = args.get<std::string>("dataset_root");
    auto dataset_output_folder = args.get<std::string>("dataset_output_folder");
    for (size_t s = 0; s < filelist.size(); s++) {
      auto filename = filelist[s];
      if (!dataset_root.empty()) filename = dataset_root + "/" + filelist[s];
      auto cv_image = cv::imread(filename);
      CHECK(!cv_image.empty())
          << "Failed to read image:" << filename << std::endl;
      vsx::Image vsx_image;
      CHECK(vsx::MakeVsxImage(cv_image, vsx_image, image_format) == 0);
      auto result = detector.Process(vsx_image);
      auto res_shape = result.Shape();
      const float* res_data = result.Data<float>();

      std::filesystem::path p(filename);
      auto outfile = dataset_output_folder + "/" + p.stem().string() + ".txt";
      std::ofstream of(outfile);
      if (!of.is_open()) {
        std::cout << "Error, Failed to open: " << outfile << std::endl;
        return -1;
      }

      std::cout << p.filename().string() << " detection objects:\n";
      for (int i = 0; i < res_shape[0]; i++) {
        if (res_data[0] < 0) break;
        std::string class_name = labels[static_cast<int>(res_data[0])];
        float score = res_data[1];
        std::cout << "Object class: " << class_name << ", score: " << score
                  << ", bbox: [" << res_data[2] << ", " << res_data[3] << ", "
                  << res_data[4] << ", " << res_data[5] << "]\n";
        of << class_name << " " << score << " " << int(res_data[2]) << " "
           << int(res_data[3]) << " " << int(res_data[2] + res_data[4]) << " "
           << int(res_data[3] + res_data[5]) << std::endl;
        res_data += vsx::kDetectionOffset;
      }
      of.close();
    }
  }

  return 0;
}