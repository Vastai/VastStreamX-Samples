
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/cmdline.hpp"
#include "common/face_detector.hpp"
#include "common/file_system.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>(
      "model_prefix", 'm', "model prefix of the model suite files", false,
      "/opt/vastai/vaststreamx/data/models/"
      "retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "./data/configs/retinaface_rgbplanar.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<float>("threshold", '\0', "threshold for detection", false, 0.5);
  args.add<std::string>("input_file", '\0', "input file", false,
                        "../data/images/face.jpg");
  args.add<std::string>("output_file", '\0', "output file", false,
                        "face_det_result.jpg");
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
  const int batch_size = 1;
  vsx::FaceDetector face_detector(args.get<std::string>("model_prefix"),
                                  args.get<std::string>("vdsp_params"),
                                  batch_size, args.get<uint32_t>("device_id"));
  face_detector.SetThreshold(args.get<float>("threshold"));
  auto image_format = face_detector.GetFusionOpIimageFormat();

  if (args.get<std::string>("dataset_filelist").empty()) {
    auto cv_image = cv::imread(args.get<std::string>("input_file"));
    CHECK(!cv_image.empty())
        << "Failed to read image:" << args.get<std::string>("input_file")
        << std::endl;
    vsx::Image vsx_image;
    vsx::MakeVsxImage(cv_image, vsx_image, image_format);
    auto result = face_detector.Process(vsx_image);
    auto res_shape = result.Shape();
    const float* res_data = result.Data<float>();
    std::cout << "Face bboxes and landmarks:\n";
    for (int i = 0; i < res_shape[0]; i++) {
      if (res_data[0] < 0) break;
      std::cout << "Index:" << i << ", score: " << res_data[0] << ", bbox: ["
                << res_data[1] << ", " << res_data[2] << ", " << res_data[3]
                << ", " << res_data[4] << "], landmarks: [ ";
      std::vector<cv::Point2f> landmarks;
      for (int j = 0; j < res_shape[1] - 5; j += 2) {
        std::cout << "[" << res_data[5 + j] << "," << res_data[5 + j + 1]
                  << "] ";
        landmarks.push_back(cv::Point2f(res_data[5 + j], res_data[5 + j + 1]));
      }
      std::cout << "]\n";
      cv::Rect2f rect = {res_data[1], res_data[2], res_data[3], res_data[4]};
      cv::rectangle(cv_image, rect, cv::Scalar(0, 0, 255), 2);
      for (auto& point : landmarks) {
        cv::circle(cv_image, point, 2, cv::Scalar(0, 255, 0), 2);
      }
      res_data += res_shape[1];
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
      vsx::Image image;
      vsx::MakeVsxImage(filename, image, image_format);
      auto result = face_detector.Process(image);
      auto res_shape = result.Shape();
      const float* res_data = result.Data<float>();

      std::filesystem::path p(filename);
      auto subdir = p.parent_path().filename().string();
      std::cout << "subdir:" << subdir << std::endl;
      std::filesystem::create_directories(dataset_output_folder + "/" + subdir);
      auto outfile = dataset_output_folder + "/" + subdir + "/" +
                     p.stem().string() + ".txt";
      std::ofstream of(outfile);
      if (!of.is_open()) {
        std::cout << "Error, Failed to open: " << outfile << std::endl;
        return -1;
      }
      of << p.filename().string() << std::endl;
      of << res_shape[0] << std::endl;  // write face count
      std::cout << p.filename().string() << std::endl;
      for (int i = 0; i < res_shape[0]; i++) {
        std::cout << "Index:" << i << ", score: " << res_data[0] << ", bbox:["
                  << res_data[1] << ", " << res_data[2] << ", " << res_data[3]
                  << ", " << res_data[4] << "], landmarks: [ ";
        std::vector<cv::Point2f> landmarks;
        for (int j = 0; j < res_shape[1] - 5; j += 2) {
          std::cout << "[" << res_data[5 + j] << "," << res_data[5 + j + 1]
                    << "] ";
          landmarks.push_back(
              cv::Point2f(res_data[5 + j], res_data[5 + j + 1]));
        }
        std::cout << "]\n";

        of << res_data[1] << " " << res_data[2] << " " << res_data[3] << " "
           << res_data[4] << " " << res_data[0] << std::endl;

        res_data += res_shape[1];
      }
      of.close();
    }
  }
  return 0;
}