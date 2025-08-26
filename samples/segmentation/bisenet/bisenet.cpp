
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <chrono>

#include "common/cmdline.hpp"
#include "common/file_system.hpp"
#include "common/segmentator.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("model_prefix", 'm',
                        "model prefix of the model suite files", false,
                        "/opt/vastai/vaststreamx/data/models/"
                        "bisenet-int8-kl_divergence-1_3_512_512-vacc/mod");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "../data/configs/bisenet_bgr888.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("input_file", '\0', "input file", false,
                        "../data/images/face.jpg");
  args.add<std::string>("output_file", '\0', "output file", false,
                        "bisenet_result.jpg");
  args.add<std::string>("dataset_filelist", '\0', "input dataset filelist",
                        false, "");
  args.add<std::string>("dataset_root", '\0', "dataset root", false, "");
  args.add<std::string>("dataset_output_folder", '\0', "dataset output folder",
                        false, "");
  args.parse_check(argc, argv);
  return args;
}

std::vector<cv::Vec3b> part_colors = {
    {0, 0, 0},       {255, 85, 0},    {255, 170, 0},  {255, 0, 85},
    {255, 0, 170},   {0, 255, 0},     {85, 255, 0},   {170, 255, 0},
    {0, 255, 85},    {0, 255, 170},   {0, 0, 255},    {85, 0, 255},
    {170, 0, 255},   {0, 85, 255},    {0, 170, 255},  {255, 255, 0},
    {255, 255, 85},  {255, 255, 170}, {255, 0, 255},  {255, 85, 255},
    {255, 170, 255}, {0, 255, 255},   {85, 255, 255}, {170, 255, 255},
};
int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  const int batch_size = 1;
  vsx::Segmentator segment(args.get<std::string>("model_prefix"),
                           args.get<std::string>("vdsp_params"), batch_size,
                           args.get<uint32_t>("device_id"));
  auto image_format = segment.GetFusionOpIimageFormat();

  if (!args.get<std::string>("dataset_filelist").empty()) {
    std::vector<std::string> filelist =
        vsx::ReadFileList(args.get<std::string>("dataset_filelist"));
    CHECK(filelist.size()) << "No file in "
                           << args.get<std::string>("dataset_filelist");
    auto dataset_root = args.get<std::string>("dataset_root");
    auto dataset_output_folder = args.get<std::string>("dataset_output_folder");
    for (size_t s = 0; s < filelist.size(); s++) {
      auto filename = filelist[s];
      if (!dataset_root.empty()) filename = dataset_root + "/" + filelist[s];
      std::cout << filename << std::endl;
      auto cv_image = cv::imread(filename);
      vsx::Image vsx_image;
      CHECK(vsx::MakeVsxImage(cv_image, vsx_image, image_format) == 0);
      auto result = segment.Process(vsx_image);
      std::filesystem::path p(filename);
      std::string basename = p.stem().string();
      auto outfile = dataset_output_folder + "/" + basename + ".npz";
      std::unordered_map<std::string, vsx::Tensor> output_map;
      output_map["output_0"] = result;
      vsx::SaveTensorMap(outfile, output_map);
    }
  } else {
    auto cv_image = cv::imread(args.get<std::string>("input_file"));
    CHECK(!cv_image.empty())
        << "Failed to read image:" << args.get<std::string>("input_file")
        << std::endl;
    vsx::Image vsx_image;
    CHECK(vsx::MakeVsxImage(cv_image, vsx_image, image_format) == 0);
    auto output_tensor = segment.Process(vsx_image);
    auto result = segment.PostProcess(output_tensor);
    auto shape = result.Shape();
    int width = shape[2];
    int height = shape[1];
    cv::Mat out_mat(height, width, CV_8UC3);
    const uchar* type_data = result.Data<uchar>();
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        out_mat.at<cv::Vec3b>(h, w) = part_colors[type_data[h * width + w]];
      }
    }
    cv::resize(out_mat, out_mat, cv_image.size());
    cv::imwrite(args.get<std::string>("output_file"), out_mat);
    std::cout << "Write result to: " << args.get<std::string>("output_file")
              << std::endl;
  }

  return 0;
}
