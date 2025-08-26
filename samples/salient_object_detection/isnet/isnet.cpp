
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
#include "common/model_cv.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("model_prefix", 'm',
                        "model prefix of the model suite files", false,
                        "/opt/vastai/vaststreamx/data/models/"
                        "/isnet-int8-kl_divergence-1_3_320_320-vacc/mod");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "../data/configs/isnet_bgr888.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("input_file", '\0', "input file", false,
                        "../data/images/cat.jpg");
  args.add<std::string>("output_file", '\0', "output file", false,
                        "./isnet_result.png");
  args.add<std::string>("dataset_filelist", '\0', "input dataset filelist",
                        false, "");
  args.add<std::string>("dataset_root", '\0', "input dataset root", false, "");
  args.add<std::string>("dataset_output_folder", '\0', "dataset output file",
                        false, "");
  args.parse_check(argc, argv);
  return args;
}

void WriteToFile(const vsx::Tensor& fp16_tensor,
                 const std::vector<float>& denorm,
                 const std::string& filename) {
  vsx::Tensor fp32_tensor = vsx::ConvertTensorFromFp16ToFp32(fp16_tensor);
  float mu = denorm[0];
  float std = denorm[1];
  float scale = denorm[2];
  int width = fp32_tensor.Shape()[3];
  int height = fp32_tensor.Shape()[2];
  cv::Mat mat(height, width, CV_8UC1);
  const float* src_r = fp32_tensor.Data<float>();
  uint8_t* dst = mat.data;
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      int r = static_cast<int>(
          floor(((src_r[h * width + w] * std) + mu) * scale + 0.5));
      dst[0] = (uchar)(r < 0 ? 0 : r > 255 ? 255 : r);
      dst++;
    }
  }
  cv::imwrite(filename, mat);
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  const int batch_size = 1;
  vsx::ModelCV model(args.get<std::string>("model_prefix"),
                     args.get<std::string>("vdsp_params"), batch_size,
                     args.get<uint32_t>("device_id"));
  auto image_format = model.GetFusionOpIimageFormat();

  if (args.get<std::string>("dataset_filelist").empty()) {
    auto cv_image = cv::imread(args.get<std::string>("input_file"));
    CHECK(!cv_image.empty())
        << "Failed to read image:" << args.get<std::string>("input_file")
        << std::endl;
    vsx::Image vsx_image;
    CHECK(vsx::MakeVsxImage(cv_image, vsx_image, image_format) == 0);
    auto fp16_tensor = model.Process(vsx_image);
    WriteToFile(fp16_tensor, {0.0f, 1.0f, 255.0f},
                args.get<std::string>("output_file"));
  } else {
    std::vector<std::string> filelist =
        vsx::ReadFileList(args.get<std::string>("dataset_filelist"));
    auto dataset_root = args.get<std::string>("dataset_root");
    for (auto file : filelist) {
      auto fullname = file;
      if (!dataset_root.empty()) fullname = dataset_root + "/" + file;
      auto cv_image = cv::imread(fullname);
      CHECK(!cv_image.empty())
          << "Failed to read image:" << fullname << std::endl;
      vsx::Image vsx_image;
      CHECK(vsx::MakeVsxImage(cv_image, vsx_image, image_format) == 0);
      auto fp16_tensor = model.Process(vsx_image);
      std::filesystem::path p(fullname);
      auto output_file = args.get<std::string>("dataset_output_folder") + "/" +
                         p.stem().string() + ".png";
      WriteToFile(fp16_tensor, {0.0f, 1.0f, 255.0f}, output_file);
    }
  }

  return 0;
}