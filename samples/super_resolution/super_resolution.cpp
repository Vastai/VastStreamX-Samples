
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
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
                        "rcan-int8-max-1_3_1080_1920-vacc/mod");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "../data/configs/rcan_bgr888.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("denorm", '\0',
                        "denormalization paramsters [mean, std, scale]", false,
                        "[0, 1, 1]");
  args.add<std::string>("input_file", '\0', "input image", false,
                        "../data/images/hd_1920x1080.png");
  args.add<std::string>("output_file", '\0', "output image", false,
                        "sr_result.jpg");
  args.add<std::string>("dataset_filelist", '\0', "input dataset filelist",
                        false, "");
  args.add<std::string>("dataset_root", '\0', "input dataset root", false, "");
  args.add<std::string>("dataset_output_folder", '\0', "dataset output folder",
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
  cv::Mat mat(height, width, CV_8UC3);
  const float* src_r = fp32_tensor.Data<float>();
  const float* src_g = src_r + width * height;
  const float* src_b = src_g + width * height;
  uint8_t* dst = mat.data;
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      int b = static_cast<int>(
          floor(((src_b[h * width + w] * std) + mu) * scale + 0.5));
      dst[0] = static_cast<uchar>(b < 0 ? 0 : b > 255 ? 255 : b);
      int g = static_cast<int>(
          floor(((src_g[h * width + w] * std) + mu) * scale + 0.5));
      dst[1] = static_cast<uchar>(g < 0 ? 0 : g > 255 ? 255 : g);
      int r = static_cast<int>(
          floor(((src_r[h * width + w] * std) + mu) * scale + 0.5));
      dst[2] = static_cast<uchar>(r < 0 ? 0 : r > 255 ? 255 : r);
      dst += 3;
    }
  }
  cv::imwrite(filename, mat);
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  const int batch_size = 1;
  const std::vector<float> denorm =
      vsx::ParseVecFloat(args.get<std::string>("denorm"));

  vsx::ModelCV sr_model(args.get<std::string>("model_prefix"),
                        args.get<std::string>("vdsp_params"), batch_size,
                        args.get<uint32_t>("device_id"));
  auto image_format = sr_model.GetFusionOpIimageFormat();

  if (args.get<std::string>("dataset_filelist").empty()) {
    vsx::Image vsx_image;
    CHECK(vsx::MakeVsxImage(args.get<std::string>("input_file"), vsx_image,
                            image_format) == 0);
    auto fp16_tensor = sr_model.Process(vsx_image);
    WriteToFile(fp16_tensor, denorm, args.get<std::string>("output_file"));

  } else {
    std::vector<std::string> filelist =
        vsx::ReadFileList(args.get<std::string>("dataset_filelist"));
    auto dataset_root = args.get<std::string>("dataset_root");

    for (size_t s = 0; s < filelist.size(); s++) {
      auto fullname = filelist[s];
      if (!dataset_root.empty()) fullname = dataset_root + "/" + fullname;
      std::cout << fullname << std::endl;
      vsx::Image vsx_image;
      CHECK(vsx::MakeVsxImage(fullname, vsx_image, image_format) == 0);
      auto fp16_tensor = sr_model.Process(vsx_image);
      std::filesystem::path p(fullname);
      auto output_file = args.get<std::string>("dataset_output_folder") + "/" +
                         p.filename().string();
      WriteToFile(fp16_tensor, denorm, output_file);
    }
  }

  return 0;
}
