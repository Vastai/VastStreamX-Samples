
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/cmdline.hpp"
#include "common/file_system.hpp"
#include "common/super_resolution.hpp"
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
  args.add<std::string>("postproc_elf", '\0', "post process elf file", false,
                        "/opt/vastai/vaststreamx/data/elf/postprocessimage");
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

cv::Mat ConvertRGBPlanarToBGR888(const uchar* rgb_planar_data, int w, int h) {
  cv::Mat bgr888(h, w, CV_8UC3);

  const uchar* r_plane = rgb_planar_data;
  const uchar* g_plane = rgb_planar_data + w * h;
  const uchar* b_plane = rgb_planar_data + 2 * w * h;

  for (int y = 0; y < h; y++) {
    cv::Vec3b* row = bgr888.ptr<cv::Vec3b>(y);
    for (int x = 0; x < w; x++) {
      int idx = y * w + x;
      row[x] = cv::Vec3b(b_plane[idx], g_plane[idx], r_plane[idx]);  // BGR 顺序
    }
  }
  return bgr888;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  const int batch_size = 1;
  const std::vector<float> denorm =
      vsx::ParseVecFloat(args.get<std::string>("denorm"));

  vsx::SuperResolution sr_model(args.get<std::string>("model_prefix"),
                                args.get<std::string>("vdsp_params"),
                                args.get<std::string>("postproc_elf"),
                                args.get<uint32_t>("device_id"), denorm[1],
                                denorm[2], denorm[0], batch_size);
  auto image_format = sr_model.GetFusionOpIimageFormat();

  if (args.get<std::string>("dataset_filelist").empty()) {
    vsx::Image vsx_image;
    CHECK(vsx::MakeVsxImage(args.get<std::string>("input_file"), vsx_image,
                            image_format) == 0);
    auto uint8_tensor = sr_model.Process(vsx_image);
    int width = uint8_tensor.Shape()[2];
    int height = uint8_tensor.Shape()[1];
    cv::Mat bgr888 =
        ConvertRGBPlanarToBGR888(uint8_tensor.Data<uint8_t>(), width, height);
    cv::imwrite(args.get<std::string>("output_file"), bgr888);

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
      auto uint8_tensor = sr_model.Process(vsx_image);
      int width = uint8_tensor.Shape()[2];
      int height = uint8_tensor.Shape()[1];
      cv::Mat bgr888 =
          ConvertRGBPlanarToBGR888(uint8_tensor.Data<uint8_t>(), width, height);
      std::filesystem::path p(fullname);
      auto output_file = args.get<std::string>("dataset_output_folder") + "/" +
                         p.filename().string();
      cv::imwrite(output_file, bgr888);
    }
  }

  return 0;
}
