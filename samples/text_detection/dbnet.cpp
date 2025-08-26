
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/cmdline.hpp"
#include "common/file_system.hpp"
#include "common/utils.hpp"
#include "dbnet_detector/dbnet_detector.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char **argv) {
  cmdline::parser args;
  args.add<std::string>(
      "model_prefix", 'm', "model prefix of the model suite files", false,
      "/opt/vastai/vaststreamx/data/models/"
      "dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "../data/configs/dbnet_rgbplanar.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<float>("threshold", '\0', "threshold for detection", false, 0.3);
  args.add<float>("box_threshold", '\0', "threshold for boxes", false, 0.6);
  args.add<float>("box_unclip_ratio", '\0', "unclip ratio", false, 1.5);
  args.add<bool>("use_polygon_score", '\0', "use_polygon_score in postprocess",
                 false, false);
  args.add<std::string>(
      "elf_file", '\0', "elf file path", false,
      "/opt/vastai/vaststreamx/data/elf/find_contours_ext_op");
  args.add<std::string>("input_file", '\0', "input image file", false,
                        "../data/images/detect.jpg");
  args.add<std::string>("output_file", '\0', "output image file", false, "");
  args.add<std::string>("dataset_filelist", '\0', "input dataset filelist",
                        false, "");
  args.add<std::string>("dataset_root", '\0', "input dataset root", false, "");
  args.add<std::string>("dataset_output_folder", '\0', "dataset output folder",
                        false, "");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char **argv) {
  auto args = ArgumentParser(argc, argv);
  const int batch_size = 1;

  vsx::DBnetDetector detector(args.get<std::string>("model_prefix"),
                              args.get<std::string>("vdsp_params"),
                              args.get<std::string>("elf_file"), batch_size,
                              args.get<uint32_t>("device_id"));
  detector.SetThreshold(args.get<float>("threshold"));
  detector.SetBoxThreshold(args.get<float>("box_threshold"));
  detector.SetPolygonScoreUsage(args.get<bool>("use_polygon_score"));
  detector.SetBoxUnclipRatio(args.get<float>("box_unclip_ratio"));
  auto image_format = detector.GetFusionOpIimageFormat();

  if (args.get<std::string>("dataset_filelist").empty()) {
    vsx::Image vsx_image;
    vsx::MakeVsxImage(args.get<std::string>("input_file"), vsx_image,
                      image_format);
    auto res = detector.Process(vsx_image);
    if (res.GetSize() == 0) {
      std::cout << "No object detected in image.\n";
    } else {
      // print result and draw rectangle
      auto cv_image = cv::imread(args.get<std::string>("input_file"));
      int obj_count = res.Shape()[0];
      const float *data = res.Data<float>();
      for (int i = 0; i < obj_count; i++) {
        std::cout << "index:" << i << ", score:" << data[i * 9 + 0]
                  << ",bbox:[ [" << static_cast<int>(data[i * 9 + 1]) << " "
                  << static_cast<int>(data[i * 9 + 2]) << "] ["
                  << static_cast<int>(data[i * 9 + 3]) << " "
                  << static_cast<int>(data[i * 9 + 4]) << "] ["
                  << static_cast<int>(data[i * 9 + 5]) << " "
                  << static_cast<int>(data[i * 9 + 6]) << "] ["
                  << static_cast<int>(data[i * 9 + 7]) << " "
                  << static_cast<int>(data[i * 9 + 8]) << "] ]\n";
        cv::line(cv_image, cv::Point2f(data[i * 9 + 1], data[i * 9 + 2]),
                 cv::Point2f(data[i * 9 + 3], data[i * 9 + 4]),
                 cv::Scalar(0, 0, 255));
        cv::line(cv_image, cv::Point2f(data[i * 9 + 3], data[i * 9 + 4]),
                 cv::Point2f(data[i * 9 + 5], data[i * 9 + 6]),
                 cv::Scalar(0, 0, 255));
        cv::line(cv_image, cv::Point2f(data[i * 9 + 5], data[i * 9 + 6]),
                 cv::Point2f(data[i * 9 + 7], data[i * 9 + 8]),
                 cv::Scalar(0, 0, 255));
        cv::line(cv_image, cv::Point2f(data[i * 9 + 1], data[i * 9 + 2]),
                 cv::Point2f(data[i * 9 + 7], data[i * 9 + 8]),
                 cv::Scalar(0, 0, 255));
      }

      // save file
      if (!args.get<std::string>("output_file").empty()) {
        cv::imwrite(args.get<std::string>("output_file"), cv_image);
      }
    }

  } else {
    std::vector<std::string> filelist =
        vsx::ReadFileList(args.get<std::string>("dataset_filelist"));
    auto dataset_root = args.get<std::string>("dataset_root");
    for (auto &file : filelist) {
      auto fullname = file;
      if (!dataset_root.empty()) fullname = dataset_root + "/" + file;
      std::cout << fullname << std::endl;
      vsx::Image vsx_image;
      vsx::MakeVsxImage(fullname, vsx_image, image_format);
      auto res = detector.Process(vsx_image);
      vsx::Tensor output_tensor;
      if (res.GetSize() == 0) {
        output_tensor = vsx::Tensor(vsx::TShape({0}), vsx::Context::CPU(),
                                    vsx::TypeFlag::kUint32);
      } else {
        int obj_count = res.Shape()[0];
        output_tensor =
            vsx::Tensor(vsx::TShape({obj_count, 4, 2}), vsx::Context::CPU(),
                        vsx::TypeFlag::kUint32);
        const float *src = res.Data<float>();
        uint32_t *dst = output_tensor.MutableData<uint32_t>();
        for (int i = 0; i < obj_count; i++) {
          for (int j = 0; j < 8; j++) {
            dst[i * 8 + j] = static_cast<uint32_t>(src[i * 9 + j + 1]);
          }
        }
      }

      std::filesystem::path p(fullname);
      auto output_file = args.get<std::string>("dataset_output_folder") + "/" +
                         p.stem().string() + ".npz";
      std::cout << output_file << std::endl;
      std::unordered_map<std::string, vsx::Tensor> output_map;
      output_map["output_0"] = output_tensor;
      vsx::SaveTensorMap(output_file, output_map);
    }
  }

  return 0;
}