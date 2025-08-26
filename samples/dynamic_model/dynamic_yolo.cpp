
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/cmdline.hpp"
#include "common/dynamic_detector.hpp"
#include "common/file_system.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("module_info", 'm', "model info json files", false,
                        "/opt/vastai/vaststreamx/data/models/"
                        "torch-yolov5s_coco-int8-percentile-Y-Y-2-none/"
                        "yolov5s_coco_module_info.json");
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "./data/configs/yolo_div255_bgr888.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("max_input_shape", '\0', "model max input shape", false,
                        "[1,3,640,640]");
  args.add<float>("threshold", '\0', "threshold for detection", false, 0.5);
  args.add<std::string>("label_file", '\0', "label file", false,
                        "../data/labels/coco2id.txt");
  args.add<std::string>("input_file", '\0', "input file", false,
                        "../data/images/dog.jpg");
  args.add<std::string>("output_file", '\0', "output file", false,
                        "dynamic_result.jpg");
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
  auto max_input_shape =
      vsx::ParseShape(args.get<std::string>("max_input_shape"));
  auto labels = vsx::LoadLabels(args.get<std::string>("label_file"));
  const int batch_size = 1;

  vsx::DynamicDetector dynamic_model(args.get<std::string>("module_info"),
                                     args.get<std::string>("vdsp_params"),
                                     {max_input_shape}, batch_size,
                                     args.get<uint32_t>("device_id"));
  dynamic_model.SetThreshold(args.get<float>("threshold"));
  auto image_format = dynamic_model.GetFusionOpIimageFormat();
  // max input size 640, min input size 320
  int model_min_input_size = 320;
  int model_max_input_size = 640;
  if (args.get<std::string>("dataset_filelist").empty()) {
    auto cv_image = cv::imread(args.get<std::string>("input_file"));
    CHECK(!cv_image.empty())
        << "Failed to read image:" << args.get<std::string>("input_file")
        << std::endl;
    int ori_w = cv_image.cols, ori_h = cv_image.rows;
    int input_size = ori_w > ori_h ? ori_w : ori_h;
    if (input_size % 2 != 0) input_size += 1;
    if (input_size < model_min_input_size)
      input_size = model_min_input_size;
    else if (input_size > model_max_input_size)
      input_size = model_max_input_size;
    dynamic_model.SetInputShape({{1, 3, input_size, input_size}});

    vsx::Image vsx_image;
    CHECK(vsx::MakeVsxImage(cv_image, vsx_image, image_format) == 0);
    auto result = dynamic_model.Process(vsx_image);
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
      int ori_w = cv_image.cols, ori_h = cv_image.rows;
      int input_size = ori_w > ori_h ? ori_w : ori_h;
      if (input_size % 2 != 0) input_size += 1;
      if (input_size < model_min_input_size)
        input_size = model_min_input_size;
      else if (input_size > model_max_input_size)
        input_size = model_max_input_size;
      dynamic_model.SetInputShape({{1, 3, input_size, input_size}});
      vsx::Image vsx_image;
      CHECK(vsx::MakeVsxImage(cv_image, vsx_image, image_format) == 0);
      auto result = dynamic_model.Process(vsx_image);
      auto res_shape = result.Shape();
      const float* res_data = result.Data<float>();

      std::filesystem::path p(filename);
      auto outfile = dataset_output_folder + "/" + p.stem().string() + ".txt";
      std::ofstream of(outfile);
      if (!of.is_open()) {
        std::cout << "Error, Failed to open: " << outfile << std::endl;
        return -1;
      }

      std::cout << p.filename().string() << " Detection objects:\n";
      for (int i = 0; i < res_shape[0]; i++) {
        if (res_data[0] < 0) break;
        std::string class_name = labels[static_cast<int>(res_data[0])];
        float score = res_data[1];
        std::cout << "Object class: " << class_name << ", score: " << score
                  << ", bbox: [" << res_data[2] << ", " << res_data[3] << ", "
                  << res_data[4] << ", " << res_data[5] << "]\n";
        of << class_name << " " << score << " " << res_data[2] << " "
           << res_data[3] << " " << (res_data[2] + res_data[4]) << " "
           << (res_data[3] + res_data[5]) << std::endl;
        res_data += vsx::kDetectionOffset;
      }
      of.close();
    }
  }
  return 0;
}