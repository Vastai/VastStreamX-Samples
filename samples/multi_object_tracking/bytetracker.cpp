
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "bytetracker.hpp"

#include "common/cmdline.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>(
      "model_prefix", 'm', "model prefix of the model suite files", false,
      "/opt/vastai/vaststreamx/data/models/"
      "bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false, "");
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "../data/configs/bytetrack_rgbplanar.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<float>("det_threshold", '\0', "threshold for detection", false, 0.2);
  args.add<uint32_t>("track_buffer", '\0', "buffer for track", false, 30);
  args.add<float>("track_thresh", '\0', "threshold for track", false, 0.6);
  args.add<std::string>("label_file", '\0', "label file", false,
                        "../data/labels/coco2id.txt");
  args.add<std::string>("input_file", '\0', "input file", false, "");
  args.add<std::string>("output_file", '\0', "output file", false,
                        "mot_result.jpg");
  args.add<std::string>("dataset_filelist", '\0', "dataset image file list ",
                        false, "");
  args.add<std::string>("dataset_root", '\0', "dataset root", false, "");
  args.add<std::string>("dataset_result_file", '\0', "dataset result file",
                        false, "");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  auto labels = vsx::LoadLabels(args.get<std::string>("label_file"));
  auto det_threshold = args.get<float>("det_threshold");
  auto track_buffer = args.get<uint32_t>("track_buffer");
  auto track_thresh = args.get<float>("track_thresh");
  auto hw_config = args.get<std::string>("hw_config");

  const int batch_size = 1;
  int fps = 30;

  vsx::ByteTracker tracker(args.get<std::string>("model_prefix"),
                           args.get<std::string>("vdsp_params"), batch_size,
                           args.get<uint32_t>("device_id"), det_threshold,
                           track_buffer, track_thresh, fps, hw_config);

  auto image_format = tracker.GetFusionOpIimageFormat();

  if (!args.get<std::string>("input_file").empty()) {
    auto cv_image = cv::imread(args.get<std::string>("input_file"));
    CHECK(!cv_image.empty())
        << "Failed to read: " << args.get<std::string>("input_file")
        << std::endl;
    vsx::Image vsx_image;
    vsx::MakeVsxImage(cv_image, vsx_image, image_format);
    auto result = tracker.Process(vsx_image);
    auto res_shape = result.Shape();
    const float* res_data = result.Data<float>();
    for (int i = 0; i < res_shape[0]; i++) {
      if (res_data[0] < 0) break;
      std::string class_name = labels[static_cast<int>(res_data[0])];
      float score = res_data[1];
      std::cout << "detected object class: " << class_name
                << ", score: " << score << ", id: " << res_data[6]
                << ", bbox: [" << res_data[2] << ", " << res_data[3] << ", "
                << res_data[4] << ", " << res_data[5] << "]\n";
      cv::Rect2f rect = {res_data[2], res_data[3], res_data[4], res_data[5]};
      cv::rectangle(cv_image, rect, cv::Scalar(0, 255, 0), 2);
      res_data += vsx::kDetectionOffset;
    }
    if (!args.get<std::string>("output_file").empty()) {
      cv::imwrite(args.get<std::string>("output_file"), cv_image);
    }
  } else if (!args.get<std::string>("dataset_filelist").empty()) {
    auto filelist =
        vsx::ReadFileList(args.get<std::string>("dataset_filelist"));
    auto dataset_root = args.get<std::string>("dataset_root");
    std::ofstream result_file;
    result_file.open(args.get<std::string>("dataset_result_file"));
    CHECK(result_file.is_open())
        << "Failed to open dataset_result_file:"
        << args.get<std::string>("dataset_result_file");
    for (size_t s = 0; s < filelist.size(); s++) {
      auto filename = filelist[s];
      if (!dataset_root.empty()) filename = dataset_root + "/" + filelist[s];
      std::cout << filename << std::endl;
      vsx::Image vsx_image;
      vsx::MakeVsxImage(filename, vsx_image, image_format);
      auto result = tracker.Process(vsx_image);
      auto res_shape = result.Shape();
      const float* res_data = result.Data<float>();
      for (int i = 0; i < res_shape[0]; i++) {
        if (res_data[0] < 0) break;
        std::string class_name = labels[static_cast<int>(res_data[0])];
        float score = res_data[1];
        // '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
        result_file << s + 1 << "," << static_cast<int>(res_data[6]) << ","
                    << std::setiosflags(std::ios::fixed) << std::setprecision(2)
                    << res_data[2] << "," << res_data[3] << "," << res_data[4]
                    << "," << res_data[5] << "," << score << ",-1,-1,-1\n";
        res_data += vsx::kDetectionOffset;
      }
    }
    result_file.close();

  } else {
    LOG(ERROR) << "No input_file or dataset_filelist";
    return -1;
  }

  return 0;
}