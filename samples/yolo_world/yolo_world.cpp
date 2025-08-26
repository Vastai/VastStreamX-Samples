
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/yolo_world.hpp"

#include "common/cmdline.hpp"
#include "common/file_system.hpp"
#include "common/json.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>(
      "imgmod_prefix", '\0', "image model prefix of the model suite files",
      false,
      "/opt/vastai/vaststreamx/data/models/"
      "yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod");
  args.add<std::string>("imgmod_hw_config", '\0',
                        "hw-config file of the model suite", false, "");
  args.add<std::string>("imgmod_vdsp_params", '\0',
                        "vdsp preprocess parameter file", false,
                        "../data/configs/yolo_world_1280_1280_bgr888.json");
  args.add<std::string>("txtmod_prefix", '\0',
                        "model prefix of the model suite files", false,
                        "/opt/vastai/vaststreamx/data/models/"
                        "yolo_world_text-fp16-none-1_16_1_16-vacc/mod");
  args.add<std::string>("txtmod_hw_config", '\0',
                        "hw-config file of the model suite", false, "");
  args.add<std::string>("txtmod_vdsp_params", '\0',
                        "vdsp preprocess parameter file", false,
                        "../data/configs/clip_txt_vdsp.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<uint32_t>("max_per_image", '\0', "max_per_image", false, 300);
  args.add<float>("score_thresh", '\0', "threshold for detection", false, 0.5);
  args.add<float>("iou_thresh", '\0', "iou threshold", false, 0.7);
  args.add<uint32_t>("nms_pre", '\0', "nms_pre ", false, 30000);
  args.add<uint32_t>("nms_threads", '\0', "nms_threads", false, 20);
  args.add<std::string>("label_file", '\0', "npz filelist of input strings",
                        false, "");
  args.add<std::string>("npz_files_path", '\0', "npz filelist of input strings",
                        false, "");
  args.add<std::string>("input_file", '\0', "input file", false,
                        "../data/images/CLIP.png");
  args.add<std::string>("output_file", '\0', "output file", false,
                        "yolo_world_result.jpg");
  args.add<std::string>("dataset_filelist", '\0', "input dataset filelist",
                        false, "");
  args.add<std::string>("dataset_root", '\0', "input dataset root", false, "");
  args.add<std::string>("dataset_output_file", '\0', "dataset output file",
                        false, "yolo_world_dataset_output.json");
  args.parse_check(argc, argv);
  return args;
}

std::vector<std::string> ReadLablesFromJson(const std::string& json_file) {
  std::vector<std::string> labels;
  std::ifstream infile(json_file);
  if (!infile.is_open()) {
    LOG(ERROR) << "Failed to open json file: " << json_file;
    return labels;
  }
  nlohmann::json net_json;
  infile >> net_json;
  infile.close();
  for (auto& item : net_json) {
    labels.push_back(item[0]);
  }
  return labels;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  auto labels = ReadLablesFromJson(args.get<std::string>("label_file"));
  const uint32_t batch_size = 1;
  float score_thresh = args.get<float>("score_thresh");
  uint32_t nms_pre = args.get<uint32_t>("nms_pre");
  uint32_t nms_threads = args.get<uint32_t>("nms_threads");
  float iou_thresh = args.get<float>("iou_thresh");
  int max_per_image = args.get<uint32_t>("max_per_image");

  std::vector<std::vector<vsx::Tensor>> input_tokens;
  for (auto label : labels) {
    std::filesystem::path p(args.get<std::string>("npz_files_path"));
    p /= label + ".npz";
    auto tensors = vsx::ReadNpzFile(p.string());
    input_tokens.push_back(tensors);
  }

  vsx::YoloWorld yolo_world(
      args.get<std::string>("imgmod_prefix"),
      args.get<std::string>("imgmod_vdsp_params"),
      args.get<std::string>("txtmod_prefix"),
      args.get<std::string>("txtmod_vdsp_params"), batch_size,
      args.get<uint32_t>("device_id"), score_thresh, nms_pre, iou_thresh,
      max_per_image, nms_threads, args.get<std::string>("imgmod_hw_config"),
      args.get<std::string>("txtmod_hw_config"));
  auto image_format = yolo_world.GetFusionOpIimageFormat();
  if (args.get<std::string>("dataset_filelist").empty()) {
    cv::Mat cv_image = cv::imread(args.get<std::string>("input_file"));
    CHECK(!cv_image.empty())
        << "Failed to open: " << args.get<std::string>("input_file");
    vsx::Image vsx_image;
    vsx::MakeVsxImage(cv_image, vsx_image, image_format);
    auto result = yolo_world.Process(vsx_image, input_tokens);
    if (result.size() == 0) {
      std::cout << "No object detected.\n";
      return 0;
    }
    // label score bbox
    const int* label_data = result[0].Data<int>();
    const float* score_data = result[1].Data<float>();
    const float* box_data = result[2].Data<float>();
    int obj_count = result[0].Shape()[0];
    std::cout << "Detection objects(" << obj_count << "):\n";
    for (int i = 0; i < obj_count; i++) {
      std::string class_name = labels[label_data[i]];
      float score = score_data[i];
      const float* box = box_data + i * 4;
      std::cout << "Object class: " << class_name << ", score: " << score
                << ", bbox: [" << box[0] << ", " << box[1] << ", " << box[2]
                << ", " << box[3] << "]\n";
      cv::rectangle(cv_image, cv::Point2f(box[0], box[1]),
                    cv::Point2f(box[2], box[3]), cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite(args.get<std::string>("output_file"), cv_image);
  } else {
    auto filelist =
        vsx::ReadFileList(args.get<std::string>("dataset_filelist"));
    auto dataset_root = args.get<std::string>("dataset_root");
    std::ofstream of(args.get<std::string>("dataset_output_file"));
    if (!of.is_open()) {
      std::cout << "Error, Failed to open: "
                << args.get<std::string>("dataset_output_file") << std::endl;
      return -1;
    }

    auto text_features = yolo_world.ProcessText(input_tokens);

    auto result_array = nlohmann::json::array();
    for (size_t s = 0; s < filelist.size(); s++) {
      auto filename = filelist[s];
      if (!dataset_root.empty()) filename = dataset_root + "/" + filelist[s];
      std::cout << s << "/" << filelist.size() << ": " << filename << std::endl;
      vsx::Image vsx_image;
      CHECK(vsx::MakeVsxImage(filename, vsx_image, image_format) == 0);
      auto result = yolo_world.ProcessImage(vsx_image, text_features);
      if (result.size() == 0) {
        continue;
      }
      std::filesystem::path p(filename);

      int image_id = std::atoi(p.stem().string().c_str());
      // label score bbox
      const int* label_data = result[0].Data<int>();
      const float* score_data = result[1].Data<float>();
      const float* box_data = result[2].Data<float>();
      int obj_count = result[0].Shape()[0];

      for (int i = 0; i < obj_count; i++) {
        float score = score_data[i];
        const float* box = box_data + i * 4;
        nlohmann::json obj;
        auto jbox = nlohmann::json::array();
        jbox.push_back(box[0]);
        jbox.push_back(box[1]);
        jbox.push_back(box[2] - box[0]);
        jbox.push_back(box[3] - box[1]);

        obj["image_id"] = image_id;
        obj["bbox"] = jbox;
        obj["category_id"] = label_data[i] + 1;
        obj["score"] = score;

        result_array.push_back(obj);
      }
    }
    of << result_array.dump();
    of.close();
  }

  return 0;
}
