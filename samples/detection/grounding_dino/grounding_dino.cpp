
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <sstream>

#include "common/cmdline.hpp"
#include "common/file_system.hpp"
#include "common/groundingdino.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("txtmod_prefix", '\0',
                        "prefix of the text model suite files", false,
                        "/opt/vastai/vaststreamx/data/models/"
                        "groundingdino/text_encoder-fp16-none-1_195_1_195_1_"
                        "195_1_195_195-vacc/mod");
  args.add<std::string>("txtmod_hw_config", '\0',
                        "hw-config file of the text model suite", false, "");
  args.add<std::string>("txtmod_vdsp_params", '\0',
                        "vdsp preprocess parameter file", false,
                        "data/configs/clip_txt_vdsp.json");
  args.add<std::string>(
      "imgmod_prefix", '\0', "prefix of the image model suite files", false,
      "/opt/vastai/vaststreamx/data/models/"
      "groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod");
  args.add<std::string>("imgmod_hw_config", '\0',
                        "hw-config file of the image model suite", false, "");
  args.add<std::string>("imgmod_vdsp_params", '\0',
                        "image model dsp preprocess parameter file", false,
                        "./data/configs/groundingdino_bgr888.json");
  args.add<std::string>("decmod_prefix", '\0',
                        "model prefix of the decoder model suite files", false,
                        "/opt/vastai/vaststreamx/data/models/"
                        "groundingdino/decoder-fp16-none-1_22223_256_1_195_256_"
                        "1_195_1_195_1_195_195-vacc/mod");
  args.add<std::string>("decmod_hw_config", '\0',
                        "hw-config file of the decoder model suite", false, "");
  args.add<std::string>("npz_file", '\0', "npz file for text model", false, "");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<float>("threshold", '\0', "threshold for detection", false, 0.2);
  args.add<std::string>("label_file", '\0', "label file", false,
                        "./data/labels/coco2id.txt");
  args.add<std::string>("input_file", '\0', "input file", false,
                        "./data/images/dog.jpg");
  args.add<std::string>("output_file", '\0', "output file", false,
                        "grounding_dino_result.jpg");
  args.add<std::string>("dataset_filelist", '\0', "dataset filename list",
                        false, "");
  args.add<std::string>("dataset_root", '\0', "dataset root", false, "");
  args.add<std::string>("dataset_output_folder", '\0',
                        "dataset output folder path", false, "");
  args.add<std::string>("positive_map_file", '\0', "positive map file", false,
                        "../data/bin/positive_map.bin");
  args.parse_check(argc, argv);
  return args;
}

std::vector<int> TokenIDsToClassIDs(const vsx::Tensor& input_ids,
                                    const std::vector<int>& token_ids) {
  std::vector<int> result;
  auto input_ids_data = input_ids.Data<int32_t>();
  std::vector<int> res_token_ids(token_ids);
  int class_id = 0;
  std::vector<int> found_ids;
  for (size_t s = 1; s < input_ids.GetSize(); s++) {
    if (input_ids_data[s] == 101 || input_ids_data[s] == 102 ||
        input_ids_data[s] == 1012) {
      if (found_ids.size() > 0) {
        result.push_back(class_id);
        found_ids.clear();
        if (res_token_ids.size() == 0) {
          break;
        } else {
          class_id = -1;
          s = 1;
        }
      }
      class_id++;
      continue;
    } else {
      if (res_token_ids[0] == input_ids_data[s]) {
        found_ids.push_back(res_token_ids[0]);
        res_token_ids.erase(res_token_ids.begin());
      }
    }
  }
  return result;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  auto label_dict = vsx::LoadLabelDict(args.get<std::string>("label_file"));
  const uint32_t batch_size = 1;
  float threshold = args.get<float>("threshold");

  auto tensor_map = vsx::LoadTensorMap(args.get<std::string>("npz_file"));
  std::vector<vsx::Tensor> input_token;
  for (size_t i = 0; i < 6; i++) {
    std::stringstream key;
    key << "input_" << i;
    input_token.push_back(tensor_map[key.str()]);
  }

  auto attention_mask = tensor_map["attention_mask"];
  auto text_token_mask = tensor_map["text_token_mask"];

  vsx::GroundingDino grounding_dino(args.get<std::string>("txtmod_prefix"),
                                    args.get<std::string>("txtmod_vdsp_params"),
                                    args.get<std::string>("imgmod_prefix"),
                                    args.get<std::string>("imgmod_vdsp_params"),
                                    args.get<std::string>("decmod_prefix"),
                                    batch_size, args.get<uint32_t>("device_id"),
                                    threshold,
                                    args.get<std::string>("txtmod_hw_config"),
                                    args.get<std::string>("imgmod_hw_config"),
                                    args.get<std::string>("decmod_hw_config"),
                                    args.get<std::string>("positive_map_file"));
  auto image_format = grounding_dino.GetFusionOpIimageFormat();
  if (args.get<std::string>("dataset_filelist").empty()) {
    cv::Mat cv_image = cv::imread(args.get<std::string>("input_file"));
    CHECK(!cv_image.empty())
        << "Failed to open: " << args.get<std::string>("input_file");
    vsx::Image vsx_image;
    vsx::MakeVsxImage(cv_image, vsx_image, image_format);
    auto text_result = grounding_dino.ProcessText(input_token);
    auto results = grounding_dino.ProcessImageAndDecode(
        text_result, input_token, attention_mask, text_token_mask, vsx_image);

    std::cout << "Detection objects:\n";
    auto scores = results[0].Data<float>();
    auto bboxes = results[1].Data<float>();
    auto token_ids = results[2].Data<int32_t>();
    for (size_t i = 0; i < results[0].GetSize(); i++) {
      auto score = scores[i];
      auto box = bboxes + i * 4;
      auto id = token_ids[i];
      std::cout << "Object class: " << label_dict[id] << ", score: " << score
                << ", bbox: [" << box[0] << ", " << box[1] << ", " << (box[2])
                << ", " << (box[3]) << "]\n";

      cv::Rect2f rect = {box[0], box[1], (box[2] - box[0]), (box[3] - box[1])};
      cv::rectangle(cv_image, rect, cv::Scalar(0, 0, 255), 1);
    }
    cv::imwrite(args.get<std::string>("output_file"), cv_image);
  } else {
    auto filelist =
        vsx::ReadFileList(args.get<std::string>("dataset_filelist"));
    auto dataset_root = args.get<std::string>("dataset_root");
    auto dataset_output_folder = args.get<std::string>("dataset_output_folder");
    auto text_result = grounding_dino.ProcessText(input_token);
    for (size_t s = 0; s < filelist.size(); s++) {
      auto filename = filelist[s];
      if (!dataset_root.empty()) filename = dataset_root + "/" + filelist[s];
      auto cv_image = cv::imread(filename);
      CHECK(!cv_image.empty())
          << "Failed to read image:" << filename << std::endl;
      vsx::Image vsx_image;
      CHECK(vsx::MakeVsxImage(cv_image, vsx_image, image_format) == 0);
      auto results = grounding_dino.ProcessImageAndDecode(
          text_result, input_token, attention_mask, text_token_mask, vsx_image);

      std::filesystem::path p(filename);
      auto outfile = dataset_output_folder + "/" + p.stem().string() + ".txt";
      std::ofstream of(outfile);
      if (!of.is_open()) {
        std::cout << "Error, Failed to open: " << outfile << std::endl;
        return -1;
      }

      std::cout << p.filename().string() << " detection objects:\n";
      auto scores = results[0].Data<float>();
      auto bboxes = results[1].Data<float>();
      auto token_ids = results[2].Data<int32_t>();
      for (size_t i = 0; i < results[0].GetSize(); i++) {
        auto score = scores[i];
        auto box = bboxes + i * 4;
        auto id = token_ids[i];
        std::cout << "Object class: " << label_dict[id] << ", score: " << score
                  << ", bbox: [" << box[0] << ", " << box[1] << ", " << box[2]
                  << ", " << box[3] << "]\n";
        of << label_dict[id] << " " << score << " " << box[0] << " " << box[1]
           << " " << box[2] << " " << box[3] << std::endl;
      }
      of.close();
    }
  }
}
