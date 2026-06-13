
/*
 * Copyright (C) 2026 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/doc_img_orient_cls.hpp"

#include <sstream>

#include "common/cmdline.hpp"
#include "common/model_profiler.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("model_prefix", 'm',
                        "model prefix of the model suite files", false,
                        "/opt/vastai/vaststreamx/data/models/"
                        "doc_ori_int8_mse_1-3-224-224/mod");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "../data/configs/doc_ori_rgbplanar.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("label_file", '\0', "label file", false, "[0, 180]");
  args.add<std::string>("input_file", '\0', "input image file", false,
                        "../data/images/ppocr-rot.jpg");
  args.add<std::string>("dataset_val_file", '\0', "dataset validation file",
                        false, "");
  args.add<std::string>("dataset_root", '\0', "input dataset root", false, "");
  args.parse_check(argc, argv);
  return args;
}

int get_angle(int index, std::vector<std::vector<int>>& labels) {
  for (auto& label : labels) {
    if (label[0] == index) {
      return label[1];
    }
  }
  std::cerr << "Error: Cann't find index: " << index << " in label file"
            << std::endl;
  return -10000;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  auto lines = vsx::LoadLabels(args.get<std::string>("label_file"));
  std::vector<std::vector<int>> labels;
  for (auto& line : lines) {
    int index, angle;
    std::istringstream iss(line);
    if (iss >> index >> angle)
      labels.push_back({index, angle});
    else {
      std::cerr << "Parsing label file Failed. line:" << line << std::endl;
      return -1;
    }
  }
  const int batch_size = 1;

  vsx::DocImgOrientClassifier model(args.get<std::string>("model_prefix"),
                                    args.get<std::string>("vdsp_params"),
                                    batch_size, args.get<uint32_t>("device_id"),
                                    args.get<std::string>("hw_config"));

  auto image_format = model.GetFusionOpIimageFormat();

  if (args.get<std::string>("dataset_val_file").empty()) {
    vsx::Image vsx_image;
    vsx::MakeVsxImage(args.get<std::string>("input_file"), vsx_image,
                      image_format);
    auto tensor = model.Process(vsx_image);
    int index;
    float score;
    model.PostProcess(tensor, index, score);
    std::cout << "Image angle: " << get_angle(index, labels)
              << ", confidence: " << score << std::endl;
  } else {
    std::vector<std::string> val_lines =
        vsx::ReadFileList(args.get<std::string>("dataset_val_file"));

    std::vector<std::string> filelist;
    std::vector<int> gt_indies;
    for (auto& line : val_lines) {
      std::istringstream iss(line);
      std::string file;
      int index;
      if (iss >> file >> index) {
        filelist.push_back(file);
        gt_indies.push_back(index);
      } else {
        std::cerr << "Parsing dataset val file Failed. line:" << line
                  << std::endl;
        return -1;
      }
    }
    auto dataset_root = args.get<std::string>("dataset_root");
    int correct_count = 0;
    for (size_t i = 0; i < filelist.size(); i++) {
      auto fullname = filelist[i];
      if (!dataset_root.empty()) fullname = dataset_root + "/" + filelist[i];
      std::cout << fullname << std::endl;
      vsx::Image vsx_image;
      vsx::MakeVsxImage(fullname, vsx_image, image_format);
      auto tensor = model.Process(vsx_image);
      int index;
      float score;
      model.PostProcess(tensor, index, score);
      if (index == gt_indies[i]) {
        correct_count++;
      }
      std::cout << "Predicted index: " << index
                << ", GT index: " << gt_indies[i]
                << ", correct_count: " << correct_count << std::endl;
    }

    std::cout << "Accuracy: " << correct_count / float(filelist.size())
              << std::endl;
  }

  return 0;
}