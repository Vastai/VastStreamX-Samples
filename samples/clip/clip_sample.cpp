
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/clip_model.hpp"
#include "common/cmdline.hpp"
#include "common/file_system.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("imgmod_prefix", '\0',
                        "image model prefix of the model suite files", false,
                        "/opt/vastai/vaststreamx/data/models/"
                        "clip_image-fp16-none-1_3_224_224-vacc/mod");
  args.add<std::string>("imgmod_hw_config", '\0',
                        "hw-config file of the model suite", false, "");
  args.add<std::string>("norm_elf", '\0', "normalize op elf file", false,
                        "/opt/vastai/vaststreamx/data/elf/normalize");
  args.add<std::string>("space2depth_elf", '\0', "space_to_depth op elf file",
                        false,
                        "/opt/vastai/vaststreamx/data/elf/space_to_depth");
  args.add<std::string>("txtmod_prefix", '\0',
                        "model prefix of the model suite files", false,
                        "/opt/vastai/vaststreamx/data/models/"
                        "clip_text-fp16-none-1_77-vacc/mod");
  args.add<std::string>("txtmod_hw_config", '\0',
                        "hw-config file of the model suite", false, "");
  args.add<std::string>("txtmod_vdsp_params", '\0',
                        "vdsp preprocess parameter file", false,
                        "../data/configs/clip_txt_vdsp.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("label_file", '\0', "npz filelist of input strings",
                        false, "");
  args.add<std::string>("npz_files_path", '\0', "npz filelist of input strings",
                        false, "");
  args.add<std::string>("input_file", '\0', "input file", false,
                        "../data/images/CLIP.png");
  args.add<std::string>("dataset_filelist", '\0', "input dataset filelist",
                        false, "");
  args.add<std::string>("dataset_root", '\0', "input dataset root", false, "");
  args.add<std::string>("dataset_output_file", '\0', "dataset output file",
                        false, "");
  args.parse_check(argc, argv);
  return args;
}

std::vector<int> Argsort(const vsx::Tensor& array) {
  const int array_len = array.GetSize();
  const float* array_data = array.Data<float>();
  std::vector<int> array_index(array_len);
  for (int i = 0; i < array_len; ++i) array_index[i] = i;

  std::sort(array_index.begin(), array_index.end(), [=](int pos1, int pos2) {
    return (array_data[pos1] > array_data[pos2]);
  });

  return array_index;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  auto labels = vsx::LoadLabels(args.get<std::string>("label_file"));
  const uint32_t batch_size = 1;
  vsx::ClipModel clip(args.get<std::string>("imgmod_prefix"),
                      args.get<std::string>("norm_elf"),
                      args.get<std::string>("space2depth_elf"),
                      args.get<std::string>("txtmod_prefix"),
                      args.get<std::string>("txtmod_vdsp_params"), batch_size,
                      args.get<uint32_t>("device_id"),
                      args.get<std::string>("imgmod_hw_config"),
                      args.get<std::string>("txtmod_hw_config"));

  std::vector<std::vector<vsx::Tensor>> input_tokens;
  for (auto label : labels) {
    std::filesystem::path p(args.get<std::string>("npz_files_path"));
    p /= label + ".npz";
    auto tensors = vsx::ReadNpzFile(p.string());
    input_tokens.push_back(tensors);
  }

  if (args.get<std::string>("dataset_filelist").empty()) {
    auto image = cv::imread(args.get<std::string>("input_file"));
    auto result = clip.Process(image, input_tokens);
    auto index = Argsort(result);
    const float* array_data = result.Data<float>();
    std::cout << "Top5:\n";
    for (size_t i = 0; i < 5 && i < index.size(); i++) {
      std::cout << i << "th, string: " << labels[index[i]]
                << ", score: " << array_data[index[i]] << std::endl;
    }
  } else {
    auto text_features = clip.ProcessText(input_tokens);
    std::vector<std::string> filelist =
        vsx::ReadFileList(args.get<std::string>("dataset_filelist"));
    auto dataset_root = args.get<std::string>("dataset_root");
    std::ofstream outfile(args.get<std::string>("dataset_output_file"));
    CHECK(outfile.is_open())
        << "Failed to open: " << args.get<std::string>("dataset_output_file");
    for (auto file : filelist) {
      auto fullname = file;
      if (!dataset_root.empty()) fullname = dataset_root + "/" + file;
      std::cout << fullname << std::endl;
      auto image = cv::imread(fullname);
      auto image_feature = clip.ProcessImage(image);
      auto result = clip.PostProcess(image_feature, text_features);
      auto index = Argsort(result);
      const float* array_data = result.Data<float>();
      for (size_t i = 0; i < 5 && i < index.size(); i++) {
        outfile << file << ": "
                << "top-" << i << " id: " << index[i]
                << ", prob: " << std::setprecision(8) << array_data[index[i]]
                << ", class name: " << labels[index[i]] << std::endl;
      }
    }
    outfile.close();
  }
  return 0;
}
