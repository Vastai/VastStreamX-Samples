
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common/cmdline.hpp"
#include "common/detection3d.hpp"
#include "common/file_system.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>(
      "model_prefixs", 'm', "model prefixs of the model suite files", false,
      "[/opt/vastai/vaststreamx/data/models/"
      "pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]");
  args.add<std::string>("hw_configs", '\0', "hw-config file of the model suite",
                        false, "[]");
  args.add<std::string>(
      "elf_file", '\0', "elf file path", false,
      "/opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<uint32_t>("max_points_num", '\0', "max_points_num to run", false,
                     2000000);
  args.add<std::string>("max_voxel_num", '\0', "model max voxel number", false,
                        "[32000]");
  args.add<std::string>("voxel_size", '\0', "model max voxel number", false,
                        "[0.32,0.32,4.2]");
  args.add<std::string>("coors_range", '\0', "model max voxel number", false,
                        "[-50,-103.6,-0.1,103.6,50,4.1]");
  args.add<std::string>("input_file", '\0', "input file", false, "");
  args.add<uint32_t>("shuffle_enabled", '\0', "shuffle enabled", false, 1);
  args.add<uint32_t>("normalize_enabled", '\0', "normalize enabled", false, 1);
  args.add<std::string>("dataset_root", '\0', "dataset root", false, "");
  args.add<std::string>("dataset_output_folder", '\0',
                        "dataset output folder path", false, "");
  args.parse_check(argc, argv);
  return args;
}

vsx::Tensor ReadBinFile(const std::string& binfile, int max_points_num) {
  std::ifstream ifile(binfile.c_str(), std::ios::binary);
  CHECK(ifile.is_open()) << "Failed to open:" << binfile << std::endl;
  ifile.seekg(0, std::ios::end);
  size_t file_len = ifile.tellg();
  ifile.seekg(0, std::ios::beg);
  size_t read_size = file_len < max_points_num * sizeof(uint16_t) * 4
                         ? file_len
                         : max_points_num * sizeof(uint16_t) * 4;
  char* buffer = new char[read_size];
  ifile.read(buffer, read_size);
  ifile.close();
  size_t fp16_size = read_size / sizeof(uint16_t);
  auto data_manager = std::make_shared<vsx::DataManager>(
      read_size, vsx::Context::CPU(), reinterpret_cast<uint64_t>(buffer),
      [](void* ptr) { delete[] reinterpret_cast<char*>(ptr); });
  vsx::Tensor tensor(vsx::TShape({static_cast<int64_t>(fp16_size)}),
                     data_manager, vsx::TypeFlag::kFloat16);
  return std::move(tensor);
}

uint16_t* LoadDataFP32(std::string filePath, int& size) {
  std::ifstream ifile(filePath.c_str(), std::ios::binary);
  int l = ifile.tellg();
  ifile.seekg(0, std::ios::end);
  int m = ifile.tellg();
  size = static_cast<int>((m - l));
  float* hostSpace = reinterpret_cast<float*>(malloc(size));
  ifile.seekg(0, std::ios_base::beg);
  ifile.read(reinterpret_cast<char*>(hostSpace), size);

  uint16_t* hostSpaceFp16 = reinterpret_cast<uint16_t*>(malloc(size / 2));
  for (int i = 0; i < static_cast<int>(size / 4); i++) {
    hostSpaceFp16[i] = vsx::FloatToHalf(hostSpace[i]);
  }
  free(hostSpace);
  size /= 4;
  ifile.close();
  return hostSpaceFp16;
}

vsx::Tensor ReadBinFileFromFP32(const std::string& binfile,
                                int max_points_num) {
  std::ifstream ifile(binfile.c_str(), std::ios::binary);
  CHECK(ifile.is_open()) << "Failed to open:" << binfile << std::endl;
  ifile.seekg(0, std::ios::end);
  size_t file_len = ifile.tellg();
  ifile.seekg(0, std::ios::beg);
  size_t read_size = file_len < max_points_num * sizeof(uint32_t) * 4
                         ? file_len
                         : max_points_num * sizeof(uint32_t) * 4;
  char* buffer = new char[read_size];
  ifile.read(buffer, read_size);
  ifile.close();
  // size_t fp16_size = read_size / sizeof(uint16_t);
  size_t tensor_size = read_size / sizeof(uint32_t);
  auto data_manager = std::make_shared<vsx::DataManager>(
      read_size, vsx::Context::CPU(), reinterpret_cast<uint64_t>(buffer),
      [](void* ptr) { delete[] reinterpret_cast<char*>(ptr); });
  vsx::Tensor fp32_tensor(vsx::TShape({static_cast<int64_t>(tensor_size)}),
                          data_manager, vsx::TypeFlag::kFloat32);

  vsx::Tensor fp16_tensor =
      vsx::Tensor(vsx::TShape({static_cast<int64_t>(tensor_size)}),
                  vsx::Context::CPU(0), vsx::TypeFlag::kUint16);
  float* fp32 = fp32_tensor.MutableData<float>();
  uint16_t* fp16 = fp16_tensor.MutableData<uint16_t>();
  for (int i = 0; i < tensor_size; i++) {
    fp16[i] = vsx::FloatToHalf(fp32[i]);
  }
  return std::move(fp16_tensor);
}

void WriteBinFile(const std::string& binfile, const vsx::Tensor& tensor) {
  std::ofstream ofile(binfile.c_str(), std::ios::binary);
  CHECK(ofile.is_open()) << "Failed to open:" << binfile << std::endl;
  const char* data = tensor.Data<char>();
  size_t bytes = tensor.GetDataBytes();
  ofile.write(data, bytes);
  ofile.close();
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  const int batch_size = 1;
  auto max_voxel_nums =
      vsx::ParseVecUint(args.get<std::string>("max_voxel_num"));
  auto voxel_size = vsx::ParseVecFloat(args.get<std::string>("voxel_size"));

  auto coors_range = vsx::ParseVecFloat(args.get<std::string>("coors_range"));
  auto model_prefixs =
      vsx::ParseVecString(args.get<std::string>("model_prefixs"));
  auto hw_configs = vsx::ParseVecString(args.get<std::string>("hw_configs"));
  CHECK(max_voxel_nums.size() == model_prefixs.size());
  auto max_points_num = args.get<uint32_t>("max_points_num");
  auto shuffle_enabled = args.get<uint32_t>("shuffle_enabled");
  auto normalize_enabled = args.get<uint32_t>("normalize_enabled");

  std::vector<vsx::PPModelConfig> model_configs;
  for (size_t i = 0; i < max_voxel_nums.size(); i++) {
    vsx::PPModelConfig config;
    config.max_voxel_num = max_voxel_nums[i];
    config.hw_config = hw_configs.size() > i ? hw_configs[i] : "";
    config.model_prefix = model_prefixs[i];
    model_configs.push_back(std::move(config));
  }

  vsx::Detection3D detector3d(
      model_configs, args.get<std::string>("elf_file"), voxel_size, coors_range,
      args.get<uint32_t>("device_id"), max_points_num, shuffle_enabled,
      normalize_enabled, 480, 480, 480, 480);

  if (!args.get<std::string>("input_file").empty()) {
    vsx::Tensor input_tensor;
    std::filesystem::path p(args.get<std::string>("input_file"));

    input_tensor = ReadBinFileFromFP32(args.get<std::string>("input_file"),
                                       max_points_num);

    auto tensors_fp16 = detector3d.Process(input_tensor);
    std::vector<vsx::Tensor> tensors_fp32 =
        vsx::ConvertTensorFromFp16ToFp32(tensors_fp16);
    auto scores = tensors_fp32[0].Data<float>();
    auto labels = tensors_fp32[1].Data<float>();
    auto boxes = tensors_fp32[2].Data<float>();
    for (int i = 0; i < 500; i++) {
      if (scores[i] < 0) break;
      std::cout << "label: " << labels[i] << ", score: " << scores[i]
                << ", box:[ ";
      for (int j = 0; j < 7; j++) {
        std::cout << boxes[i * 7 + j] << " ";
      }
      std::cout << "]\n";
    }
  } else {
    std::vector<std::string> filelist;

    // 使用C++17的filesystem库来遍历目录
    auto dataset_root = args.get<std::string>("dataset_root");
    auto dataset_output_folder = args.get<std::string>("dataset_output_folder");
    for (const auto& entry :
         std::filesystem::directory_iterator(dataset_root)) {
      if (entry.is_regular_file() && entry.path().extension() == ".bin") {
        // 将符合条件的文件路径添加到filelist中
        filelist.push_back(entry.path().string());
      }
    }

    std::sort(filelist.begin(), filelist.end());

    for (int s = 0; s < filelist.size(); s++) {
      auto filename = filelist[s];
      std::cout << filename << std::endl;
      vsx::Tensor input_tensor;

      input_tensor = ReadBinFileFromFP32(filename, max_points_num);
      auto output_tensors = detector3d.Process(input_tensor);
      std::filesystem::path p(filename);

      std::vector<vsx::Tensor> tensors_fp32 =
          vsx::ConvertTensorFromFp16ToFp32(output_tensors);
      std::string npz_file =
          dataset_output_folder + "/" + p.stem().string() + ".npz";
      vsx::SaveTensorMap(npz_file, {{"score", tensors_fp32[0]},
                                    {"label", tensors_fp32[1]},
                                    {"boxes", tensors_fp32[2]}});
    }
  }

  return 0;
}