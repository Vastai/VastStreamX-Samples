#include <iomanip>
#include <sstream>

#include "common/bert_base.hpp"
#include "common/cmdline.hpp"
#include "common/utils.hpp"
#include "vaststreamx/vaststreamx.h"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>(
      "model_prefix", 'm', "model prefix of the model suite files", false,
      "/opt/vastai/vaststreamx/data/models/"
      "bert_base_en_qa-384-int8-max-1_384_1_384_1_384-vacc/mod");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "../data/configs/bert_vdsp.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("input_file", '\0', "input npz file", false, "");
  args.add<std::string>("output_file", '\0', "output npz file", false, "");
  args.add<std::string>("dataset_filelist", '\0', "dataset npz filename list",
                        false, "");
  args.add<std::string>("dataset_root", '\0', "dataset root", false, "");
  args.add<std::string>("dataset_output_folder", '\0',
                        "dataset result output folder", false, "");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  const int batch_size = 1;
  auto bert = vsx::Bert(args.get<std::string>("model_prefix"),
                        args.get<std::string>("vdsp_params"), batch_size,
                        args.get<uint32_t>("device_id"),
                        args.get<std::string>("hw_config"));
  std::vector<std::string> filelist;
  std::string output_file;
  std::string dataset_output_folder;

  if (!args.get<std::string>("input_file").empty()) {
    filelist.push_back(args.get<std::string>("input_file"));
    output_file = args.get<std::string>("output_file");
  } else if (!args.get<std::string>("dataset_filelist").empty()) {
    filelist = vsx::ReadFileList(args.get<std::string>("dataset_filelist"));
    dataset_output_folder = args.get<std::string>("dataset_output_folder");
  }
  if (filelist.size() == 0) {
    LOG(ERROR) << "No input_file or dataset_filelist";
    return -1;
  }

  if (filelist.size() == 1) {
    auto tensor_map = vsx::LoadTensorMap(filelist[0]);
    std::vector<vsx::Tensor> input_tensors;
    input_tensors.reserve(tensor_map.size());
    for (size_t i = 0; i < tensor_map.size(); i++) {
      std::stringstream key;
      key << "input_" << i;
      input_tensors.push_back(tensor_map[key.str()]);
    }

    std::vector<vsx::Tensor> result = bert.Process(input_tensors);
    std::unordered_map<std::string, vsx::Tensor> output_map;
    int index = 0;
    for (auto& tensor : result) {
      std::stringstream key;
      key << "output_" << index++;
      output_map[key.str()] = tensor;
    }
    vsx::SaveTensorMap(output_file, output_map);
    std::cout << "write result to: " << output_file << std::endl;
  } else {
    auto dataset_root = args.get<std::string>("dataset_root");
    for (size_t s = 0; s < filelist.size(); s++) {
      auto filename = filelist[s];
      if (!dataset_root.empty()) filename = dataset_root + "/" + filelist[s];
      std::cout << filename << std::endl;
      auto tensor_map = vsx::LoadTensorMap(filename);
      std::vector<vsx::Tensor> input_tensors;
      input_tensors.reserve(tensor_map.size());
      for (size_t i = 0; i < tensor_map.size(); i++) {
        std::stringstream key;
        key << "input_" << i;
        input_tensors.push_back(tensor_map[key.str()]);
      }

      std::vector<vsx::Tensor> result = bert.Process(input_tensors);
      std::unordered_map<std::string, vsx::Tensor> output_map;
      int index = 0;
      for (auto& tensor : result) {
        std::stringstream key;
        key << "output_" << index++;
        output_map[key.str()] = tensor;
      }
      char out_file[20] = {0};
      sprintf(out_file, "/%06lu.npz", s);
      output_file = dataset_output_folder + out_file;
      vsx::SaveTensorMap(output_file, output_map);
    }
  }

  return 0;
}