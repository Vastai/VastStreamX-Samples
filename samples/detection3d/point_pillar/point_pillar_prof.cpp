
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/cmdline.hpp"
#include "common/detection3d.hpp"
#include "common/model_profiler.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>(
      "model_prefixs", 'm', "model prefix of the model suite files", false,
      "[/opt/vastai/vaststreamx/data/models/"
      "pointpillar-int8-percentile-16000_32_10_3_16000_1_16000-vacc/mod]");
  args.add<std::string>("hw_configs", '\0', "hw-config file of the model suite",
                        false, "[]");
  args.add<std::string>(
      "elf_file", '\0', "elf file path", false,
      "/opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op");
  args.add<std::string>("max_voxel_num", '\0', "model max voxel number", false,
                        "[16000]");
  args.add<uint32_t>("max_points_num", '\0', "max_points_num to run", false,
                     120000);
  args.add<std::string>("voxel_size", '\0', "model max voxel number", false,
                        "[0.16, 0.16, 4]");
  args.add<std::string>("coors_range", '\0', "model max voxel number", false,
                        "[0, -39.68, -3, 69.12, 39.68, 1]");
  args.add<uint32_t>("shuffle_enabled", '\0', "shuffle enabled", false, 0);
  args.add<uint32_t>("normalize_enabled", '\0', "normalize enabled", false, 0);
  args.add<std::string>("feat_size", '\0',
                        "set model feature "
                        "sizes,[max_feature_width,max_feature_height,actual_"
                        "feature_width,actual_feature_height]",
                        false, "[864,496,480,480]");
  args.add<std::string>("dataset_filelist", '\0', "dataset filename list",
                        false, "");
  args.add<std::string>("device_ids", 'd', "device id to run", false, "[0]");
  args.add<uint32_t>("batch_size", 'b', "profiling batch size of the model",
                     false, 1);
  args.add<uint32_t>("instance", 'i',
                     "instance number or range for each device", false, 1);
  args.add<std::string>("shape", 's', "model input shape", true);
  args.add<int>("iterations", '\0', "iterations count for one profiling", false,
                10240);
  args.add<std::string>("percentiles", '\0', "percentiles of latency", false,
                        "[50,90,95,99]");
  args.add<bool>("input_host", '\0', "cache input data into host memory", false,
                 0);
  args.add<uint32_t>("queue_size", 'q', "aync wait queue size", false, 1);
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  auto max_voxel_nums =
      vsx::ParseVecUint(args.get<std::string>("max_voxel_num"));
  auto voxel_size = vsx::ParseVecFloat(args.get<std::string>("voxel_size"));
  auto coors_range = vsx::ParseVecFloat(args.get<std::string>("coors_range"));
  auto model_prefixs =
      vsx::ParseVecString(args.get<std::string>("model_prefixs"));
  auto hw_configs = vsx::ParseVecString(args.get<std::string>("hw_configs"));
  auto elf_file = args.get<std::string>("elf_file");
  auto max_points_num = args.get<uint32_t>("max_points_num");
  auto device_ids = vsx::ParseVecUint(args.get<std::string>("device_ids"));
  auto batch_size = args.get<uint32_t>("batch_size");
  auto instance = args.get<uint32_t>("instance");
  auto iterations = args.get<int>("iterations");
  auto input_host = args.get<bool>("input_host");
  auto queue_size = args.get<uint32_t>("queue_size");
  auto percentiles = vsx::ParseVecUint(args.get<std::string>("percentiles"));
  auto shuffle_enabled = args.get<uint32_t>("shuffle_enabled");
  auto normalize_enabled = args.get<uint32_t>("normalize_enabled");
  auto feat_size = vsx::ParseVecUint(args.get<std::string>("feat_size"));

  CHECK(max_voxel_nums.size() == model_prefixs.size());
  std::vector<vsx::PPModelConfig> model_configs;
  for (size_t i = 0; i < max_voxel_nums.size(); i++) {
    vsx::PPModelConfig config;
    config.max_voxel_num = max_voxel_nums[i];
    config.hw_config = hw_configs.size() > i ? hw_configs[i] : "";
    config.model_prefix = model_prefixs[i];
    model_configs.push_back(std::move(config));
  }

  std::vector<std::shared_ptr<vsx::Detection3D>> models;
  models.reserve(instance);
  std::vector<vsx::Context> contexts;

  for (uint32_t i = 0; i < instance; i++) {
    uint32_t device_id = device_ids[i % (device_ids.size())];
    if (input_host) {
      contexts.push_back(vsx::Context::CPU());
    } else {
      contexts.push_back(vsx::Context::VACC(device_id));
    }
    models.push_back(std::make_shared<vsx::Detection3D>(
        model_configs, elf_file, voxel_size, coors_range, device_id,
        max_points_num, shuffle_enabled, normalize_enabled, feat_size[0],
        feat_size[1], feat_size[2], feat_size[3]));
  }
  vsx::TShape shape = vsx::ParseShape(args.get<std::string>("shape"));

  vsx::ProfilerConfig config = {instance,      iterations,  batch_size,
                                vsx::kFloat16, device_ids,  contexts,
                                {shape},       percentiles, queue_size};
  vsx::ModelProfiler<vsx::Detection3D> profiler(config, models);
  std::cout << profiler.Profiling() << std::endl;
  return 0;
}