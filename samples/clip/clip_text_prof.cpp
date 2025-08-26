
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/clip_text.hpp"
#include "common/cmdline.hpp"
#include "common/model_profiler.hpp"
#include "common/utils.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("model_prefix", 'm',
                        "model prefix of the model suite files", false,
                        "/opt/vastai/vaststreamx/data/models/"
                        "clip_text-fp16-none-1_77-vacc/mod");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false, "");
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "../data/configs/clip_txt_vdsp.json");
  args.add<std::string>("device_ids", 'd', "device id to run", false, "[0]");
  args.add<uint32_t>("batch_size", 'b', "profiling batch size of the model",
                     false, 1);
  args.add<uint32_t>("instance", 'i',
                     "instance number or range for each device", false, 1);
  args.add<int>("iterations", '\0', "iterations count for one profiling", false,
                1024);
  args.add<std::string>("percentiles", '\0', "percentiles of latency", false,
                        "[50, 90, 95, 99]");
  args.add<bool>("input_host", '\0', "cache input data into host memory", false,
                 0);
  args.add<uint32_t>("queue_size", 'q', "aync wait queue size", false, 2);
  args.add<std::string>("test_input_npz", '\0', "test input npz file", true);
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  auto model_prefix = args.get<std::string>("model_prefix");
  auto vdsp_params = args.get<std::string>("vdsp_params");
  auto hw_config = args.get<std::string>("hw_config");
  auto device_ids = vsx::ParseVecUint(args.get<std::string>("device_ids"));
  auto batch_size = args.get<uint32_t>("batch_size");
  auto instance = args.get<uint32_t>("instance");
  auto iterations = args.get<int>("iterations");
  auto input_host = args.get<bool>("input_host");
  auto queue_size = args.get<uint32_t>("queue_size");
  auto percentiles = vsx::ParseVecUint(args.get<std::string>("percentiles"));

  std::vector<std::shared_ptr<vsx::ClipText>> models;
  models.reserve(instance);
  std::vector<vsx::Context> contexts;
  for (uint32_t i = 0; i < instance; i++) {
    uint32_t device_id = device_ids[i % (device_ids.size())];
    if (input_host) {
      contexts.push_back(vsx::Context::CPU());
    } else {
      contexts.push_back(vsx::Context::VACC(device_id));
    }
    models.push_back(std::make_shared<vsx::ClipText>(
        model_prefix, vdsp_params, batch_size, device_id, hw_config));
  }

  auto input_tensors =
      vsx::ReadNpzFile(args.get<std::string>("test_input_npz"));

  for (auto model : models) {
    model->SetCPUTestData(input_tensors);
  }
  uint32_t input_count;
  models[0]->GetInputCount(input_count);
  std::vector<vsx::TShape> input_shapes;
  input_shapes.reserve(input_count);
  for (uint32_t i = 0; i < input_count && i < 6; i++) {
    vsx::TShape shape;
    models[0]->GetInputShapeByIndex(0, shape);
    input_shapes.push_back(std::move(shape));
  }

  vsx::ProfilerConfig config = {instance,     iterations,  batch_size,
                                vsx::kInt32,  device_ids,  contexts,
                                input_shapes, percentiles, queue_size};
  vsx::ModelProfiler<vsx::ClipText> profiler(config, models);
  std::cout << profiler.Profiling() << std::endl;
  return 0;
}