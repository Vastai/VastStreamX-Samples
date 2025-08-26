
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/cmdline.hpp"
#include "common/dynamic_detector.hpp"
#include "common/model_profiler.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("module_info", 'm', "model info json files", false,
                        "/opt/vastai/vaststreamx/data/models/"
                        "torch-yolov5s_coco-int8-percentile-Y-Y-2-none/"
                        "yolov5s_coco_module_info.json");
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "./data/configs/yolo_div255_bgr888.json");
  args.add<std::string>("device_ids", 'd', "device id to run", false, "[0]");
  args.add<std::string>("max_input_shape", '\0', "model max input shape", false,
                        "[1,3,640,640]");
  args.add<uint32_t>("batch_size", 'b', "profiling batch size of the model",
                     false, 1);
  args.add<uint32_t>("instance", 'i',
                     "instance number or range for each device", false, 1);
  args.add<std::string>("shape", 's', "model input shape", false);
  args.add<int>("iterations", '\0', "iterations count for one profiling", false,
                10240);
  args.add<std::string>("percentiles", '\0', "percentiles of latency", false,
                        "[50,90,95,99]");
  args.add<float>("threshold", '\0', "threshold for detection", false, 0.5);
  args.add<bool>("input_host", '\0', "cache input data into host memory", false,
                 0);
  args.add<uint32_t>("queue_size", 'q', "aync wait queue size", false, 2);
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  auto module_info = args.get<std::string>("module_info");
  auto vdsp_params = args.get<std::string>("vdsp_params");
  auto max_input_shape =
      vsx::ParseShape(args.get<std::string>("max_input_shape"));
  auto device_ids = vsx::ParseVecUint(args.get<std::string>("device_ids"));
  auto batch_size = args.get<uint32_t>("batch_size");
  auto instance = args.get<uint32_t>("instance");
  auto iterations = args.get<int>("iterations");
  auto threshold = args.get<float>("threshold");
  auto input_host = args.get<bool>("input_host");
  auto queue_size = args.get<uint32_t>("queue_size");
  auto percentiles = vsx::ParseVecUint(args.get<std::string>("percentiles"));

  std::vector<std::shared_ptr<vsx::DynamicDetector>> models;
  models.reserve(instance);
  std::vector<vsx::Context> contexts;
  auto input_shapes = std::vector<vsx::TShape>({max_input_shape});
  for (uint32_t i = 0; i < instance; i++) {
    uint32_t device_id = device_ids[i % (device_ids.size())];
    if (input_host) {
      contexts.push_back(vsx::Context::CPU());
    } else {
      contexts.push_back(vsx::Context::VACC(device_id));
    }
    models.push_back(std::make_shared<vsx::DynamicDetector>(
        module_info, vdsp_params, input_shapes, batch_size, device_id,
        threshold));
  }
  vsx::TShape shape = vsx::ParseShape(args.get<std::string>("shape"));
  CHECK(shape.ndim() == 4) << "input shape dims should be 4";
  for (uint32_t i = 0; i < instance; i++) {
    std::vector<vsx::TShape> batch_shapes;
    for (uint32_t j = 0; j < batch_size; j++) {
      batch_shapes.push_back(shape);
    }
    models[i]->SetInputShape(batch_shapes);
  }

  vsx::ProfilerConfig config = {instance,    iterations,  batch_size,
                                vsx::kUint8, device_ids,  contexts,
                                {shape},     percentiles, queue_size};
  vsx::ModelProfiler<vsx::DynamicDetector> profiler(config, models);
  std::cout << profiler.Profiling() << std::endl;
  return 0;
}