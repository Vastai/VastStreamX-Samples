
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <chrono>
#include <thread>

#include "argmax_op.hpp"
#include "common/cmdline.hpp"
#include "common/model_profiler.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("elf_file", '\0', "elf_file path", false,
                        "/opt/vastai/vaststreamx/data/elf/planar_argmax");
  args.add<std::string>("device_ids", 'd', "device id to run", false, "[0]");
  args.add<std::string>("shape", '\0', "input shape [c,h,w]", false,
                        "[19,512,512]");
  args.add<uint32_t>("instance", 'i',
                     "instance number or range for each device", false, 1);
  args.add<int>("iterations", '\0', "iterations count for one profiling", false,
                10240);
  args.add<std::string>("percentiles", '\0', "percentiles of latency", false,
                        "[50,90,95,99]");
  args.add<bool>("input_host", '\0', "cache input data into host memory", false,
                 0);
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char* argv[]) {
  auto args = ArgumentParser(argc, argv);
  auto elf_file = args.get<std::string>("elf_file");
  auto device_ids = vsx::ParseVecUint(args.get<std::string>("device_ids"));
  uint32_t batch_size = 1;
  auto instance = args.get<uint32_t>("instance");
  auto iterations = args.get<int>("iterations");
  auto input_host = args.get<bool>("input_host");
  uint32_t queue_size = 0;
  auto percentiles = vsx::ParseVecUint(args.get<std::string>("percentiles"));
  auto shape = vsx::ParseShape(args.get<std::string>("shape"));

  std::vector<std::shared_ptr<vsx::ArgmaxOp>> ops;
  ops.reserve(instance);
  std::vector<vsx::Context> contexts;
  for (uint32_t i = 0; i < instance; i++) {
    uint32_t device_id = device_ids[i % (device_ids.size())];
    if (input_host) {
      contexts.push_back(vsx::Context::CPU());
    } else {
      contexts.push_back(vsx::Context::VACC(device_id));
    }
    ops.push_back(std::make_shared<vsx::ArgmaxOp>("planar_argmax_op", elf_file,
                                                  device_id));
  }
  std::cout << "shape:" << shape << ", instance:" << instance
            << ", iterations:" << iterations << ",batch_size:" << batch_size
            << ", queue_size: " << queue_size << std::endl;
  vsx::ProfilerConfig config = {instance,      iterations,  batch_size,
                                vsx::kFloat16, device_ids,  contexts,
                                {shape},       percentiles, queue_size};
  vsx::ModelProfiler<vsx::ArgmaxOp> profiler(config, ops);
  std::cout << profiler.Profiling() << std::endl;
  return 0;
}
