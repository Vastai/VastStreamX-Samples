
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <sys/stat.h>

#include <fstream>

#include "common/cmdline.hpp"
#include "common/file_system.hpp"
#include "common/jencoder.hpp"
#include "common/media_profiler.hpp"
#include "common/utils.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("device_ids", 'd', "device id to run", false, "[0]");
  args.add<std::string>("input_file", '\0', "input file", false,
                        "../data/images/plate_1920_1080.yuv");

  args.add<uint32_t>("width", '\0', "width", false, 1920);
  args.add<uint32_t>("height", '\0', "height", false, 1080);
  args.add<uint32_t>("instance", 'i', "instance number for each device", false,
                     1);
  args.add<int>("iterations", '\0', "iterations count for one profiling", false,
                10240);
  args.add<std::string>("percentiles", '\0', "percentiles of latency", false,
                        "[50, 90, 95, 99]");

  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);

  std::string input_file = args.get<std::string>("input_file");
  uint32_t width = args.get<uint32_t>("width");
  uint32_t height = args.get<uint32_t>("height");
  auto device_ids = vsx::ParseVecUint(args.get<std::string>("device_ids"));
  auto instance = args.get<uint32_t>("instance");
  auto iterations = args.get<int>("iterations");
  auto percentiles = vsx::ParseVecUint(args.get<std::string>("percentiles"));

  std::vector<std::shared_ptr<vsx::Jencoder>> jencoders;
  jencoders.reserve(instance);
  for (uint32_t i = 0; i < instance; i++) {
    uint32_t device_id = device_ids[i % (device_ids.size())];
    vsx::SetDevice(device_id);
    jencoders.push_back(std::make_shared<vsx::Jencoder>(
        device_id, input_file, width, height, vsx::ImageFormat::YUV_NV12));
  }

  vsx::MediaProfilerConfig config = {instance, iterations, device_ids,
                                     percentiles};
  vsx::MediaProfiler<vsx::Jencoder, std::shared_ptr<vsx::DataManager>> profiler(
      config, jencoders);
  std::cout << profiler.Profiling() << std::endl;
  return 0;
}