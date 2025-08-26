
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
#include "common/media_profiler.hpp"
#include "common/utils.hpp"
#include "common/vdecoder.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("codec_type", '\0', "codec type", false, "H264");
  args.add<std::string>("device_ids", 'd', "device id to run", false, "[0]");
  args.add<std::string>("input_file", '\0', "input file", false, "");

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
  std::string codec_type = args.get<std::string>("codec_type");
  vsx::CodecType type;
  if (codec_type == "H264" || codec_type == "h264") {
    type = vsx::CODEC_TYPE_H264;
  } else if (codec_type == "H265" || codec_type == "h265") {
    type = vsx::CODEC_TYPE_HEVC;
  } else if (codec_type == "AV1" || codec_type == "av1") {
    type = vsx::CODEC_TYPE_AV1;
  } else {
    std::cerr << "undefined codec_type:" << codec_type << std::endl;
    return -1;
  }
  std::string input_file = args.get<std::string>("input_file");

  auto device_ids = vsx::ParseVecUint(args.get<std::string>("device_ids"));
  auto instance = args.get<uint32_t>("instance");
  auto iterations = args.get<int>("iterations");
  auto percentiles = vsx::ParseVecUint(args.get<std::string>("percentiles"));

  std::vector<std::shared_ptr<vsx::Vdecoder>> vdecoders;
  vdecoders.reserve(instance);
  for (uint32_t i = 0; i < instance; i++) {
    uint32_t device_id = device_ids[i % (device_ids.size())];
    vsx::SetDevice(device_id);
    vdecoders.push_back(
        std::make_shared<vsx::Vdecoder>(type, device_id, input_file));
  }

  vsx::MediaProfilerConfig config = {instance, iterations, device_ids,
                                     percentiles};
  vsx::MediaProfiler<vsx::Vdecoder, vsx::Image> profiler(config, vdecoders);
  std::cout << profiler.Profiling() << std::endl;
  return 0;
}