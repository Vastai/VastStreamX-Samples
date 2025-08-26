
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
#include "common/utils.hpp"
#include "common/vencoder.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("codec_type", '\0', "codec type", false, "H264");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<uint32_t>("width", '\0', "width", false, 0);
  args.add<uint32_t>("height", '\0', "height", false, 0);
  args.add<uint32_t>("frame_rate", '\0', "frame rate", false, 30);
  args.add<std::string>("input_file", '\0', "input file", false, "");
  args.add<std::string>("output_file", '\0', "output file", false, "");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  auto device_id = args.get<uint32_t>("device_id");
  auto width = args.get<uint32_t>("width");
  auto height = args.get<uint32_t>("height");
  auto frame_rate = args.get<uint32_t>("frame_rate");
  auto codec_type = args.get<std::string>("codec_type");

  auto input_file = args.get<std::string>("input_file");
  auto output_file = args.get<std::string>("output_file");

  CHECK(vsx::SetDevice(device_id) == 0)
      << "Failed to set device id: " << device_id;

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
  uint32_t frame_rate_numerator = frame_rate;
  uint32_t frame_rate_denominator = 1;
  vsx::Vencoder vencoder(type, device_id, input_file, width, height,
                         vsx::ImageFormat::YUV_NV12, frame_rate_numerator,
                         frame_rate_denominator);

  std::thread get_frame_thread([&]() {
    std::shared_ptr<vsx::DataManager> data;
    std::ofstream ofs(output_file, std::ios::binary);
    while (vencoder.GetResult(data)) {
      ofs.write(reinterpret_cast<const char*>(data->GetDataPtr()),
                data->GetDataSize());
    }
    ofs.close();
  });

  do {
    auto image = vencoder.GetTestData(false);
    if (image.Format() == vsx::ImageFormat::YUV_NV12 && image.GetDataPtr() &&
        image.GetDataBytes()) {
      vencoder.Process(image, false);
    } else {
      vencoder.Process(image, true);
      break;
    }
  } while (1);
  get_frame_thread.join();
  std::cout << "Write encode data to: " << output_file << std::endl;

  return 0;
}