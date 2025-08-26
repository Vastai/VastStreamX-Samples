
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
#include "common/vdecoder.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("codec_type", '\0', "codec type", false, "H264");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("input_file", '\0', "input file", false, "");
  args.add<std::string>("output_folder", '\0', "output folder", false, "");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  uint32_t device_id = args.get<uint32_t>("device_id");
  CHECK(vsx::SetDevice(device_id) == 0)
      << "Failed to set device id " << device_id;

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

  auto input_file = args.get<std::string>("input_file");
  auto output_folder = args.get<std::string>("output_folder");

  vsx::Vdecoder vdecoder(type, device_id, input_file);

  std::thread get_frame_thread([&]() {
    vsx::Image image;
    std::shared_ptr<vsx::FrameAttr> frame_attr;
    uint32_t count = 0;
    while (vdecoder.GetResult(image)) {
      auto cpu_image = vsx::Image(image.Format(), image.Width(), image.Height(),
                                  vsx::Context::CPU(), image.WidthPitch(),
                                  image.HeightPitch());
      cpu_image.CopyFrom(image);
      if (image.Width() < image.WidthPitch() ||
          image.Height() < image.HeightPitch()) {
        auto temp_image =
            vsx::Image(image.Format(), image.Width(), image.Height());
        temp_image.CopyFrom(cpu_image);
        image = temp_image;
      } else {
        image = cpu_image;
      }
      std::string file_name = output_folder + "/" +
                              std::to_string(image.Width()) + "x" +
                              std::to_string(image.Height()) + "_" +
                              std::to_string(count++) + ".yuv";
      std::ofstream ofs(file_name, std::ios::binary);
      CHECK(ofs.is_open()) << "Failed to open: " << file_name;
      ofs.write(image.Data<char>(), image.GetDataBytes());
      ofs.close();
      std::cout << "Write yuv file: " << file_name << std::endl;
    }
  });

  while (true) {
    auto media_data = vdecoder.GetTestData(false);
    if (media_data) {
      vdecoder.Process(media_data, false);
    } else {
      vdecoder.Process(media_data, true);
      break;
    }
  }
  get_frame_thread.join();
  return 0;
}