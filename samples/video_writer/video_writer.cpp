
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "vaststreamx/media/video_writer.h"

#include <sys/stat.h>

#include <fstream>
#include <memory>

#include "common/cmdline.hpp"
#include "common/file_system.hpp"
#include "common/utils.hpp"
#include "common/yuv_reader.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("codec_type", '\0', "codec type", false, "H264");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<uint32_t>("width", '\0', "width", false, 0);
  args.add<uint32_t>("height", '\0', "height", false, 0);
  args.add<uint32_t>("frame_rate", '\0', "frame rate", false, 30);
  args.add<std::string>("input_file", '\0', "input file", false, "");
  args.add<std::string>("output_uri", '\0', "output uri", false, "./test.ts");
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
  auto output_uri = args.get<std::string>("output_uri");

  CHECK(vsx::SetDevice(device_id) == 0)
      << "Failed to set device id: " << device_id;

  vsx::CodecType type;
  if (codec_type == "H264" || codec_type == "h264") {
    type = vsx::CODEC_TYPE_H264;
  } else if (codec_type == "H265" || codec_type == "h265") {
    type = vsx::CODEC_TYPE_HEVC;
  } else {
    std::cerr << "undefined codec_type:" << codec_type << std::endl;
    return -1;
  }

  vsx::YUVReader reader(input_file, width, height, vsx::ImageFormat::YUV_NV12);
  vsx::VideoWriter writer(output_uri, frame_rate, type, 0, 0, device_id);
  if (!writer.isOpened()) {
    LOG(ERROR) << "VideoWriter is not opened,uri:" << output_uri;
    writer.release();
    return -1;
  }
  int ts = 0;
  do {
    auto image = reader.GetTestData(false);
    if (image.Format() == vsx::ImageFormat::YUV_NV12 && image.GetDataPtr() &&
        image.GetDataBytes()) {
      auto frame_attr = std::make_shared<vsx::FrameAttr>();
      frame_attr->frame_pts = ts;
      frame_attr->frame_dts = ts;
      writer.write(image, frame_attr);
      ts++;
    } else {
      writer.release();
      break;
    }
  } while (1);

  return 0;
}