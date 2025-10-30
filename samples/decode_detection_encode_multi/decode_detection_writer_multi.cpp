/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <filesystem>
#include <fstream>
#include <iostream>

#include "common/cmdline.hpp"
#include "common/detector.hpp"
#include "opencv2/opencv.hpp"
#include "vaststreamx/vaststreamx.h"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>(
      "model_prefix", 'm', "model prefix of the model suite files", false,
      "/opt/vastai/vaststreamx/data/models/"
      "yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod");
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "./data/configs/yolo_div255_yuv_nv12.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<float>("threshold", '\0', "threshold for detection", false, 0.5);
  args.add<std::string>("uri", '\0', "input uri", false,
                        "../data/videos/test.mp4");
  args.add<std::string>("output_path", '\0', "output_path", false, "");
  args.add<uint32_t>("num_channels", '\0',
                     "number of channels to decode [ default: 1 ]", false, 1);
  args.add<uint32_t>("loop", '\0', "loop count for each channel [ default: 1 ]",
                     false, 1);
  args.add<uint32_t>("keep", '\0', "keep num [default:1]", false, 1);
  args.add<uint32_t>("drop", '\0', "drop num [default:0]", false, 0);
  args.add<std::string>("uri_list", '\0', "input uri list file", false, "");
  args.add<uint32_t>("disable_encoder", '\0', "enable encoder or not", false,
                     0);
  args.parse_check(argc, argv);
  return args;
}

int Process(const cmdline::parser& args, std::string input_uri,
            uint32_t cur_device_id, uint32_t gap, uint32_t index = 0,
            uint32_t loop = 1) {
  // initialize device
  CHECK(vsx::SetDevice(cur_device_id) == 0)
      << "Failed to set device id: " << cur_device_id;

  for (uint32_t inn = 0; inn < loop; inn++) {
    std::cout << "Process " << index << " on device " << cur_device_id
              << " for uri:" << input_uri << ", loop: " << inn << std::endl;
    // initialize model
    int batch_size = 1;
    vsx::Detector detector(args.get<std::string>("model_prefix"),
                           args.get<std::string>("vdsp_params"), batch_size,
                           cur_device_id);
    detector.SetThreshold(args.get<float>("threshold"));
    // open uri
    vsx::VideoCapture cap(input_uri, vsx::FULLSPEED_MODE, cur_device_id, true,
                          false);
    CHECK(cap.isOpened()) << "Failed to open uri: " << input_uri;

    vsx::Image frame;  // frame format is nv12, memory is in device
    std::shared_ptr<vsx::FrameAttr> frame_attr;
    if (!cap.read(frame, frame_attr)) {  // get first frame
      std::cout << "Failed to read frame\n";
      return -1;
    }

    // print video info
    // std::cout << "Frame width: " << frame.Width()
    //           << ", width_pitch: " << frame.WidthPitch()
    //           << ", height: " << frame.Height()
    //           << ", height_pitch: " << frame.HeightPitch() << std::endl;
    // std::cout << "Frame rate: " << frame_attr->video_fps << std::endl;
    // std::cout << "Video codec_info: " << frame_attr->codec_info << std::endl;
    // if (frame_attr->color_space == vsx::ImageColorSpace::kCOLOR_SPACE_BT709)
    //   std::cout << "Frame ColorSpace:  BT709 \n";
    // else if (frame_attr->color_space ==
    //          vsx::ImageColorSpace::kCOLOR_SPACE_BT601)
    //   std::cout << "Frame ColorSpace:  BT601 \n";
    // else if (frame_attr->color_space ==
    //          vsx::ImageColorSpace::kCOLOR_SPACE_BT709_LIMIT_RANGE)
    //   std::cout << "Frame ColorSpace:  BT709_LIMIT_RANGE \n";
    // else if (frame_attr->color_space ==
    //          vsx::ImageColorSpace::kCOLOR_SPACE_BT601_FULL_RANGE)
    //   std::cout << "Frame ColorSpace:  BT601_FULL_RANGE \n";

    auto disable_encoder = args.get<uint32_t>("disable_encoder");
    std::unique_ptr<vsx::VideoWriter> video_writer;

    // create video_writer
    if (!disable_encoder) {
      float frame_rate = frame_attr->video_fps;

      vsx::CodecType codec_type;
      if (frame_attr->codec_info.find("avc1") != std::string::npos) {
        codec_type = vsx::CODEC_TYPE_H264;
      } else if (frame_attr->codec_info.find("hevc") != std::string::npos) {
        codec_type = vsx::CODEC_TYPE_HEVC;
      } else {
        std::cerr << "undefined codec_type:" << frame_attr->codec_info
                  << std::endl;
        return -1;
      }

      // create video writer
      auto output_path = args.get<std::string>("output_path");
      auto outfile_name =
          output_path + "/channel_" + std::to_string(index) + ".ts";
      video_writer = std::make_unique<vsx::VideoWriter>(
          outfile_name, frame_rate, codec_type, 0, 0, cur_device_id);
    }

    // detection result file
    auto output_path = args.get<std::string>("output_path");
    auto detection_file_name =
        output_path + "/channel_" + std::to_string(index) + "_detection.txt";
    std::ofstream det_file(detection_file_name);
    if (!det_file.is_open()) {
      std::cout << "Error, Failed to open detection result file: "
                << detection_file_name << std::endl;
      return -1;
    }
    det_file << "frame_num, " << "class_id, " << "score, " << "box\n";

    auto drop = args.get<uint32_t>("drop");
    auto keep = args.get<uint32_t>("keep");
    uint32_t frame_num = 0;
    uint32_t count = 0;
    do {
      frame_num++;
      count++;
      if (count <= keep) {
        // inference
        auto result = detector.Process(frame);
        // parse output
        auto res_shape = result.Shape();
        const float* res_data = result.Data<float>();
        std::vector<vsx::Rect> obj_rects;
        for (int i = 0; i < res_shape[0]; i++) {
          if (res_data[0] < 0) break;
          int classe_id = int(res_data[0] + 0.1);
          float score = res_data[1];
          vsx::Rect rect;
          rect.x = int(res_data[2]);
          rect.y = int(res_data[3]);
          rect.w = int(res_data[4]);
          rect.h = int(res_data[5]);

          obj_rects.push_back(rect);

          res_data += vsx::kDetectionOffset;
          // write objects to file, frame_num, class_id, score,box
          det_file << frame_num << ", " << classe_id << ", " << score << ", "
                   << rect.x << "," << rect.y << "," << rect.w << "," << rect.h
                   << std::endl;
        }
        if (!disable_encoder) {
          // draw box to image, ONLY support YUV_NV12 format image
          vsx::Image output_image;
          vsx::DrawRectangle(frame, output_image, obj_rects,
                             vsx::YUVColor::YUV_RED, 2);
          frame = output_image;
        }
      }
      if (!disable_encoder) {
        // write frame ( encode frame )
        video_writer->write(frame, frame_attr);
      }
      if (count == keep + drop) {
        count = 0;
      }
    } while (cap.read(frame, frame_attr));

    std::cout << "Close cap\n";
    cap.release();
    det_file.close();

    if (!disable_encoder) {
      std::cout << "Close video writer\n";
      video_writer->release();
    }
  }
  return 0;
}

int main(int argc, char* argv[]) {
  // get command parameters
  auto args = ArgumentParser(argc, argv);
  auto num_channels = args.get<uint32_t>("num_channels");
  // get input uri
  std::vector<std::string> input_list;
  if (!args.get<std::string>("uri_list").empty())
    // load uri list from file
    input_list = vsx::LoadLabels(args.get<std::string>("uri_list"));
  else if (!args.get<std::string>("uri").empty()) {
    for (uint32_t i = 0; i < num_channels; i++) {
      input_list.push_back(args.get<std::string>("uri"));
    }
  }

  if (args.get<std::string>("output_path").empty()) {
    std::cerr << "please set output_path \n";
    return -1;
  }
  std::filesystem::path output_path(args.get<std::string>("output_path"));
  std::error_code ec;
  std::filesystem::create_directories(output_path, ec);
  if (ec) {
    std::cerr << "Failed to create output_path "
              << args.get<std::string>("output_path")
              << ", message:" << ec.message() << std::endl;
    return -1;
  }

  assert(input_list.size() == num_channels);

  auto device_id = args.get<uint32_t>("device_id");
  auto drop = args.get<uint32_t>("drop");
  auto keep = args.get<uint32_t>("keep");
  auto loop = args.get<uint32_t>("loop");

  uint32_t split_num = num_channels / 2;
  if (split_num == 0) split_num = 1;
  uint32_t gap = drop / keep + 1;

  std::vector<std::thread> process_list;
  process_list.reserve(num_channels);
  for (size_t i = 0; i < num_channels; i++) {
    auto cur_device_id = device_id + int(i / split_num);
    std::cout << "input_uri:" << input_list[i]
              << ", device_id: " << cur_device_id << std::endl;
    process_list.emplace_back(&Process, std::cref(args), input_list[i],
                              cur_device_id, gap, i, loop);
  }

  for (auto& p : process_list) {
    p.join();
  }
}
