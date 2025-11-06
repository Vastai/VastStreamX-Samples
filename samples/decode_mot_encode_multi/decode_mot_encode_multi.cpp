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

#include "bytetracker.hpp"
#include "common/cmdline.hpp"
#include "opencv2/opencv.hpp"
#include "vaststreamx/vaststreamx.h"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>(
      "model_prefix", 'm', "model prefix of the model suite files", false,
      "/opt/vastai/vaststreamx/data/models/"
      "bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod");
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "./data/configs/yolo_div255_yuv_nv12.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<float>("detect_thresh", '\0', "threshold for detection", false, 0.5);

  args.add<float>("track_thresh", '\0', "threshold for tracker", false, 0.6);
  args.add<uint32_t>("track_buffer", '\0', "the frames for keep lost tracks",
                     false, 30);
  args.add<float>("match_thresh", '\0', "matching threshold for tracking",
                  false, 0.9);
  args.add<float>("min_box_area", '\0', "filter out tiny boxes", false, 100);

  args.add<std::string>("uri", '\0', "input uri", false,
                        "../data/videos/test.mp4");
  args.add<std::string>("output_path", '\0', "output_path", false, "");
  args.add<uint32_t>("num_channels", '\0',
                     "number of channels to decode [ default: 1 ]", false, 1);
  args.add<uint32_t>("loop", '\0', "loop count for each channel [ default: 1 ]",
                     false, 1);
  args.add<std::string>("uri_list", '\0', "input uri list file", false, "");
  args.add<uint32_t>("disable_encoder", '\0', "enable encoder or not", false,
                     0);
  args.parse_check(argc, argv);
  return args;
}

int Process(const cmdline::parser& args, std::string input_uri,
            uint32_t cur_device_id, uint32_t index = 0, uint32_t loop = 1) {
  // initialize device
  CHECK(vsx::SetDevice(cur_device_id) == 0)
      << "Failed to set device id: " << cur_device_id;

  auto detect_thresh = args.get<float>("detect_thresh");
  auto track_buffer = args.get<uint32_t>("track_buffer");
  auto track_thresh = args.get<float>("track_thresh");

  for (uint32_t inn = 0; inn < loop; inn++) {
    std::cout << "Process " << index << " on device " << cur_device_id
              << " for uri:" << input_uri << ", loop: " << inn << std::endl;
    // initialize model
    int batch_size = 1;
    int fps = 30;
    vsx::ByteTracker tracker(args.get<std::string>("model_prefix"),
                             args.get<std::string>("vdsp_params"), batch_size,
                             cur_device_id, detect_thresh, track_buffer,
                             track_thresh, fps);
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
    auto track_file_name =
        output_path + "/channel_" + std::to_string(index) + ".txt";
    std::ofstream track_file(track_file_name);
    if (!track_file.is_open()) {
      std::cout << "Error, Failed to open track result file: "
                << track_file_name << std::endl;
      return -1;
    }

    uint32_t frame_id = 0;
    do {
      frame_id++;
      // inference
      auto result = tracker.Process(frame);
      auto res_shape = result.Shape();
      const float* res_data = result.Data<float>();
      std::vector<vsx::Rect> obj_rects;
      for (int i = 0; i < res_shape[0]; i++) {
        if (res_data[0] < 0) break;

        vsx::Rect rect;
        rect.x = int(res_data[2]);
        rect.y = int(res_data[3]);
        rect.w = int(res_data[4]);
        rect.h = int(res_data[5]);
        obj_rects.push_back(rect);

        float score = res_data[1];
        // '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
        track_file << frame_id << "," << static_cast<int>(res_data[6]) << ","
                   << std::setiosflags(std::ios::fixed) << std::setprecision(2)
                   << res_data[2] << "," << res_data[3] << "," << res_data[4]
                   << "," << res_data[5] << "," << score << ",-1,-1,-1\n";
        res_data += vsx::kDetectionOffset;
      }
      if (!disable_encoder) {
        // draw box to image, ONLY support YUV_NV12 format image
        vsx::Image output_image;
        vsx::DrawRectangle(frame, output_image, obj_rects,
                           vsx::YUVColor::YUV_RED, 2);
        video_writer->write(output_image, frame_attr);
      }

    } while (cap.read(frame, frame_attr));

    std::cout << "Close cap\n";
    cap.release();
    track_file.close();

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
  auto loop = args.get<uint32_t>("loop");

  uint32_t split_num = num_channels / 2;
  if (split_num == 0) split_num = 1;

  std::vector<std::thread> process_list;
  process_list.reserve(num_channels);
  for (size_t i = 0; i < num_channels; i++) {
    auto cur_device_id = device_id + int(i / split_num);
    std::cout << "input_uri:" << input_list[i]
              << ", device_id: " << cur_device_id << std::endl;
    process_list.emplace_back(&Process, std::cref(args), input_list[i],
                              cur_device_id, i, loop);
  }

  for (auto& p : process_list) {
    p.join();
  }
}
