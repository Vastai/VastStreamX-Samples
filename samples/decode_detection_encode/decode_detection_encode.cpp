
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "./data/configs/yolo_div255_yuv_nv12.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<float>("threshold", '\0', "threshold for detection", false, 0.5);
  args.add<std::string>("label_file", '\0', "label file", false,
                        "../data/labels/coco2id.txt");
  args.add<std::string>("input_uri", '\0', "input uri", false,
                        "../data/videos/test.mp4");
  args.add<std::string>("output_file", '\0', "output_file", false, "");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char* argv[]) {
  // get command parameters
  auto args = ArgumentParser(argc, argv);
  auto device_id = args.get<uint32_t>("device_id");
  // initialize device
  CHECK(vsx::SetDevice(device_id) == 0)
      << "Failed to set device id: " << device_id;
  // initialize model
  auto labels = vsx::LoadLabels(args.get<std::string>("label_file"));
  const int batch_size = 1;
  vsx::Detector detector(args.get<std::string>("model_prefix"),
                         args.get<std::string>("vdsp_params"), batch_size,
                         args.get<uint32_t>("device_id"));
  detector.SetThreshold(args.get<float>("threshold"));

  // open uri
  vsx::VideoCapture cap(args.get<std::string>("input_uri"), vsx::FULLSPEED_MODE,
                        device_id, true, false);
  CHECK(cap.isOpened()) << "Failed to open uri: "
                        << args.get<std::string>("input_uri");
  // output file
  bool save_video = false;
  std::shared_ptr<vsx::VideoWriter> writer;
  auto output_file = args.get<std::string>("output_file");
  if (output_file != "") {
    save_video = true;
    std::cout << "Save video\n";
  }

  vsx::Image frame;  // frame format is nv12, memory is in device
  std::shared_ptr<vsx::FrameAttr> frame_attr;
  if (!cap.read(frame, frame_attr)) {  // get first frame
    std::cout << "Failed to read frame\n";
    return -1;
  }

  // print video info
  std::cout << "Frame width: " << frame.Width()
            << ", width_pitch: " << frame.WidthPitch()
            << ", height: " << frame.Height()
            << ", height_pitch: " << frame.HeightPitch() << std::endl;
  std::cout << "Frame rate: " << frame_attr->video_fps << std::endl;
  std::cout << "Video codec_info: " << frame_attr->codec_info << std::endl;
  if (frame_attr->color_space == vsx::ImageColorSpace::kCOLOR_SPACE_BT709)
    std::cout << "Frame ColorSpace:  BT709 \n";
  else if (frame_attr->color_space == vsx::ImageColorSpace::kCOLOR_SPACE_BT601)
    std::cout << "Frame ColorSpace:  BT601 \n";
  else if (frame_attr->color_space ==
           vsx::ImageColorSpace::kCOLOR_SPACE_BT709_LIMIT_RANGE)
    std::cout << "Frame ColorSpace:  BT709_LIMIT_RANGE \n";
  else if (frame_attr->color_space ==
           vsx::ImageColorSpace::kCOLOR_SPACE_BT601_FULL_RANGE)
    std::cout << "Frame ColorSpace:  BT601_FULL_RANGE \n";

  if (save_video) {
    // int width = frame.Width();
    // int height = frame.Height();
    float frame_rate = frame_attr->video_fps;
    // uint32_t frame_rate_numerator = frame_rate;
    // uint32_t frame_rate_denominator = 1;
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

    // open video_writer
    int default_bit_rate = 0;
    int key_frame_interval = 25;
    writer = std::make_shared<vsx::VideoWriter>(
        output_file, static_cast<uint32_t>(frame_rate), codec_type,
        default_bit_rate, key_frame_interval, device_id);
    if (!writer->isOpened()) {
      std::cout << "Failed to open video_writer, params are output_file: "
                << output_file << ", frame_rate: " << frame_rate
                << ", codec_type: " << codec_type
                << ", bit_rate: " << default_bit_rate
                << ", key_frame_interval: " << key_frame_interval
                << ", device_id: " << device_id << std::endl;
      return -1;
    }
  }
  int frame_num = 0;
  do {
    // inference
    auto result = detector.Process(frame);
    auto res_shape = result.Shape();
    const float* res_data = result.Data<float>();
    std::cout << "Frame : " << frame_num++ << " detection objects:\n";
    for (int i = 0; i < res_shape[0]; i++) {
      if (res_data[0] < 0) break;
      std::string class_name = labels[static_cast<int>(res_data[0])];
      float score = res_data[1];
      std::cout << "Object class: " << class_name << ", score: " << score
                << ", bbox: [" << res_data[2] << ", " << res_data[3] << ", "
                << res_data[4] << ", " << res_data[5] << "]\n";
      res_data += vsx::kDetectionOffset;
    }

    if (save_video) {  // draw box in frame
      // cvtcolor to rgb, and copy to cpu
      vsx::Image rgb_image_cpu;
      vsx::CvtColor(frame, rgb_image_cpu, vsx::ImageFormat::RGB_PLANAR,
                    frame_attr->color_space, true);
      // convert to opencv mat
      cv::Mat cv_mat;
      vsx::ConvertVsxImageToCvMatBgrPacked(rgb_image_cpu, cv_mat);
      // draw box
      const float* res_data = result.Data<float>();
      for (int i = 0; i < res_shape[0]; i++) {
        if (res_data[0] < 0) break;
        cv::Rect2f rect = {res_data[2], res_data[3], res_data[4], res_data[5]};
        cv::rectangle(cv_mat, rect, cv::Scalar(0, 255, 0), 2);
        res_data += vsx::kDetectionOffset;
      }
      // convert cv_mat to vsx::Image
      vsx::Image vsx_yuv_image;
      CHECK(vsx::MakeVsxImage(cv_mat, vsx_yuv_image,
                              vsx::ImageFormat::YUV_NV12) == 0);
      // write frame
      writer->write(vsx_yuv_image, frame_attr);
    }
  } while (cap.read(frame, frame_attr));
  std::cout << "Close cap\n";
  cap.release();
  if (save_video) {
    std::cout << "Close writer\n";
    writer->release();
  }
  return 0;
}