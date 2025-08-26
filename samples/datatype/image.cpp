
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/cmdline.hpp"
#include "common/utils.hpp"
#include "vaststreamx/vaststreamx.h"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("input_file", '\0', "input image", false,
                        "../data/images/dog.jpg");
  args.add<std::string>("output_file", '\0', "output image", false,
                        "image_out.jpg");
  args.parse_check(argc, argv);
  return args;
}

vsx::Image ReadImageFromFile(const std::string& image_file,
                             vsx::ImageFormat format) {
  vsx::Image image;
  if (vsx::MakeVsxImage(image_file, image, format) != 0) {
    LOG(ERROR) << "MakeVsxImage failed, filename: " << image_file
               << ", format: " << vsx::ImageFormatToString(format);
    exit(-1);
  }
  return image;
}

int WriteImageToFile(const std::string& image_file, vsx::Image image) {
  cv::Mat mat_bgr_packed;
  if (vsx::ConvertVsxImageToCvMatBgrPacked(image, mat_bgr_packed) != 0) {
    LOG(ERROR) << "ConvertVsxImageToCvMatBgrPacked failed";
    return -1;
  }
  cv::imwrite(image_file, mat_bgr_packed);
  return 0;
}

int CopyDeviceImageToHostImage(vsx::Image& image_cpu,
                               const vsx::Image& image_vacc) {
  if (image_vacc.GetContext().dev_type != vsx::Context::kVACC) {
    LOG(ERROR) << "image_vacc memory is not in device.";
    return -1;
  }

  image_cpu =
      vsx::Image(image_vacc.Format(), image_vacc.Width(), image_vacc.Height(),
                 image_vacc.WidthPitch(), image_vacc.HeightPitch());
  image_cpu.CopyFrom(image_vacc);
  return 0;
}

int CopyHostImageToDeviceImage(vsx::Image& image_vacc, int device_id,
                               const vsx::Image& image_cpu) {
  if (image_cpu.GetContext().dev_type != vsx::Context::kCPU) {
    LOG(ERROR) << "image_cpu memory is not in host.";
    return -1;
  }

  image_vacc = vsx::Image(image_cpu.Format(), image_cpu.Width(),
                          image_cpu.Height(), vsx::Context::VACC(device_id),
                          image_cpu.WidthPitch(), image_cpu.HeightPitch());
  image_vacc.CopyFrom(image_cpu);
  return 0;
}

int ChangePixels(vsx::Image& image_cpu) {
  if (image_cpu.GetContext().dev_type != vsx::Context::kCPU) {
    LOG(ERROR) << "image_cpu memory is not in host.";
    return -1;
  }
  auto data_uint8 = image_cpu.MutableData<uint8_t>();
  int linesize =
      image_cpu.WidthPitch() > 0 ? image_cpu.WidthPitch() : image_cpu.Width();
  int plane_height = image_cpu.HeightPitch() > 0 ? image_cpu.HeightPitch()
                                                 : image_cpu.Height();
  int plane_offset = linesize * plane_height;

  if (image_cpu.Format() == vsx::RGB_PLANAR) {
    uchar* r_plane = data_uint8;
    uchar* g_plane = r_plane + plane_offset;
    uchar* b_plane = g_plane + plane_offset;

    // change pixels
    for (int h = 0; h < 10; h++) {
      for (int w = 0; w < 10; w++) {
        r_plane[h * linesize + w] = w;
        g_plane[h * linesize + w] = w;
        b_plane[h * linesize + w] = w;
      }
    }
  } else if (image_cpu.Format() == vsx::YUV_NV12) {
    // change pixels
    uchar* y_plane = data_uint8;
    uchar* uv_plane = y_plane + plane_offset;
    for (int h = 0; h < 10; h++) {
      for (int w = 0; w < 10; w++) {
        y_plane[h * linesize + w] = w;
        if (h % 2 == 0 && w % 2 == 0) {
          uv_plane[h / 2 * linesize + w + 0] = w;
          uv_plane[h / 2 * linesize + w + 1] = w;
        }
      }
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  uint32_t device_id = args.get<uint32_t>("device_id");
  std::string input_file = args.get<std::string>("input_file");
  std::string output_file = args.get<std::string>("output_file");
  // init env
  vsx::SetDevice(device_id);
  // read image from file
  auto input_image = ReadImageFromFile(input_file, vsx::RGB_PLANAR);
  std::cout << "image width: " << input_image.Width()
            << ", height: " << input_image.Height() << std::endl;
  // copy to device
  vsx::Image image_vacc;
  CopyHostImageToDeviceImage(image_vacc, device_id, input_image);

  // copy to host
  vsx::Image image_cpu;
  CopyDeviceImageToHostImage(image_cpu, image_vacc);

  // change pixels
  ChangePixels(image_cpu);
  // write to file
  WriteImageToFile(output_file, image_cpu);
  return 0;
}