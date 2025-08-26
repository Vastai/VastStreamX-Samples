
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/cmdline.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"
#include "vaststreamx/vaststreamx.h"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("input_file", '\0', "input image file", false,
                        "../data/images/dog.jpg");
  args.add<std::string>("output_file", '\0', "output image file", false,
                        "./image_process_result.jpg");
  args.parse_check(argc, argv);
  return args;
}
int main(int argc, char* argv[]) {
  auto args = ArgumentParser(argc, argv);
  uint32_t device_id = args.get<uint32_t>("device_id");
  std::string input_file = args.get<std::string>("input_file");
  std::string output_file = args.get<std::string>("output_file");

  vsx::Image image_bgr_interleave;
  if (vsx::MakeVsxImage(input_file, image_bgr_interleave,
                        vsx::ImageFormat::BGR_INTERLEAVE) != 0) {
    std::cout << "MakeVsxImage failed.\n";
    return -1;
  }

  if (vsx::SetDevice(device_id) != 0) {
    std::cout << "SetDevice failed.\n";
    return -1;
  }

  // cvtcolor sample
  vsx::Image image_rgb_planar;
  vsx::CvtColor(image_bgr_interleave, image_rgb_planar, vsx::RGB_PLANAR,
                vsx::ImageColorSpace::kCOLOR_SPACE_BT601);
  std::cout << "image_rgb_planar format is "
            << vsx::ImageFormatToString(image_rgb_planar.Format()) << std::endl;

  vsx::Image image_yuv_nv12;
  vsx::CvtColor(image_rgb_planar, image_yuv_nv12, vsx::YUV_NV12,
                vsx::ImageColorSpace::kCOLOR_SPACE_BT601, true);
  std::cout << "image_yuv_nv12 format is "
            << vsx::ImageFormatToString(image_yuv_nv12.Format()) << std::endl;
  vsx::Image image_gray;
  vsx::CvtColor(image_yuv_nv12, image_gray, vsx::GRAY,
                vsx::ImageColorSpace::kCOLOR_SPACE_BT601, true);
  std::cout << "image_gray format is "
            << vsx::ImageFormatToString(image_gray.Format()) << std::endl;

  // resize sample
  vsx::Image image_416_416;
  vsx::Resize(image_rgb_planar, image_416_416,
              vsx::kRESIZE_TYPE_BILINEAR_PILLOW, 416, 416);
  std::cout << "image_416_416 size is ( " << image_416_416.Width() << " x "
            << image_416_416.Height() << " )\n";
  vsx::Image image_600_800;
  vsx::Resize(image_yuv_nv12, image_600_800, vsx::kRESIZE_TYPE_BILINEAR_CV, 600,
              800);
  std::cout << "image_600_800 size is ( " << image_600_800.Width() << " x "
            << image_600_800.Height() << " )\n";

  // crop sample
  vsx::Image image_crop_224_224;
  vsx::Rect crop_rect;
  crop_rect.x = 0;
  crop_rect.y = 0;
  crop_rect.w = 224;
  crop_rect.h = 224;
  vsx::Crop(image_rgb_planar, image_crop_224_224, crop_rect);
  std::cout << "image_crop_224_224 size is ( " << image_crop_224_224.Width()
            << " x " << image_crop_224_224.Height() << " )\n";

  // yuvflip sample, just support yuv_nv12 format image
  vsx::Image image_flip_x, image_flip_y;
  vsx::YuvFlip(image_yuv_nv12, image_flip_x, vsx::kFLIP_TYPE_X_AXIS);
  vsx::YuvFlip(image_yuv_nv12, image_flip_y, vsx::kFLIP_TYPE_Y_AXIS);
  std::cout << "image_flip_x size is ( " << image_flip_x.Width() << " x "
            << image_flip_x.Height() << " )\n";
  std::cout << "image_flip_y size is ( " << image_flip_y.Width() << " x "
            << image_flip_y.Height() << " )\n";

  // warpaffine sample
  vsx::Image image_warpaffine;
  vsx::WarpAffine(image_yuv_nv12, image_warpaffine, {0.5, 0, 0, 0, 0.5, 0},
                  vsx::kWARP_AFFINE_MODE_BILINEAR, vsx::kPADDING_TYPE_CONSTANT,
                  {128, 128, 0});
  std::cout << "image_warpaffine size is ( " << image_warpaffine.Width()
            << " x " << image_warpaffine.Height() << " )\n";

  // resize_copy_make_border sample, resize to (w x h ) = (512 x 5120 and
  // padding to (w x h) = (600 x 800) padding_edges = (top,bottom,left,right)
  vsx::Image image_resize_copy_make_border;
  int resize_width = 512, resize_height = 512;
  vsx::ImagePaddingEdges padding_edges = {114, 114, 44, 44};
  vsx::ResizeCopyMakeBorder(image_bgr_interleave, image_resize_copy_make_border,
                            vsx::kRESIZE_TYPE_NEAREST, resize_width,
                            resize_height, vsx::kPADDING_TYPE_CONSTANT,
                            {128, 128, 0}, &padding_edges);
  std::cout << "image_resize_copy_make_border size is ( "
            << image_resize_copy_make_border.Width() << " x "
            << image_resize_copy_make_border.Height() << " )\n";

  // batch_crop_resize sample ,crop image with rectangles and
  // resize to specific size
  std::vector<vsx::Image> images_batch_crop_resize;
  std::vector<vsx::Rect> crop_rects;
  crop_rect.x = 0;
  crop_rect.y = 0;
  crop_rect.w = 200;
  crop_rect.h = 200;
  crop_rects.push_back(crop_rect);

  crop_rect.x = 50;
  crop_rect.y = 100;
  crop_rect.w = 150;
  crop_rect.h = 200;
  crop_rects.push_back(crop_rect);

  vsx::BatchCropResize(image_bgr_interleave, images_batch_crop_resize,
                       crop_rects, vsx::kRESIZE_TYPE_BILINEAR_PILLOW, 640, 640);
  for (size_t i = 0; i < images_batch_crop_resize.size(); i++) {
    auto img = images_batch_crop_resize[i];
    std::cout << "image_resize_copy_make_border[" << i << "] size is ( "
              << img.Width() << " x " << img.Height() << " )\n";
  }

  // scale sample
  std::vector<vsx::Image> images_scale;
  vsx::Scale(image_yuv_nv12, images_scale, vsx::kRESIZE_TYPE_BILINEAR,
             {{224, 224}, {416, 416}, {800, 600}});
  for (size_t i = 0; i < images_scale.size(); i++) {
    auto img = images_scale[i];
    std::cout << "images_scale[" << i << "] size is ( " << img.Width() << " x "
              << img.Height() << " )\n";
  }

  // save image_600_800 to file
  if (image_600_800.GetContext().dev_type == vsx::Context::kVACC) {
    // image_600_800 memory is in device,copy to host
    vsx::Image temp(image_600_800.Format(), image_600_800.Width(),
                    image_600_800.Height(), image_600_800.WidthPitch(),
                    image_600_800.HeightPitch());
    temp.CopyFrom(image_600_800);
    image_600_800 = temp;
  }
  cv::Mat save_mat_bgr;
  vsx::ConvertVsxImageToCvMatBgrPacked(image_600_800, save_mat_bgr);
  cv::imwrite(output_file, save_mat_bgr);
  return 0;
}