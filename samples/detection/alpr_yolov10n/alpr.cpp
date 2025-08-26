/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <opencv2/imgcodecs.hpp>
#include <sstream>

#include "common/cmdline.hpp"
#include "common/detector.hpp"
#include "common/text_rec.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char **argv) {
  cmdline::parser args;
  args.add<std::string>("yolov10_model_prefix", '\0',
                        "yolov10 model prefix of the model suite files", false,
                        "/home/aico/Downloads/docker/yolov10n/deploy_weights/"
                        "yolov10n_alpr/mod");
  args.add<std::string>("ocrv4_model_prefix", '\0',
                        "ocrv4 model prefix of the model suite files", false,
                        "/home/aico/Downloads/docker/ocrv4/deploy_weights/"
                        "PP-OCRv4_rec_infer/mod");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  args.add<std::string>("yolov10_vdsp_params", '\0',
                        "vdsp preprocess parameter file", false,
                        "/home/aico/Downloads/docker/yolov10n/"
                        "official-yolov10n-vdsp_params.json");
  args.add<std::string>(
      "ocrv4_vdsp_params", '\0', "ocrv4 vdsp preprocess parameter file", false,
      "/home/aico/Downloads/docker/ocrv4/ppocr-v4-rec-vdsp_params.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<float>("threshold", 't', "threshold for result", false, 0.25);
  args.add<std::string>("yolov10_label_file", '\0', "label file", false,
                        "../data/labels/alpr.txt");
  args.add<std::string>("ocrv4_label_file", '\0', "label file", false,
                        "/home/aico/Downloads/docker/ocrv4/ppocr_keys_v1.txt");
  args.add<std::string>("input_file", '\0', "input file", false,
                        "/home/aico/Downloads/images/double_yellow.jpg");
  args.add<std::string>("output_file", '\0', "output file", false,
                        "result.png");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char **argv) {
  auto args = ArgumentParser(argc, argv);
  const int batch_size = 1;
  std::vector<vsx::Image> alpr_images;
  std::vector<std::string> alpr_images_info;
  // yolov10
  {
    auto labels = vsx::LoadLabels(args.get<std::string>("yolov10_label_file"));
    vsx::Detector detector(args.get<std::string>("yolov10_model_prefix"),
                           args.get<std::string>("yolov10_vdsp_params"),
                           batch_size, args.get<uint32_t>("device_id"));
    detector.SetThreshold(args.get<float>("threshold"));
    auto image_format = detector.GetFusionOpIimageFormat();

    auto cv_image = cv::imread(args.get<std::string>("input_file"));
    CHECK(!cv_image.empty())
        << "Failed to read image:" << args.get<std::string>("input_file")
        << std::endl;
    vsx::Image vsx_image;
    CHECK(vsx::MakeVsxImage(cv_image, vsx_image, image_format) == 0);
    auto result = detector.Process(vsx_image);
    auto res_shape = result.Shape();
    const float *res_data = result.Data<float>();
    std::cout << "Detection objects:\n";
    for (int i = 0; i < res_shape[0]; i++) {
      if (res_data[0] < 0) break;
      std::string class_name = labels[static_cast<int>(res_data[0])];
      float score = res_data[1];
      std::stringstream istream;
      istream << "Object class: " << class_name << ", score: " << score
              << ", bbox: [" << res_data[2] << ", " << res_data[3]
              << res_data[4] << ", " << res_data[5] << "], number: ";
      alpr_images_info.push_back(istream.str());

      vsx::Image out;
      vsx::Rect rect_vsx(res_data[2], res_data[3], res_data[4], res_data[5]);
      if (static_cast<int>(res_data[0]) == 2) {
        cv::Rect2f rect = {res_data[2], res_data[3], res_data[4], res_data[5]};
        auto img = cv_image(rect);
        int h = img.rows;
        cv::Mat img_upper =
            img(cv::Range(0, static_cast<int>(5.0 / 12 * h)), cv::Range::all());
        cv::Mat img_lower =
            img(cv::Range(static_cast<int>(1.0 / 3 * h), h), cv::Range::all());
        cv::resize(img_upper, img_upper,
                   cv::Size(img_lower.cols, img_lower.rows));
        cv::Mat new_img;
        cv::hconcat(img_upper, img_lower, new_img);
        vsx::MakeVsxImage(new_img, out, image_format);
      } else {
        vsx::Crop(vsx_image, out, rect_vsx);
      }
      // draw bbox in cv_image
      cv::Rect2f rect = {res_data[2], res_data[3], res_data[4], res_data[5]};
      cv::rectangle(cv_image, rect, cv::Scalar(0, 255, 0), 2);
      // std::cout << i << " shape is: " << out.Shape() << std::endl;
      // cv::Mat cv_out;
      // vsx::ConvertVsxImageToCvMatBgrPacked(out.Clone(), cv_out);
      // cv::imwrite(std::to_string(i) + ".png", cv_out);
      alpr_images.push_back(out);
      res_data += vsx::kDetectionOffset;
    }
    cv::imwrite(args.get<std::string>("output_file"), cv_image);
  }
  // ocrv4
  {
    vsx::TextRecognizer text_rec(args.get<std::string>("ocrv4_model_prefix"),
                                 args.get<std::string>("ocrv4_vdsp_params"),
                                 batch_size, args.get<uint32_t>("device_id"),
                                 args.get<std::string>("ocrv4_label_file"),
                                 args.get<std::string>("hw_config"));

    for (size_t i = 0; i < alpr_images.size(); i++) {
      auto &vsx_image = alpr_images[i];
      auto result = text_rec.Process(vsx_image);
      auto result_str = vsx::GetStringFromTensor(result);
      alpr_images_info[i] += result_str;
    }
  }
  // print
  {
    for (const auto &info : alpr_images_info) std::cout << info << std::endl;
  }

  return 0;
}
