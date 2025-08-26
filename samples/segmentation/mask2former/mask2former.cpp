
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "common/mask2former.hpp"

#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>

#include "common/cmdline.hpp"
#include "common/file_system.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char **argv) {
  cmdline::parser args;
  args.add<std::string>("model_prefix", 'm',
                        "model prefix of the model suite files", false,
                        "/opt/vastai/vaststreamx/data/models/"
                        "mask2former-fp16-none-1_3_1024_1024-vacc/mod");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "../data/configs/mask2former_rgbplanar.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);

  args.add<float>("threshold", '\0', "threshold for detection", false, 0.5);
  args.add<std::string>("label_file", '\0', "label file", false,
                        "../data/labels/coco2id.txt");
  args.add<std::string>("input_file", '\0', "input image file", false,
                        "../data/images/cycling.jpg");
  args.add<std::string>("output_file", '\0', "output image file", false, "");
  args.add<std::string>("dataset_filelist", '\0', "input dataset filelist",
                        false, "");
  args.add<std::string>("dataset_root", '\0', "input dataset root", false, "");
  args.add<std::string>("dataset_output_file", '\0', "dataset output file",
                        false, "");
  args.parse_check(argc, argv);
  return args;
}

std::vector<cv::Vec3b> colors = {
    {62, 140, 230},  {255, 85, 0},  {255, 170, 0},  {255, 0, 85},
    {0, 255, 0},     {85, 255, 0},  {170, 255, 0},  {0, 255, 85},
    {0, 255, 170},   {0, 0, 255},   {85, 0, 255},   {170, 0, 255},
    {0, 85, 255},    {0, 170, 255}, {255, 255, 0},  {255, 255, 85},
    {255, 255, 170}, {255, 0, 170}, {255, 0, 255},  {255, 85, 255},
    {255, 170, 255}, {0, 255, 255}, {85, 255, 255}, {170, 255, 255},
};

// for retina_masks equal 1
void SaveMask(const std::vector<vsx::Tensor> &tensors,
              const std::string &origin_image, const std::string &output_image,
              const std::vector<std::string> &labels, float threshold) {
  const float *classes = tensors[0].Data<float>();
  const float *scores = tensors[1].Data<float>();
  const float *boxes = tensors[2].Data<float>();
  auto masks = tensors[3];
  uint32_t det_count = tensors[4].Data<uint32_t>()[0];

  uint8_t *masks_ptr = masks.MutableData<uint8_t>();
  const auto &mask_shape = masks.Shape();
  int mask_h = mask_shape[mask_shape.ndim() - 2],
      mask_w = mask_shape[mask_shape.ndim() - 1];
  cv::Mat cvMask = cv::Mat::zeros(mask_h, mask_w, CV_8UC3);

  for (uint32_t i = 0; i < det_count; i++) {
    if (scores[i] < threshold) continue;
    int mask_offset = i * mask_h * mask_w;
    cv::Mat mask(mask_h, mask_w, CV_8U,
                 reinterpret_cast<void *>(masks_ptr + mask_offset));
    cv::Mat cvtemp = cv::Mat::zeros(mask_h, mask_w, CV_8UC3);
    for (int y = 0; y < mask_h; ++y) {
      for (int x = 0; x < mask_w; ++x) {
        uchar pixelValue = mask.at<uchar>(y, x);
        int c = i % colors.size();
        cvtemp.at<cv::Vec3b>(y, x) =
            cv::Vec3b(pixelValue * colors[c][0], pixelValue * colors[c][1],
                      pixelValue * colors[c][2]);
      }
    }
    cvMask += cvtemp;
  }

  cv::Mat org_image = cv::imread(origin_image);
  org_image += cvMask;

  for (uint32_t i = 0; i < det_count; i++) {
    if (scores[i] < threshold) continue;
    auto class_id = static_cast<int>(classes[i] + 0.1);
    auto class_name = labels[class_id];
    auto score = scores[i];
    cv::Rect rect(cv::Point(boxes[i * 4 + 0], boxes[i * 4 + 1]),
                  cv::Point(boxes[i * 4 + 2], boxes[i * 4 + 3]));
    auto color = colors[i % colors.size()];
    cv::rectangle(org_image, rect, color);
    std::ostringstream oss;
    oss << class_name << std::fixed << std::setprecision(2) << ": "
        << score * 100 << "%";
    std::string text = oss.str();
    auto top_left = rect.tl();
    top_left.y = top_left.y - 15 > 15 ? top_left.y - 15 : top_left.y + 15;
    int thickness = 1;
    cv::putText(org_image, text, top_left, cv::FONT_HERSHEY_SIMPLEX, 0.5, color,
                thickness);
  }
  cv::imwrite(output_image, org_image);
  return;
}

int main(int argc, char **argv) {
  auto args = ArgumentParser(argc, argv);
  const int batch_size = 1;
  auto labels = vsx::LoadLabels(args.get<std::string>("label_file"));
  float threshold = args.get<float>("threshold");
  vsx::Mask2Former segmenter(args.get<std::string>("model_prefix"),
                             args.get<std::string>("vdsp_params"), batch_size,
                             args.get<uint32_t>("device_id"),
                             args.get<float>("threshold"),
                             args.get<std::string>("hw_config"));
  auto image_format = segmenter.GetFusionOpIimageFormat();

  if (args.get<std::string>("dataset_filelist").empty()) {
    vsx::Image vsx_image;
    CHECK(vsx::MakeVsxImage(args.get<std::string>("input_file"), vsx_image,
                            image_format) == 0);
    auto tensors = segmenter.Process(vsx_image);
    if (tensors.size() == 0) {
      std::cout << "No object detected in image.\n";
    } else {
      const float *classes = tensors[0].Data<float>();
      const float *scores = tensors[1].Data<float>();
      const float *boxes = tensors[2].Data<float>();
      uint32_t det_count = tensors[4].Data<uint32_t>()[0];
      for (uint32_t i = 0; i < det_count; ++i) {
        if (scores[i] < threshold) continue;
        auto class_name = labels[static_cast<int>(classes[i] + 0.1)];
        std::cout << "Object class: " << class_name << ", score: " << scores[i]
                  << ", bbox: [" << boxes[i * 4 + 0] << ", " << boxes[i * 4 + 1]
                  << ", " << boxes[i * 4 + 2] << ", " << boxes[i * 4 + 3]
                  << "]\n";
      }
      if (!args.get<std::string>("output_file").empty()) {
        SaveMask(tensors, args.get<std::string>("input_file"),
                 args.get<std::string>("output_file"), labels, threshold);
        std::cout << "Save result to " << args.get<std::string>("output_file")
                  << std::endl;
      }
    }
  } else {
    std::ofstream of(args.get<std::string>("dataset_output_file"));
    if (!of.is_open()) {
      std::cout << "Error, Failed to open: "
                << args.get<std::string>("dataset_output_file") << std::endl;
      return -1;
    }
    std::vector<std::string> filelist =
        vsx::ReadFileList(args.get<std::string>("dataset_filelist"));
    auto dataset_root = args.get<std::string>("dataset_root");
    auto result_array = nlohmann::json::array();

    for (size_t s = 0; s < filelist.size(); s++) {
      auto fullname = filelist[s];
      if (!dataset_root.empty()) fullname = dataset_root + "/" + fullname;
      std::cout << s + 1 << "/" << filelist.size() << " " << fullname
                << std::endl;
      std::filesystem::path p(fullname);
      int image_id = std::atoi(p.stem().string().c_str());
      vsx::Image vsx_image;
      CHECK(vsx::MakeVsxImage(fullname, vsx_image, image_format) == 0);
      auto tensors = segmenter.Process(vsx_image);
      if (tensors.size() > 0) {
        const float *classes = tensors[0].Data<float>();
        const float *scores = tensors[1].Data<float>();
        const float *bboxes = tensors[2].Data<float>();
        auto &mask = tensors[3];
        auto det_num = tensors[4].Data<uint32_t>()[0];
        for (uint32_t n = 0; n < det_num; n++) {
          nlohmann::json obj;
          auto jbox = nlohmann::json::array();
          const float *box = bboxes + n * 4;
          jbox.push_back(box[0]);
          jbox.push_back(box[1]);
          jbox.push_back(box[2] - box[0]);
          jbox.push_back(box[3] - box[1]);

          obj["image_id"] = image_id;
          obj["category_id"] =
              vsx::coco80_to_coco91_class(static_cast<int>(classes[n] + 0.1));
          obj["bbox"] = jbox;
          obj["score"] = scores[n];
          obj["segmentation"] = vsx::single_encode(mask, n);
          result_array.push_back(obj);
        }
      }
    }
    of << result_array.dump();
    of.close();
  }

  return 0;
}