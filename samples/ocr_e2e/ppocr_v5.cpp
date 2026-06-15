
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "ppocr_v5.hpp"

#include <chrono>
#include <thread>

#include "common/cmdline.hpp"
#include "common/file_system.hpp"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  // document image orientation classification
  args.add<std::string>(
      "doc_ori_model", '\0',
      "document image orientation classify model prefix of the model suite "
      "files",
      false, "/opt/vastai/vaststreamx/data/models/det_model_vacc_fp16/mod");
  args.add<std::string>(
      "doc_ori_config", '\0',
      "document image orientation classify vdsp preprocess parameter file",
      false, "../data/configs/dbnet_rgbplanar.json");
  args.add<std::string>(
      "doc_ori_label_file", '\0',
      "document image orientation classify vdsp preprocess parameter file",
      false, "../data/configs/dbnet_rgbplanar.json");
  args.add<bool>("use_doc_ori_cls", '\0',
                 "use image orientation classification", false, false);
  // text detection
  args.add<std::string>(
      "det_model", '\0', "text detection model prefix of the model suite files",
      false, "/opt/vastai/vaststreamx/data/models/det_model_vacc_fp16/mod");
  args.add<std::string>("det_config", '\0',
                        "text detection vdsp preprocess parameter file", false,
                        "../data/configs/dbnet_rgbplanar.json");
  args.add<std::string>(
      "det_elf_file", '\0', "text detection elf file", false,
      "/opt/vastai/vaststreamx/data/elf/find_contours_ext_op");

  args.add<std::string>("det_box_type", '\0', "text detection box type", false,
                        "quad");
  // textline orientation classification
  args.add<std::string>(
      "text_ori_model", '\0',
      "textline orientation classification model prefix of the model suite "
      "files",
      false, "/opt/vastai/vaststreamx/data/models/cls_model_vacc_fp16/mod");
  args.add<std::string>("text_ori_config", '\0',
                        "text classification vdsp preprocess parameter file",
                        false, "../data/configs/crnn_rgbplanar.json");
  args.add<float>("text_ori_thresh", '\0', "text classification thresh", false,
                  0.9);
  args.add<bool>("use_text_ori_cls", '\0', "use text classification", false,
                 true);
  // text recognition
  args.add<std::string>(
      "rec_model", '\0',
      "text recognition model prefix of the model suite files", false,
      "/opt/vastai/vaststreamx/data/models/rec_model_vacc_fp16/mod");
  args.add<std::string>("rec_config", '\0',
                        "text recognition vdsp preprocess parameter file",
                        false, "../data/configs/crnn_rgbplanar.json");
  args.add<std::string>("rec_label_file", '\0', "text recognition label file",
                        false, "../data/labels/ppocr_keys_v1.txt");
  args.add<float>("rec_drop_score", '\0',
                  "text recogniztion drop score threshold", false, 0.5);
  // vdsp op file
  args.add<std::string>(
      "rotate_elf", '\0', "rotate op elf file", false,
      "/opt/vastai/vaststreamx/data/elf/simple_rotate_ext_op");
  args.add<std::string>(
      "warp_perspective_elf", '\0', "warp perspective op elf file", false,
      "/opt/vastai/vaststreamx/data/elf/warp_perspective_ext_op");
  // common config
  args.add<uint32_t>("batch_size", '\0', "batch size of the model", false, 1);
  args.add<std::string>("device_ids", '\0', "device id to run", false, "[0]");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  // test files
  args.add<std::string>("input_file", '\0', "input image", false,
                        "../data/images/ppocr.jpg");
  args.add<std::string>("output_file", '\0', "output image file", false, "");
  args.add<std::string>("dataset_filelist", '\0', "input dataset filelist",
                        false, "");
  args.add<std::string>("dataset_root", '\0', "input dataset root", false, "");
  args.add<std::string>("dataset_output_file", '\0', "dataset output file",
                        false, "dataset_output.txt");
  args.parse_check(argc, argv);
  return args;
}

void InferenceThread(std::shared_ptr<vsx::PPOCR_v5> model,
                     cmdline::parser& args, std::mutex& merge_mutex,
                     std::vector<int64_t>& costs, uint32_t device_id,
                     float& throughput) {
  vsx::SetDevice(device_id);

  // test one image
  if (args.get<std::string>("dataset_filelist").empty()) {
    auto cv_image = cv::imread(args.get<std::string>("input_file"));
    CHECK(!cv_image.empty())
        << "Failed to read image:" << args.get<std::string>("input_file")
        << std::endl;
    vsx::Image vsx_image;
    vsx::MakeVsxImage(cv_image, vsx_image, vsx::ImageFormat::RGB_PLANAR);
    int rotate_angle = 0;
    auto result = model->Process(vsx_image, rotate_angle);
    if (result.empty()) {
      std::cout << "No text detected in image:"
                << args.get<std::string>("input_file") << std::endl;
    } else {
      std::cout << "Thread " << device_id << " get "
                << args.get<std::string>("input_file") << " result:\n";
      for (auto& item : result) {
        auto coor = std::get<0>(item);
        auto score = std::get<1>(item);
        auto str = std::get<2>(item);
        std::cout << "bbox:[ [" << static_cast<int>(coor[0]) << " "
                  << static_cast<int>(coor[1]) << "] ["
                  << static_cast<int>(coor[2]) << " "
                  << static_cast<int>(coor[3]) << "] ["
                  << static_cast<int>(coor[4]) << " "
                  << static_cast<int>(coor[5]) << "] [ "
                  << static_cast<int>(coor[6]) << " "
                  << static_cast<int>(coor[7]) << "] ], score: " << score
                  << ", string: " << str << std::endl;
      }
      if (args.get<std::string>("output_file") != "") {
        if (rotate_angle == 90 || rotate_angle == 270) {
          cv::rotate(cv_image, cv_image, cv::ROTATE_90_COUNTERCLOCKWISE);
        } else if (rotate_angle == 180) {
          cv::rotate(cv_image, cv_image, cv::ROTATE_180);
        }
        for (auto& item : result) {
          auto coor = std::get<0>(item);
          auto str = std::get<2>(item);
          cv::line(cv_image, cv::Point2f(coor[0], coor[1]),
                   cv::Point2f(coor[2], coor[3]), cv::Scalar(0, 0, 255));
          cv::line(cv_image, cv::Point2f(coor[2], coor[3]),
                   cv::Point2f(coor[4], coor[5]), cv::Scalar(0, 0, 255));
          cv::line(cv_image, cv::Point2f(coor[4], coor[5]),
                   cv::Point2f(coor[6], coor[7]), cv::Scalar(0, 0, 255));
          cv::line(cv_image, cv::Point2f(coor[0], coor[1]),
                   cv::Point2f(coor[6], coor[7]), cv::Scalar(0, 0, 255));
        }
        if (rotate_angle == 90 || rotate_angle == 270) {
          cv::rotate(cv_image, cv_image, cv::ROTATE_90_CLOCKWISE);
        } else if (rotate_angle == 180) {
          cv::rotate(cv_image, cv_image, cv::ROTATE_180);
        }

        fs::path output_path = args.get<std::string>("output_file");
        auto dir = output_path.parent_path().string();
        if (dir.empty()) dir = ".";
        auto filename = output_path.filename().string();
        auto save_file =
            dir + "/thread_" + std::to_string(device_id) + "_" + filename;
        std::cout << "Save file to: " << save_file << std::endl;
        cv::imwrite(save_file, cv_image);
      }
    }
    return;
  }

  // test dataset
  std::vector<std::string> filelist =
      vsx::ReadFileList(args.get<std::string>("dataset_filelist"));
  auto dataset_root = args.get<std::string>("dataset_root");
  fs::path output_path = args.get<std::string>("dataset_output_file");
  auto dir = output_path.parent_path().string();
  auto filename = output_path.filename().string();
  auto save_file =
      dir + "/thread_" + std::to_string(device_id) + "_" + filename;
  std::ofstream outfile(save_file, std::ios::out);
  CHECK(outfile.is_open()) << "Failed to open " << save_file;
  std::vector<time_point> ticks;
  std::vector<time_point> tocks;
  for (size_t s = 0; s < filelist.size(); s++) {
    auto fullname = filelist[s];
    if (!dataset_root.empty()) fullname = dataset_root + "/" + fullname;
    std::cout << "Thread: " << device_id << "," << fullname << std::endl;
    auto cv_image = cv::imread(fullname);
    vsx::Image vsx_image;
    vsx::MakeVsxImage(cv_image, vsx_image, vsx::ImageFormat::RGB_PLANAR);
    ticks.push_back(std::chrono::high_resolution_clock::now());
    int rotate_angle = 0;
    auto result = model->Process(vsx_image, rotate_angle);
    tocks.push_back(std::chrono::high_resolution_clock::now());

    for (auto& item : result) {
      auto coor = std::get<0>(item);
      auto score = std::get<1>(item);
      auto str = std::get<2>(item);
      outfile << "bbox:[ [" << static_cast<int>(coor[0]) << " "
              << static_cast<int>(coor[1]) << "] [" << static_cast<int>(coor[2])
              << " " << static_cast<int>(coor[3]) << "] ["
              << static_cast<int>(coor[4]) << " " << static_cast<int>(coor[5])
              << "] [ " << static_cast<int>(coor[6]) << " "
              << static_cast<int>(coor[7]) << "] ], score: " << score
              << ", string: " << str << std::endl;
    }
  }
  outfile.close();

  if (ticks.size() != tocks.size()) {
    std::cout << "Error! ticks.size() != tocks.szie(). ticks.size() = "
              << ticks.size() << ", tocks.size() = " << tocks.size()
              << std::endl;
  }

  int64_t cost_sum = 0;
  merge_mutex.lock();
  for (size_t i = 0; i < ticks.size(); i++) {
    auto cost = std::chrono::duration_cast<std::chrono::milliseconds>(tocks[i] -
                                                                      ticks[i])
                    .count();
    cost_sum += cost;
    costs.push_back(cost);
  }
  throughput += ticks.size() * 1000.0f / cost_sum;
  merge_mutex.unlock();
}
std::vector<std::vector<int>> get_doc_ori_labels(
    const std::string& label_file) {
  auto lines = vsx::LoadLabels(label_file);
  std::vector<std::vector<int>> labels;
  for (auto& line : lines) {
    int index, angle;
    std::istringstream iss(line);
    if (iss >> index >> angle)
      labels.push_back({index, angle});
    else {
      std::cerr << "Parsing label file Failed. line:" << line << std::endl;
      return {{}};
    }
  }
  return labels;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);

  auto device_ids = vsx::ParseVecUint(args.get<std::string>("device_ids"));
  auto use_text_ori_cls = args.get<bool>("use_text_ori_cls");
  auto use_doc_ori_cls = args.get<bool>("use_doc_ori_cls");

  std::vector<std::vector<int>> doc_ori_labels;
  if (use_doc_ori_cls) {
    doc_ori_labels =
        get_doc_ori_labels(args.get<std::string>("doc_ori_label_file"));
  }

  std::vector<std::shared_ptr<vsx::PPOCR_v5>> models;
  models.reserve(device_ids.size());
  for (auto device_id : device_ids) {
    auto model = std::make_shared<vsx::PPOCR_v5>(
        // document image orientation classify
        args.get<std::string>("doc_ori_model"),
        args.get<std::string>("doc_ori_config"), doc_ori_labels,
        use_doc_ori_cls,
        // text detection
        args.get<std::string>("det_model"), args.get<std::string>("det_config"),
        args.get<std::string>("det_box_type"),
        args.get<std::string>("det_elf_file"),
        // textline orientation classify
        args.get<std::string>("text_ori_model"),
        args.get<std::string>("text_ori_config"),
        args.get<float>("text_ori_thresh"), use_text_ori_cls,
        // text recognition
        args.get<std::string>("rec_model"), args.get<std::string>("rec_config"),
        args.get<std::string>("rec_label_file"),
        args.get<float>("rec_drop_score"),
        // vdsp op
        args.get<std::string>("rotate_elf"),
        args.get<std::string>("warp_perspective_elf"),
        // common
        args.get<uint32_t>("batch_size"), device_id,
        args.get<std::string>("hw_config"));
    models.push_back(model);
  }

  std::mutex merge_mutex;
  std::vector<int64_t> costs;
  std::vector<std::thread> threads;
  float throughput = 0;
  threads.reserve(device_ids.size());

  for (size_t i = 0; i < device_ids.size(); i++) {
    threads.emplace_back(InferenceThread, models[i], std::ref(args),
                         std::ref(merge_mutex), std::ref(costs), device_ids[i],
                         std::ref(throughput));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // test one image
  if (args.get<std::string>("dataset_filelist").empty()) {
    return 0;
  }
  // test dataset
  int64_t cost_sum = 0;
  for (auto cost : costs) cost_sum += cost;
  float avg_cost = cost_sum * 1.0 / costs.size();
  cost_sum /= device_ids.size();

  std::cout << "Image count: " << costs.size() << ", total cost: " << cost_sum
            << " ms, throughput: " << throughput
            << " fps. Average latency: " << avg_cost << " ms. " << std::endl;
  return 0;
}