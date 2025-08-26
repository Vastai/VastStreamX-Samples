
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <chrono>
#include <mutex>

#include "common/cmdline.hpp"
#include "common/file_system.hpp"
#include "common/utils.hpp"
// #include "ocr_e2e_multi_thread.hpp"
#include "ocr_e2e_async.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>(
      "det_model", '\0', "text detection model prefix of the model suite files",
      false, "/opt/vastai/vaststreamx/data/models/det_model_vacc_fp16/mod");
  args.add<std::string>("det_config", '\0',
                        "text detection vdsp preprocess parameter file", false,
                        "../data/configs/dbnet_rgbplanar.json");
  args.add<std::string>(
      "cls_model", '\0',
      "text classification model prefix of the model suite files", false,
      "/opt/vastai/vaststreamx/data/models/cls_model_vacc_fp16/mod");
  args.add<std::string>("cls_config", '\0',
                        "text classification vdsp preprocess parameter file",
                        false, "../data/configs/crnn_rgbplanar.json");
  args.add<std::string>(
      "rec_model", '\0',
      "text recognition model prefix of the model suite files", false,
      "/opt/vastai/vaststreamx/data/models/rec_model_vacc_fp16/mod");
  args.add<std::string>("rec_config", '\0',
                        "text recognition vdsp preprocess parameter file",
                        false, "../data/configs/crnn_rgbplanar.json");

  args.add<std::string>("det_box_type", '\0', "text detection box type", false,
                        "quad");
  args.add<std::string>(
      "det_elf_file", '\0', "text detection elf file", false,
      "/opt/vastai/vaststreamx/data/elf/find_contours_ext_op");
  args.add<std::string>("cls_labels", '\0', "text classification label list",
                        false, "[0, 180]");
  args.add<float>("cls_thresh", '\0', "text classification thresh", false, 0.9);
  args.add<std::string>("rec_label_file", '\0', "text recognition label file",
                        false, "../data/labels/ppocr_keys_v1.txt");
  args.add<float>("rec_drop_score", '\0',
                  "text recogniztion drop score threshold", false, 0.5);
  args.add<int>("use_angle_cls", '\0', "use text classification", false, 1);
  args.add<uint32_t>("batch_size", '\0', "batch size of the model", false, 1);
  args.add<std::string>("device_ids", '\0', "device id to run", false, "[0]");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  args.add<std::string>("input_file", '\0', "input image", false,
                        "../data/images/word_336.png");
  args.add<std::string>("output_file", '\0', "output image file", false, "");
  args.add<std::string>("dataset_filelist", '\0', "input dataset filelist",
                        false, "");
  args.add<std::string>("dataset_root", '\0', "input dataset root", false, "");
  args.add<std::string>("dataset_output_file", '\0', "dataset output file",
                        false, "");
  args.add<uint32_t>("queue_size", '\0', "set queue size", false, 1);

  args.parse_check(argc, argv);
  return args;
}

void InferenceThread(std::shared_ptr<vsx::OCR_e2e_Async> model,
                     cmdline::parser& args, std::mutex& merge_mutex,
                     std::vector<int64_t>& costs, uint32_t device_id) {
  std::vector<time_point> ticks;
  std::vector<time_point> tocks;
  std::vector<std::string> filelist;
  vsx::SetDevice(device_id);
  // get output
  auto output_thread = std::thread([&]() {
    vsx::SetDevice(device_id);
    int index = 0;
    while (true) {
      std::vector<vsx::TextObject> text_objs;
      if (model->GetOutput(text_objs)) {
        tocks.push_back(std::chrono::high_resolution_clock::now());
        std::cout << "Thread " << device_id << " get " << filelist[index]
                  << " result.\n";
        if (filelist.size() == 1) {
          for (auto& item : text_objs) {
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
          // draw box to image
          if (!args.get<std::string>("output_file").empty()) {
            auto cv_image = cv::imread(filelist[0]);
            for (auto& item : text_objs) {
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
        index++;
      } else {
        break;
      }
    }
  });

  // one image test
  if (args.get<std::string>("dataset_filelist").empty()) {
    filelist.push_back(args.get<std::string>("input_file"));
    auto cv_image = cv::imread(filelist[0]);
    model->ProcessAsync(cv_image);
    model->Stop();
    output_thread.join();
    return;
  }

  // dataset test
  std::vector<std::string> namelist =
      vsx::ReadFileList(args.get<std::string>("dataset_filelist"));
  auto dataset_root = args.get<std::string>("dataset_root");
  for (size_t s = 0; s < namelist.size(); s++) {
    auto fullname = namelist[s];
    if (!dataset_root.empty()) fullname = dataset_root + "/" + fullname;
    filelist.push_back(fullname);
  }

  // set input
  for (size_t s = 0; s < filelist.size(); s++) {
    auto cv_image = cv::imread(filelist[s]);
    model->ProcessAsync(cv_image);
    ticks.push_back(std::chrono::high_resolution_clock::now());
  }
  // set stop
  model->Stop();
  // wait thread done
  output_thread.join();

  if (ticks.size() != tocks.size()) {
    std::cout << "Error! ticks.size() != tocks.szie(). ticks.size() = "
              << ticks.size() << ", tocks.size() = " << tocks.size()
              << std::endl;
  }

  merge_mutex.lock();
  for (size_t i = 0; i < ticks.size(); i++) {
    auto cost = std::chrono::duration_cast<std::chrono::milliseconds>(tocks[i] -
                                                                      ticks[i])
                    .count();
    costs.push_back(cost);
  }
  merge_mutex.unlock();
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  auto device_ids = vsx::ParseVecUint(args.get<std::string>("device_ids"));
  auto cls_labels = vsx::ParseVecUint(args.get<std::string>("cls_labels"));

  std::vector<std::shared_ptr<vsx::OCR_e2e_Async>> models;
  models.reserve(device_ids.size());
  for (auto device_id : device_ids) {
    auto model = std::make_shared<vsx::OCR_e2e_Async>(
        args.get<std::string>("det_model"), args.get<std::string>("det_config"),
        args.get<std::string>("det_box_type"),
        args.get<std::string>("det_elf_file"),
        args.get<std::string>("cls_model"), args.get<std::string>("cls_config"),
        cls_labels, args.get<float>("cls_thresh"),
        args.get<std::string>("rec_model"), args.get<std::string>("rec_config"),
        args.get<std::string>("rec_label_file"),
        args.get<float>("rec_drop_score"), args.get<int>("use_angle_cls"),
        args.get<uint32_t>("batch_size"), device_id,
        args.get<std::string>("hw_config"), args.get<uint32_t>("queue_size"));
    models.push_back(model);
  }

  std::mutex merge_mutex;
  std::vector<int64_t> costs;
  std::vector<std::thread> threads;
  threads.reserve(device_ids.size());

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < device_ids.size(); i++) {
    threads.emplace_back(InferenceThread, models[i], std::ref(args),
                         std::ref(merge_mutex), std::ref(costs), device_ids[i]);
  }

  for (auto& thread : threads) {
    thread.join();
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto total_cost =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  // test one image
  if (args.get<std::string>("dataset_filelist").empty()) {
    return 0;
  }

  // test dataset
  int64_t cost_sum = 0;
  for (auto cost : costs) cost_sum += cost;

  std::cout << "Image count: " << costs.size() << ", total cost: " << total_cost
            << " ms, throughput: " << costs.size() * 1000.0 / total_cost
            << " fps. Average latency: " << cost_sum * 1.0 / costs.size()
            << " ms. " << std::endl;

  return 0;
}