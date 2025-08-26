
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <thread>

#include "common/classifier_async.hpp"
#include "common/cmdline.hpp"
#include "common/file_system.hpp"
#include "common/readerwritercircularbuffer.h"
#include "common/utils.hpp"
#include "opencv2/opencv.hpp"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("model_prefix", 'm',
                        "model prefix of the model suite files", false,
                        "/opt/vastai/vaststreamx/data/models/"
                        "resnet50-int8-percentile-1_3_224_224-vacc/mod");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  args.add<std::string>("vdsp_params", '\0', "vdsp preprocess parameter file",
                        false, "../data/configs/resnet_bgr888.json");
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("label_file", '\0', "label file", false,
                        "../data/labels/imagenet.txt");
  args.add<std::string>("input_file", '\0', "input file", false,
                        "../data/images/cat.jpg");
  args.add<std::string>("dataset_filelist", '\0', "input dataset filelist",
                        false, "");
  args.add<std::string>("dataset_root", '\0', "input dataset root", false, "");
  args.add<std::string>("dataset_output_file", '\0', "dataset output file",
                        false, "");
  args.parse_check(argc, argv);
  return args;
}

std::vector<int> Argsort(const vsx::Tensor& array) {
  const int array_len = array.GetSize();
  const float* array_data = array.Data<float>();
  std::vector<int> array_index(array_len);
  for (int i = 0; i < array_len; ++i) array_index[i] = i;

  std::sort(array_index.begin(), array_index.end(), [=](int pos1, int pos2) {
    return (array_data[pos1] > array_data[pos2]);
  });

  return array_index;
}

int main(int argc, char** argv) {
  auto args = ArgumentParser(argc, argv);
  auto labels = vsx::LoadLabels(args.get<std::string>("label_file"));
  const int batch_size = 1;
  vsx::ClassifierAsync classifier(args.get<std::string>("model_prefix"),
                                  args.get<std::string>("vdsp_params"),
                                  batch_size, args.get<uint32_t>("device_id"));
  auto image_format = classifier.GetFusionOpIimageFormat();

  if (args.get<std::string>("dataset_filelist").empty()) {
    auto get_output_thread = std::thread([&]() {
      std::vector<vsx::Tensor> outputs;
      while (classifier.GetOutput(outputs)) {
        for (auto out : outputs) {
          auto index = Argsort(out);
          const float* array_data = out.Data<float>();
          std::cout << "Top5:\n";
          for (int i = 0; i < 5; i++) {
            std::cout << i << "th, class name: " << labels[index[i]]
                      << ", score: " << array_data[index[i]] << std::endl;
          }
        }
      }
    });
    vsx::Image vsx_image;
    CHECK(vsx::MakeVsxImage(args.get<std::string>("input_file"), vsx_image,
                            image_format) == 0);
    // send input
    classifier.ProcessAsync(vsx_image);
    // close input in order to let get_output_thread exit normally
    classifier.CloseInput();
    // wait thread exit
    get_output_thread.join();
    classifier.WaitUntilDone();
  } else {
    std::vector<std::string> filelist =
        vsx::ReadFileList(args.get<std::string>("dataset_filelist"));
    auto dataset_root = args.get<std::string>("dataset_root");
    std::ofstream outfile(args.get<std::string>("dataset_output_file"));
    CHECK(outfile.is_open())
        << "Failed to open: " << args.get<std::string>("dataset_output_file");
    moodycamel::BlockingReaderWriterCircularBuffer<std::string> filename_queue(
        50);
    auto get_output_thread = std::thread([&]() {
      while (true) {
        std::vector<vsx::Tensor> outputs;
        if (!classifier.GetOutput(outputs)) break;
        std::string file;
        filename_queue.wait_dequeue(file);
        for (auto out : outputs) {
          auto index = Argsort(out);
          const float* array_data = out.Data<float>();
          for (int i = 0; i < 5; i++) {
            outfile << file << ": "
                    << "top-" << i << " id: " << index[i]
                    << ", prob: " << std::setprecision(8)
                    << array_data[index[i]]
                    << ", class name: " << labels[index[i]] << std::endl;
          }
        }
      }
    });
    for (auto file : filelist) {
      auto fullname = file;
      if (!dataset_root.empty()) fullname = dataset_root + "/" + file;
      std::cout << fullname << std::endl;
      vsx::Image vsx_image;
      CHECK(vsx::MakeVsxImage(fullname, vsx_image, image_format) == 0);
      classifier.ProcessAsync(vsx_image);
      filename_queue.wait_enqueue(file);
    }
    classifier.CloseInput();
    get_output_thread.join();
    classifier.WaitUntilDone();
    outfile.close();
  }

  return 0;
}