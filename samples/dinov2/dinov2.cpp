
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <sstream>

#include "common/cmdline.hpp"
#include "common/dinov2_model.hpp"
#include "common/utils.hpp"
#include "read_pickle.hpp"
#include "utils.hpp"
#include "vaststreamx/vaststreamx.h"

cmdline::parser ArgumentParser(int argc, char** argv) {
  cmdline::parser args;
  args.add<std::string>("model_prefix", 'm',
                        "model prefix of the model suite files", false,
                        "/opt/vastai/vaststreamx/data/models/"
                        "dinov2-b-fp16-none-1_3_224_224-vacc/mod");
  args.add<std::string>("norm_elf_file", '\0', "normalize elf file path", false,
                        "/opt/vastai/vaststreamx/data/elf/normalize");
  args.add<std::string>("space_to_depth_elf_file", '\0',
                        "space to depth elf file path", false,
                        "/opt/vastai/vaststreamx/data/elf/space_to_depth");
  args.add<std::string>("hw_config", '\0', "hw-config file of the model suite",
                        false);
  args.add<uint32_t>("device_id", 'd', "device id to run", false, 0);
  args.add<std::string>("input_file", '\0', "input file", false,
                        "../data/images/oxford_003681.jpg");
  args.add<std::string>("dataset_root", '\0', "input dataset root", false, "");
  args.add<std::string>("dataset_conf", '\0', "dataset config file", false, "");
  args.parse_check(argc, argv);
  return args;
}

int main(int argc, char* argv[]) {
  auto args = ArgumentParser(argc, argv);
  const int batch_size = 1;
  vsx::Dinov2Model dinov2(args.get<std::string>("model_prefix"),
                          args.get<std::string>("norm_elf_file"),
                          args.get<std::string>("space_to_depth_elf_file"),
                          batch_size, args.get<uint32_t>("device_id"));

  if (args.get<std::string>("dataset_root").empty()) {
    auto image = cv::imread(args.get<std::string>("input_file"));
    auto result = dinov2.Process(image);
    const float* data = result.Data<float>();
    std::cout << "output:\n[";
    for (size_t i = 0; i < result.GetSize(); i++) {
      std::cout << data[i] << ",";
    }
    std::cout << "]\n";
  } else {
    auto dataset_root = args.get<std::string>("dataset_root");
    auto pickle_data = ReadPickleFile(args.get<std::string>("dataset_conf"));
    vsx::TShape shape;
    dinov2.GetOutputShapeByIndex(0, shape);
    int feature_len = shape.Size();

    vsx::Tensor train_features(
        {static_cast<int64_t>(pickle_data.imlist.size()), feature_len},
        vsx::Context::CPU(),
        vsx::kFloat32);  // 4993 * 1024
    vsx::Tensor query_features(
        {feature_len, static_cast<int64_t>(pickle_data.qimlist.size())},
        vsx::Context::CPU(),
        vsx::kFloat32);  // 1024 * 70
    // train_features
    auto train_data = train_features.MutableData<float>();
    for (auto file : pickle_data.imlist) {
      auto fullname = dataset_root + "/" + file + ".jpg";
      std::cout << fullname << std::endl;
      auto image = cv::imread(fullname);
      auto result = dinov2.Process(image);
      auto res_data = result.MutableData<float>();
      memcpy(train_data, res_data, result.GetDataBytes());
      train_data += result.GetSize();
    }

    // normalize
    int dim0 = train_features.Shape()[0];  // 4993
    int dim1 = train_features.Shape()[1];  // 1024
    train_data = train_features.MutableData<float>();
    for (int i = 0; i < dim1; i++) {
      double sum = 0;
      for (int j = 0; j < dim0; j++) {
        sum += train_data[j * dim1 + i] * train_data[j * dim1 + i];
      }
      sum = std::sqrt(sum);
      for (int j = 0; j < dim0; j++) {
        train_data[j * dim1 + i] /= sum;
      }
    }
    // query_features
    auto query_data = query_features.MutableData<float>();
    int img_count = pickle_data.qimlist.size();
    for (int i = 0; i < img_count; i++) {
      auto fullname = dataset_root + "/" + pickle_data.qimlist[i] + ".jpg";
      std::cout << fullname << std::endl;
      auto image = cv::imread(fullname);
      auto result = dinov2.Process(image);
      auto res_data = result.MutableData<float>();
      for (size_t j = 0; j < result.GetSize(); j++) {
        query_data[j * img_count + i] = res_data[j];
      }
    }
    dim0 = query_features.Shape()[0];
    dim1 = query_features.Shape()[1];
    for (int i = 0; i < dim0; i++) {
      vsx::Normalize<float>(query_data, dim1);
      query_data += dim1;
    }

    // matrix muliplication
    int M = pickle_data.imlist.size();
    int N = pickle_data.qimlist.size();
    int K = feature_len;

    vsx::Tensor ranks({M, N}, vsx::Context::CPU(), vsx::kInt32);
    auto A = train_features.Data<float>();
    auto B = query_features.Data<float>();
    auto ranks_data = ranks.MutableData<int>();

    for (int i = 0; i < N; i++) {
      std::vector<std::pair<double, int>> rank;
      rank.reserve(M);
      for (int j = 0; j < M; j++) {
        double sum = 0;
        for (int s = 0; s < K; s++) {
          sum += A[j * K + s] * B[s * N + i];
        }
        rank.emplace_back(sum, j);
      }
      std::sort(rank.begin(), rank.end(),
                [](std::pair<double, int>& a, std::pair<double, int>& b) {
                  return a.first > b.first;
                });
      for (int j = 0; j < M; j++) {
        ranks_data[j * N + i] = rank[j].second;
      }
    }

    auto& gnd = pickle_data.gnd;
    std::vector<int> ks{1, 5, 10};
    std::vector<std::pair<std::vector<int>, std::vector<int>>> gndt;
    gndt.reserve(gnd.size());
    for (size_t i = 0; i < gnd.size(); i++) {
      std::vector<int> ok(gnd[i].easy);
      ok.insert(ok.end(), gnd[i].hard.begin(), gnd[i].hard.end());
      gndt.push_back(std::make_pair(ok, gnd[i].junk));
    }

    // search for easy & hard
    float mapM;
    std::vector<float> apsM;
    std::vector<float> mprM;
    std::vector<std::vector<float>> prsM;
    compute_map(ranks, gndt, ks, mapM, apsM, mprM, prsM);

    gndt.clear();
    for (size_t i = 0; i < gnd.size(); i++) {
      std::vector<int> junk(gnd[i].junk);
      junk.insert(junk.end(), gnd[i].easy.begin(), gnd[i].easy.end());
      gndt.emplace_back(gnd[i].hard, std::move(junk));
    }
    float mapH;
    std::vector<float> apsH;
    std::vector<float> mprH;
    std::vector<std::vector<float>> prsH;
    compute_map(ranks, gndt, ks, mapH, apsH, mprH, prsH);

    std::stringstream outstr;
    outstr << "mAP M: " << mapM * 100 << ", H: " << mapH * 100 << std::endl;
    outstr << "mP@k[ ";
    for (auto k : ks) outstr << k << " ";
    outstr << "], M: [";
    for (auto m : mprM) outstr << m * 100 << " ";
    outstr << "], H: [";
    for (auto m : mprH) outstr << m * 100 << " ";
    outstr << "]\n";
    std::cout << outstr.str();
  }

  return 0;
}
