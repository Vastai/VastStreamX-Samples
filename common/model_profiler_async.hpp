
/*
 * Copyright (C) 2025 Vastai-tech Company.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#include "readerwritercircularbuffer.h"
#include "readerwriterqueue.h"
#include "utils.hpp"
#include "vaststreamx/datatypes/tensor.h"

namespace vsx {

using moodycamel::BlockingReaderWriterCircularBuffer;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

struct ProfilerConfig {
  uint32_t instance;
  int iterations;
  uint32_t batch_size;
  uint32_t data_type;
  std::vector<uint32_t> device_ids;
  std::vector<Context> contexts;
  std::vector<TShape> input_shapes;
  std::vector<uint32_t> percentiles;
  uint32_t queue_size = 4;
};

struct ProfilerResult {
  double throughput = 0;
  uint64_t latency_avg;
  uint64_t latency_max;
  uint64_t latency_min;
  std::vector<uint64_t> latency_pecents;
  const ProfilerConfig* config = nullptr;
};

template <typename T>
class ModelProfilerAsync {
 public:
  using InferType =
      std::invoke_result_t<decltype(std::mem_fn(&T::GetTestData)), T, uint32_t,
                           uint32_t, Context, std::vector<TShape>>;

  ModelProfilerAsync(const ProfilerConfig& config,
                     std::vector<std::shared_ptr<T>> models)
      : config_(&config), models_(models) {
    threads_.reserve(models.size());
    long_time_test_ = config.iterations > 0 ? false : true;
    iters_left_ = config.iterations > 0 ? config.iterations : 1;
    latency_begin_.reserve(iters_left_);
    latency_end_.reserve(iters_left_);
    CHECK(config.queue_size > 0) << ",queue_size(=" << config.queue_size
                                 << ") must be larger than 0 in async mode";
  }

  ProfilerResult Profiling() {
    for (size_t i = 0; i < models_.size(); i++) {
      uint32_t device_id =
          config_->device_ids[i % (config_->device_ids.size())];
      threads_.emplace_back(&ModelProfilerAsync::DriveOneInstance, this, i,
                            device_id);
    }
    for (auto& thd : threads_) {
      thd.join();
    }
    std::vector<uint64_t> latency_us;
    latency_us.reserve(latency_begin_.size());
    for (size_t i = 0; i < latency_begin_.size(); i++) {
      latency_us.push_back(
          duration_cast<microseconds>(latency_end_[i] - latency_begin_[i])
              .count());
    }
    ProfilerResult result;
    auto pcnts = config_->percentiles;
    auto idx = pcnts.size();
    pcnts.push_back(0);
    pcnts.push_back(100);
    result.latency_pecents = CalcPercentiles(latency_us, pcnts);
    result.latency_min = result.latency_pecents[idx];
    result.latency_max = result.latency_pecents[idx + 1];
    result.latency_pecents.resize(idx);
    result.latency_avg = CalcMean(latency_us);
    result.throughput = throughput_;
    result.config = config_;
    return result;
  }

 private:
  void DriveOneInstance(uint32_t idx, uint32_t device_id) {
    vsx::SetDevice(device_id);
    std::vector<time_point> ticks;
    std::vector<time_point> tocks;
    ticks.reserve(iters_left_);
    tocks.reserve(iters_left_);

    BlockingReaderWriterCircularBuffer<int> queue_futs(config_->queue_size);
    std::thread cunsume_thread([&] {
      std::vector<vsx::Tensor> outputs;
      while (models_[idx]->GetOutput(outputs)) {
        if (!long_time_test_) {
          auto tock = std::chrono::high_resolution_clock::now();
          merge_mutex.lock();
          tocks.push_back(tock);
          merge_mutex.unlock();
        }
        int input_id;
        queue_futs.wait_dequeue(input_id);
      }
    });
    auto infer_data = models_[idx]->GetTestData(
        config_->batch_size, config_->data_type, config_->contexts[idx],
        config_->input_shapes);

    auto start = std::chrono::high_resolution_clock::now();
    int input_id = 0;
    while (--iters_left_ >= 0 || long_time_test_) {
      queue_futs.wait_enqueue(input_id++);
      auto tick = std::chrono::high_resolution_clock::now();
      models_[idx]->ProcessAsync(infer_data);
      if (!long_time_test_) {
        merge_mutex.lock();
        ticks.push_back(tick);
        merge_mutex.unlock();
      }
    }
    models_[idx]->CloseInput();
    cunsume_thread.join();
    models_[idx]->WaitUntilDone();
    auto end = std::chrono::high_resolution_clock::now();
    merge_mutex.lock();
    uint64_t used = duration_cast<microseconds>(end - start).count();
    throughput_ += static_cast<double>(ticks.size()) * config_->batch_size *
                   1000000 / used;
    CHECK(ticks.size() == tocks.size())
        << "ticks and tocks should be the same size";
    latency_begin_.insert(latency_begin_.end(), ticks.begin(), ticks.end());
    latency_end_.insert(latency_end_.end(), tocks.begin(), tocks.end());
    merge_mutex.unlock();
  }

 private:
  const ProfilerConfig* config_;
  std::vector<std::shared_ptr<T>> models_;
  std::atomic<int> iters_left_;
  std::vector<std::thread> threads_;
  double throughput_ = 0;
  std::vector<time_point> latency_begin_;
  std::vector<time_point> latency_end_;
  std::mutex merge_mutex;
  float long_time_test_ = false;
};

}  // namespace vsx

inline std::ostream& operator<<(std::ostream& os,
                                const vsx::ProfilerResult& result) {
  os << "\n- ";
  if (result.config != nullptr) {
    os << "number of instances: " << result.config->instance << std::endl;
    os << "  devices: [ ";
    for (auto device_id : result.config->device_ids) {
      os << device_id << " ";
    }
    os << "]" << std::endl;
    os << "  queue size: " << result.config->queue_size << std::endl;
    os << "  batch size: " << result.config->batch_size << std::endl;
    os << "  ";
  }
  os << "throughput (qps): " << result.throughput << std::endl;
  os << "  latency (us):" << std::endl;
  os << "    avg latency: " << result.latency_avg << std::endl;
  os << "    min latency: " << result.latency_min << std::endl;
  os << "    max latency: " << result.latency_max << std::endl;
  if (result.config != nullptr) {
    for (size_t i = 0; i < result.config->percentiles.size(); i++) {
      os << "    p" << result.config->percentiles[i]
         << " latency: " << result.latency_pecents[i] << std::endl;
    }
  }
  os << std::endl;
  return os;
}
