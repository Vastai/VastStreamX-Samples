
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
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "utils.hpp"
#include "vaststreamx/datatypes/frame_packet.h"

namespace vsx {

using std::chrono::duration_cast;
using std::chrono::microseconds;
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

struct MediaProfilerConfig {
  uint32_t instance;
  int iterations;
  std::vector<uint32_t> device_ids;
  std::vector<uint32_t> percentiles;
};

struct MediaProfilerResult {
  double throughput = 0;
  uint64_t latency_avg;
  uint64_t latency_max;
  uint64_t latency_min;
  std::vector<uint64_t> latency_pecents;
  const MediaProfilerConfig* config = nullptr;
};

template <typename T, typename P>
class MediaProfiler {
 public:
  using MediaType =
      std::invoke_result_t<decltype(std::mem_fn(&T::GetTestData)), T, bool>;

  MediaProfiler(const MediaProfilerConfig& config,
                std::vector<std::shared_ptr<T>> medias)
      : config_(&config), medias_(medias) {
    threads_.reserve(medias.size());
    long_time_test_ = config.iterations > 0 ? false : true;
    iters_left_ = long_time_test_ ? 1 : config.iterations;
    latency_begin_.reserve(iters_left_);
    latency_end_.reserve(iters_left_);
  }

  MediaProfilerResult Profiling() {
    for (size_t i = 0; i < medias_.size(); i++) {
      uint32_t device_id =
          config_->device_ids[i % (config_->device_ids.size())];
      threads_.emplace_back(&MediaProfiler::DriveOneInstance, this, i,
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
    MediaProfilerResult result;
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
    std::vector<time_point> ticks;
    std::vector<time_point> tocks;
    // typedef decltype(medias_[idx]->Process(media_data_, end_flag_))
    // MediaResult;

    std::atomic<bool> stopped = {false};
    std::atomic<int> send = {0};
    std::atomic<int> recv = {0};
    std::thread cunsume_thread([&] {
      while (!stopped || recv < send) {
        P param;
        if (!medias_[idx]->GetResult(param)) {
          break;
        }

        auto tock = std::chrono::high_resolution_clock::now();
        if (!long_time_test_) tocks.push_back(tock);
        ++recv;
      }
    });
    bool is_key_frame = false;
    auto start = std::chrono::high_resolution_clock::now();
    while (--iters_left_ >= 0 || !is_key_frame || long_time_test_) {
      auto media_data = medias_[idx]->GetTestData(true);
      is_key_frame = medias_[idx]->IsKeyFrame();
      if (long_time_test_) {
        int value = medias_[idx]->Process(media_data, false);
        if (value == 0) {
          ++send;
        }
      } else {
        auto tick = std::chrono::high_resolution_clock::now();
        int value = medias_[idx]->Process(media_data, false);
        if (value == 0) {
          ticks.push_back(tick);
          ++send;
        }
      }
    }

    stopped = true;
    medias_[idx]->Stop();
    cunsume_thread.join();
    auto end = std::chrono::high_resolution_clock::now();
    uint64_t used = duration_cast<microseconds>(end - start).count();
    throughput_ += static_cast<double>(ticks.size()) * 1000000 / used;
    CHECK(ticks.size() == tocks.size())
        << "ticks and tocks should be the same size,tick.size=" << ticks.size()
        << ", tocks.size=" << tocks.size();
    merge_mutex.lock();
    latency_begin_.insert(latency_begin_.end(), ticks.begin(), ticks.end());
    latency_end_.insert(latency_end_.end(), tocks.begin(), tocks.end());
    merge_mutex.unlock();
  }

 private:
  const MediaProfilerConfig* config_;
  std::vector<std::shared_ptr<T>> medias_;
  std::atomic<int> iters_left_;
  MediaType media_data_;
  bool end_flag_;
  std::vector<std::thread> threads_;
  double throughput_ = 0;
  std::vector<time_point> latency_begin_;
  std::vector<time_point> latency_end_;
  std::mutex merge_mutex;
  bool long_time_test_ = false;
};

}  // namespace vsx

inline std::ostream& operator<<(std::ostream& os,
                                const vsx::MediaProfilerResult& result) {
  os << "\n- ";
  if (result.config != nullptr) {
    os << "number of instances: " << result.config->instance << std::endl;
    os << "  devices: [ ";
    for (auto device_id : result.config->device_ids) {
      os << device_id << " ";
    }
    os << "]" << std::endl;
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
