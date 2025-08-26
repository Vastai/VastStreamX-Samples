#
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import time
import queue
import multiprocessing
import threading

# import concurrent.futures
from easydict import EasyDict as edict
import numpy as np
import vaststreamx as vsx


class MediaProfilerResult:
    def __init__(self) -> None:
        self.throughput = 0
        self.latency_avg = 0
        self.latency_max = 0
        self.latency_min = 0
        self.latency_pecents = []
        self.config = None

    def __str__(self) -> str:
        repr_str = "\n- "
        if self.config:
            repr_str += f"""number of instances: {self.config.instance}
  devices: {self.config.device_ids}
"""
        repr_str += f"""  throughput (qps): {self.throughput:.2f}
  latency (us):
    avg latency: {self.latency_avg}
    min latency: {self.latency_min}
    max latency: {self.latency_max}
"""
        if self.config:
            for i, pecent in enumerate(self.config.percentiles):
                repr_str += f"    p{pecent} latency: {self.latency_pecents[i]}\n"
        return repr_str


class MediaProfiler:
    def __init__(self, config, medias) -> None:
        self.manager = multiprocessing.Manager()
        self.config_ = config
        self.medias_ = medias
        self.iters_left_ = config.iterations / len(self.medias_)
        self.merge_lock = multiprocessing.Lock()
        self.throughput_ = self.manager.Value("d", 0.0)
        self.latency_begin_ = self.manager.list()
        self.latency_end_ = self.manager.list()
        if config.iterations > 0:
            self.long_time_test_ = False
        else:
            self.long_time_test_ = True

    def profiling(self):
        threads = []
        for i in range(len(self.medias_)):
            device_id = self.config_.device_ids[i % len(self.config_.device_ids)]
            thread_inst = multiprocessing.Process(
                target=self.drive_one_instance, args=(i, device_id)
            )
            thread_inst.start()
            threads.append(thread_inst)
        for thread_inst in threads:
            thread_inst.join()
        latency_us = (
            np.array(self.latency_end_) - np.array(self.latency_begin_)
        ) * 1000000
        result = MediaProfilerResult()
        result.latency_pecents = (
            np.percentile(latency_us, self.config_.percentiles + [0, 100])
            .astype("int")
            .tolist()
        )
        result.latency_max = result.latency_pecents.pop()
        result.latency_min = result.latency_pecents.pop()
        result.latency_avg = int(np.mean(latency_us))
        result.throughput = self.throughput_.get()
        result.config = self.config_
        return result

    def drive_one_instance(self, idx, device_id):
        assert vsx.set_device(device_id) == 0
        ticks = []
        tocks = []
        context = edict(stopped=False, left=0, send=0, recv=0)

        def cunsume_thread_func(context, idx, tocks):
            assert vsx.set_device(device_id) == 0
            while not context.stopped or context.recv < context.send:
                try:
                    result = self.medias_[idx].get_result()
                    tock = time.time()
                    if self.long_time_test_ is not True:
                        tocks.append(tock)
                    context.recv += 1
                except:
                    break

        cunsume_thread = threading.Thread(
            target=cunsume_thread_func, args=(context, idx, tocks)
        )
        cunsume_thread.start()
        start = time.time()

        while self.iters_left_ > 0 or self.long_time_test_:
            media_data = self.medias_[idx].get_test_data(True)
            tick = time.time()
            self.medias_[idx].process(media_data, False)

            self.iters_left_ -= 1
            context.send += 1
            if self.long_time_test_ is False:
                ticks.append(tick)

        context.stopped = True
        self.medias_[idx].stop()
        cunsume_thread.join()
        end = time.time()
        self.merge_lock.acquire()
        time_used = (end - start) * 1000000
        througput_temp = self.throughput_.get()
        througput_temp += len(ticks) * 1000000.0 / time_used
        self.throughput_.set(througput_temp)

        assert len(ticks) == len(tocks)
        self.latency_begin_ += ticks
        self.latency_end_ += tocks
        self.merge_lock.release()
