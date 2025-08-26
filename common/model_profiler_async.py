import time
import asyncio
import queue
import threading
import concurrent.futures
from easydict import EasyDict as edict
import numpy as np


class ProfilerResult:
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
  batch size: {self.config.batch_size}
  queue size: {self.config.queue_size}
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


class ModelProfilerAsync:
    def __init__(self, config, models) -> None:
        self.config_ = config
        self.models_ = models
        self.iters_left_ = config.iterations
        self.merge_lock = threading.Lock()
        self.throughput_ = 0
        self.latency_begin_ = []
        self.latency_end_ = []
        assert config.queue_size > 0, "queue_size must be larger than 0 in async mode"

    def profiling(self):
        threads = []
        for i in range(len(self.models_)):
            thread_inst = threading.Thread(target=self.drive_one_instance, args=(i,))
            thread_inst.start()
            threads.append(thread_inst)
        for thread_inst in threads:
            thread_inst.join()
        latency_us = (
            np.array(self.latency_end_) - np.array(self.latency_begin_)
        ) * 1000000
        result = ProfilerResult()
        result.latency_pecents = (
            np.percentile(latency_us, self.config_.percentiles + [0, 100])
            .astype("int")
            .tolist()
        )
        result.latency_max = result.latency_pecents.pop()
        result.latency_min = result.latency_pecents.pop()
        result.latency_avg = int(np.mean(latency_us))
        result.throughput = self.throughput_
        result.config = self.config_
        return result

    def process_async(self, model, input):
        return model.process(input)

    def drive_one_instance(self, idx):
        infer_data = self.models_[idx].get_test_data(
            self.config_.data_type,
            self.config_.input_shape,
            self.config_.batch_size,
            self.config_.contexts[idx],
        )
        que = queue.Queue(self.config_.queue_size)
        ticks = []
        tocks = []

        def cunsume_thread_func(que, tocks):
            while True:
                try:
                    self.models_[idx].get_output()
                    tock = time.time()
                    tocks.append(tock)
                    que.get()
                except ValueError:
                    break

        cunsume_thread = threading.Thread(target=cunsume_thread_func, args=(que, tocks))
        cunsume_thread.start()
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while self.iters_left_ >= 0:
                tick = time.time()
                self.models_[idx].process_async(infer_data)
                self.iters_left_ -= 1
                que.put(0)
                ticks.append(tick)
        self.models_[idx].close_input()
        cunsume_thread.join()
        self.models_[idx].wait_until_done()
        end = time.time()
        self.merge_lock.acquire()
        time_used = (end - start) * 1000000
        self.throughput_ += len(ticks) * self.config_.batch_size * 1000000.0 / time_used
        assert len(ticks) == len(tocks)
        self.latency_begin_ += ticks
        self.latency_end_ += tocks
        self.merge_lock.release()
