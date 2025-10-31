# EFFICIENT ADS SAMPLE

本目录提供基于 efficient_ads 模型的异常检测 sample.

## 模型信息

| 模型信息     | 值                                                                    |
| -------- | -------------------------------------------------------------------- |
| 来源       | [github](https://github.com/open-edge-platform/anomalib/tree/v1.1.0) |
| 输入 shape | [ (1,3,256,256) ]                                                    |
| INT8量化方式 | 无                                                                    |

## 数据准备

下载模型 official_efficientAD_run_stream_fp16 到 /opt/vastai/vaststreamx/data/models 里
(official_efficientAD_run_stream_fp16 里包含原始的 onnx 模型 和模型三件套)
下载数据集 zipper 到 /opt/vastai/vaststreamx/data/datasets 里

## Python Sample

### efficient_ads.py 命令行参数说明

```bash
options:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file
  --onnx_file ONNX_FILE
                        onnx file
  --dataset_root_dir DATASET_ROOT_DIR
                        dataset root dir, such as zipper
  --split SPLIT         split of the dataset
```

### efficient_ads.py 命令行示例

在本目录下运行

```bash
# 单张图片测试示例
python3 ./efficient_ads.py \
--model_prefix /opt/vastai/vaststreamx/data/models/official_efficientAD_run_stream_fp16/mod \
--vdsp_params ../../data/configs/efficient_ads_rgbplanar.json \
--device_id 0 \
--input_file ../../data/images/broken_teeth.png \
--output_file ./efficient_ads_vsx_output_py.npz

# 数据集测试示例
python3 ./efficient_ads.py \
--model_prefix /opt/vastai/vaststreamx/data/models/official_efficientAD_run_stream_fp16/mod \
--vdsp_params ../../data/configs/efficient_ads_rgbplanar.json \
--device_id 0 \
--onnx_file /opt/vastai/vaststreamx/data/models/official_efficientAD_run_stream_fp16/model.onnx \
--dataset_root_dir /opt/vastai/vaststreamx/data/datasets/zipper \
--split test
```

### efficient_ads.py 命令结果示例

```bash
# 单张图片测试结果示例
保存于 efficient_ads_vsx_output_py.npz 文件中
# 精度统计结果示例
Cosine Similarity: [0.99997294 0.9999667 0.9999812]
```

### efficient_ads_prof.py 命令行参数说明

```bash
options:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  -d DEVICE_IDS, --device_ids DEVICE_IDS
                        device ids to run
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        profiling batch size of the model
  -i INSTANCE, --instance INSTANCE
                        instance number for each device
  -s SHAPE, --shape SHAPE
                        model input shape
  --iterations ITERATIONS
                        iterations count for one profiling
  --queue_size QUEUE_SIZE
                        aync wait queue size
  --percentiles PERCENTILES
                        percentiles of latency
  --input_host INPUT_HOST
                        cache input data into host memory
```

### efficient_ads_prof.py 命令行示例

在本目录运行

```bash
# 测试最大吞吐
python3 ./efficient_ads_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/models/official_efficientAD_run_stream_fp16/mod \
--vdsp_params ../../data/configs/efficient_ads_rgbplanar.json \
--device_ids "[0]" \
--batch_size 1 \
--instance 1 \
--iterations 4096 \
--queue_size 1 \
--percentiles "[50, 90, 95, 99]" \
--input_host 0

# 测试最小时延
python3 ./efficient_ads_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/models/official_efficientAD_run_stream_fp16/mod \
--vdsp_params ../../data/configs/efficient_ads_rgbplanar.json \
--device_ids "[0]" \
--batch_size 1 \
--instance 1 \
--iterations 4096 \
--queue_size 0 \
--percentiles "[50, 90, 95, 99]" \
--input_host 0
```

### face_detection_prof.py 命令结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 88.45
  latency (us):
    avg latency: 33861
    min latency: 13139
    max latency: 38723
    p50 latency: 33873
    p90 latency: 34041
    p95 latency: 34096
    p99 latency: 34232

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 82.16
  latency (us):
    avg latency: 12169
    min latency: 11893
    max latency: 21882
    p50 latency: 12143
    p90 latency: 12287
    p95 latency: 12362
    p99 latency: 12518
```
