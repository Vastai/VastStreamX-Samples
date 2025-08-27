# Salient Object Detection

本目录提供基于 u2net 模型的 salient object detection sample。

## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/xuebinqin/U-2-Net/tree/master)  [modelzoo](-) |
|  输入 shape |   [ (1,3,320,320) ]     |
| INT8量化方式 |   -          |
|  官方精度 |  "mae":0.033, "avgfmeasure": 0.922, "sm": 0.928    |
|  VACC FP16  精度 |  "mae":0.032, "avgfmeasure": 0.925, "sm": 0.928  |
|  VACC INT8  精度 |  "mae":0.035, "avgfmeasure": 0.922, "sm": 0.925   |


## 数据准备

下载模型 u2net-int8-percentile-1_3_320_320-vacc 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 ECSSD 到 /opt/vastai/vaststreamx/data/datasets 里

## C++ Sample 

### u2net 命令行参数
```bash
options:
  -m, --model_prefix           model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod])
      --hw_config              hw-config file of the model suite (string [=])
      --vdsp_params            vdsp preprocess parameter file (string [=../data/configs/u2net_bgr888.json])
  -d, --device_id              device id to run (unsigned int [=0])
      --input_file             input file (string [=../data/images/cat.jpg])
      --output_file            output file (string [=./u2net_result.png])
      --dataset_filelist       input dataset filelist (string [=])
      --dataset_root           input dataset root (string [=])
      --dataset_output_file    dataset output file (string [=])
  -?, --help                   print this message
```

### u2net 命令行示例
在build 目录里执行
```bash
#测试单张图片
./vaststreamx-samples/bin/u2net \
-m /opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod \
--vdsp_params ../data/configs/u2net_bgr888.json \
--device_id 0 \
--input_file ../data/images/cat.jpg \
--output_file ./u2net_result.png
# 灰度图将被保存为 u2net_result.png。

#测试数据集
mkdir -p u2net_output
./vaststreamx-samples/bin/u2net \
-m /opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod \
--vdsp_params ../data/configs/u2net_bgr888.json \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ECSSD/filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ECSSD/image/ \
--dataset_output_folder ./u2net_output
# 灰度图将被保存在 ./u2net_output 文件夹。
```
### u2net 运行结果示例

测试数据集精度
```bash
python3 ../evaluation/salient_object_detection/PySODEval/eval.py \
--dataset-json ../evaluation/salient_object_detection/PySODEval/examples/config_dataset.json \
--method-json ../evaluation/salient_object_detection/PySODEval/examples/config_method_u2net.json
```

输出精度
```bash
All methods have been evaluated:
Dataset: ECSSD
| methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
|-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
| Method1   | 0.035 |         0.939 |         0.922 |         0.927 |          0.984 |          0.943 |           1 |       0.905 |   0.957 |   0.944 |   0.952 | 0.925 | 0.904 |
```


### u2net_prof 命令行参数
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/u2net_bgr888.json])
  -d, --device_ids      device id to run (string [=[0]])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        instance number for each device (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (unsigned int [=10240])
      --percentiles     percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=1])
  -?, --help            print this message
```

### u2net_prof 运行示例

在 build 目录里   
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/u2net_prof \
-m /opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod \
--vdsp_params ../data/configs/u2net_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 1024 \
--shape "[3,320,320]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/u2net_prof \
-m /opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod \
--vdsp_params ../data/configs/u2net_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 1024 \
--shape "[3,320,320]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### u2net_prof 运行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 92.5299
  latency (us):
    avg latency: 32108
    min latency: 13631
    max latency: 36158
    p50 latency: 32128
    p90 latency: 32574
    p95 latency: 32799
    p99 latency: 33638
    
# 测试最小延迟
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 70.8418
  latency (us):
    avg latency: 14114
    min latency: 12599
    max latency: 15555
    p50 latency: 14108
    p90 latency: 14493
    p95 latency: 14576
    p99 latency: 14785
```

## Python Sample 

### u2net.py 命令行参数说明
```bash
optional arguments:
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
  --dataset_filelist DATASET_FILELIST
                        input dataset filelist
  --dataset_root DATASET_ROOT
                        input dataset root
  --dataset_output_folder DATASET_OUTPUT_FOLDER
                        dataset output folder

```

### u2net.py 运行示例

在本目录下运行  
```bash
#测试单张图片
python3 u2net.py \
-m /opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod \
--vdsp_params ../../../data/configs/u2net_bgr888.json \
--device_id 0 \
--input_file ../../../data/images/cat.jpg \
--output_file ./u2net_result.png
# 灰度图将被保存为 u2net_result.png。

#测试数据集
mkdir -p u2net_output
python3 u2net.py \
-m /opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod \
--vdsp_params ../../../data/configs/u2net_bgr888.json \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ECSSD/filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ECSSD/image/ \
--dataset_output_folder ./u2net_output
# 灰度图将被保存在 ./u2net_output 文件夹。
```

### u2net.py 运行结果示例
在本目录下运行
```bash
python3 ../../../evaluation/salient_object_detection/PySODEval/eval.py \
--dataset-json ../../../evaluation/salient_object_detection/PySODEval/examples/config_dataset.json \
--method-json ../../../evaluation/salient_object_detection/PySODEval/examples/config_method_u2net.json
```

数据集精度为
```bash
All methods have been evaluated:
Dataset: ECSSD
| methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
|-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
| Method1   | 0.035 |         0.939 |         0.919 |         0.922 |          0.983 |          0.933 |           1 |       0.918 |   0.958 |   0.945 |   0.953 | 0.926 | 0.902 |
```


### u2net_prof.py 命令行参数说明

```bash
optional arguments:
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


### u2net_prof.py 运行示例

在本目录下运行  
```bash
# 测试最大吞吐
python3 u2net_prof.py \
-m /opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod \
--vdsp_params ../../../data/configs/u2net_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 1000 \
--shape [3,320,320] \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 1

# 测试最佳延迟
python3 u2net_prof.py \
-m /opt/vastai/vaststreamx/data/models/u2net-int8-percentile-1_3_320_320-vacc/mod \
--vdsp_params ../../../data/configs/u2net_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 1000 \
--shape [3,320,320] \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 0
```

### u2net_prof.py 运行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 92.51030259132953
  latency (us):
    avg latency: 32208
    min latency: 14123
    max latency: 33993
    p50 latency: 32228
    p90 latency: 32541
    p95 latency: 32711
    p99 latency: 32963
    
# 测试最佳延迟
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 80.67
  latency (us):
    avg latency: 12392
    min latency: 11956
    max latency: 14620
    p50 latency: 12366
    p90 latency: 12648
    p95 latency: 12744
    p99 latency: 12907
```
