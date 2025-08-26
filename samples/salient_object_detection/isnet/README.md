# Salient Object Detection

本目录提供基于 isnet 模型的 salient object detection sample。

## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/xuebinqin/DIS)  [modelzoo](http://gitlabdev.vastai.com/VastML/algorithm_modelzoo/-/tree/develop/salient_object_detection/isnet) |
|  输入 shape |   [ (1,3,320,320) ]     |
| INT8量化方式 |   -          |
|  官方精度 |  "mae":0.114, "avgfmeasure": 0.672, "sm": 0.789    |
|  VACC FP16  精度 |  "mae":0.115, "avgfmeasure": 0.672, "sm": 0.789  |
|  VACC INT8  精度 |  "mae":0.117, "avgfmeasure": 0.668, "sm": 0.787   |


## 数据准备

下载模型 isnet-int8-kl_divergence-1_3_320_320-vacc 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 ECSSD 到 /opt/vastai/vaststreamx/data/datasets 里
## C++ Sample 

### isnet 命令行参数
```bash
options:
  -m, --model_prefix           model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/isnet-int8-kl_divergence-1_3_320_320-vacc/mod])
      --hw_config              hw-config file of the model suite (string [=])
      --vdsp_params            vdsp preprocess parameter file (string [=../data/configs/isnet_bgr888.json])
  -d, --device_id              device id to run (unsigned int [=0])
      --input_file             input file (string [=../data/images/cat.jpg])
      --output_file            output file (string [=./isnet_result.png])
      --dataset_filelist       input dataset filelist (string [=])
      --dataset_root           input dataset root (string [=])
      --dataset_output_folder  dataset output folder (string [=])
  -?, --help                   print this message
```

### isnet 命令行示例
在build 目录里执行
```bash
#测试单张图片
./vaststreamx-samples/bin/isnet \
-m /opt/vastai/vaststreamx/data/models/isnet-int8-kl_divergence-1_3_320_320-vacc/mod \
--vdsp_params ../data/configs/isnet_bgr888.json \
--device_id 0 \
--input_file ../data/images/cat.jpg \
--output_file ./isnet_result.png
# 灰度图将被保存为 isnet_result.png。

#测试数据集
mkdir -p isnet_output
./vaststreamx-samples/bin/isnet \
-m /opt/vastai/vaststreamx/data/models/isnet-int8-kl_divergence-1_3_320_320-vacc/mod \
--vdsp_params ../data/configs/isnet_bgr888.json \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ECSSD/filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ECSSD/image/ \
--dataset_output_folder ./isnet_output
# 灰度图将被保存在文件夹 isnet_output 内。
```
### isnet 运行结果示例

测试数据集精度
```bash
python3 ../evaluation/salient_object_detection/PySODEval/eval.py \
--dataset-json ../evaluation/salient_object_detection/PySODEval/examples/config_dataset.json \
--method-json ../evaluation/salient_object_detection/PySODEval/examples/config_method_isnet.json
```

输出精度
```bash
All methods have been evaluated:
Dataset: ECSSD
| methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
|-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
| Method1   | 0.116 |         0.755 |         0.671 |         0.737 |          0.847 |           0.78 |           1 |         0.7 |   0.845 |   0.751 |   0.835 | 0.789 | 0.647 |
```


### isnet_prof 命令行参数
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/isnet-int8-kl_divergence-1_3_320_320-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../../../data/configs/isnet_bgr888.json])
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

### isnet_prof 运行示例

在 build 目录里   
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/isnet_prof \
-m /opt/vastai/vaststreamx/data/models/isnet-int8-kl_divergence-1_3_320_320-vacc/mod \
--vdsp_params ../data/configs/isnet_bgr888.json \
--device_ids [0] \
--batch_size 2 \
--instance 1 \
--iterations 1024 \
--shape "[3,320,320]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/isnet_prof \
-m /opt/vastai/vaststreamx/data/models/isnet-int8-kl_divergence-1_3_320_320-vacc/mod \
--vdsp_params ../data/configs/isnet_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 1024 \
--shape "[3,320,320]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### isnet_prof 运行结果示例

```bash
#测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 2
  throughput (qps): 199.603
  latency (us):
    avg latency: 29771
    min latency: 11983
    max latency: 31934
    p50 latency: 29787
    p90 latency: 29966
    p95 latency: 30029
    p99 latency: 30233

#测试最小延迟
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 153.185
  latency (us):
    avg latency: 6525
    min latency: 5986
    max latency: 7474
    p50 latency: 6571
    p90 latency: 6732
    p95 latency: 6807
    p99 latency: 7011
```

## Python Sample 

### isnet.py 命令行参数说明
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

### isnet.py 运行示例

在本目录下运行  
```bash
#测试单张图片
python3 isnet.py \
-m /opt/vastai/vaststreamx/data/models/isnet-int8-kl_divergence-1_3_320_320-vacc/mod \
--vdsp_params ../../../data/configs/isnet_bgr888.json \
--device_id 0 \
--input_file ../../../data/images/cat.jpg \
--output_file ./isnet_result.png
# 灰度图将被保存为 isnet_result.png。

#测试数据集
mkdir -p isnet_output
python3 isnet.py \
-m /opt/vastai/vaststreamx/data/models/isnet-int8-kl_divergence-1_3_320_320-vacc/mod \
--vdsp_params ../../../data/configs/isnet_bgr888.json \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ECSSD/filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ECSSD/image/ \
--dataset_output_folder ./isnet_output
# 灰度图将被保存在 ./isnet_output 文件夹。
```

### isnet.py 运行结果示例
在本目录下运行
```bash
python3 ../../../evaluation/salient_object_detection/PySODEval/eval.py \
--dataset-json ../../../evaluation/salient_object_detection/PySODEval/examples/config_dataset.json \
--method-json ../../../evaluation/salient_object_detection/PySODEval/examples/config_method_isnet.json
```

数据集精度为
```bash
All methods have been evaluated:
Dataset: ECSSD
| methods   |   mae |   maxfmeasure |   avgfmeasure |   adpfmeasure |   maxprecision |   avgprecision |   maxrecall |   avgrecall |   maxem |   avgem |   adpem |    sm |   wfm |
|-----------|-------|---------------|---------------|---------------|----------------|----------------|-------------|-------------|---------|---------|---------|-------|-------|
| Method1   | 0.116 |         0.755 |         0.671 |         0.737 |          0.847 |           0.78 |           1 |         0.7 |   0.845 |   0.751 |   0.835 | 0.789 | 0.647 |
```


### isnet_prof.py 命令行参数说明

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


### isnet_prof.py 运行示例

在本目录下运行  
```bash
# 测试最大吞吐
python3 isnet_prof.py \
-m /opt/vastai/vaststreamx/data/models/isnet-int8-kl_divergence-1_3_320_320-vacc/mod \
--vdsp_params ../../../data/configs/isnet_bgr888.json \
--device_ids [0] \
--batch_size 2 \
--instance 1 \
--iterations 1000 \
--shape [3,320,320] \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 1

#测试最佳延迟
python3 isnet_prof.py \
-m /opt/vastai/vaststreamx/data/models/isnet-int8-kl_divergence-1_3_320_320-vacc/mod \
--vdsp_params ../../../data/configs/isnet_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 1000 \
--shape [3,320,320] \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 0
```

### isnet_prof.py 运行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 2
  throughput (qps): 199.4827701646825
  latency (us):
    avg latency: 29843
    min latency: 12661
    max latency: 30820
    p50 latency: 29868
    p90 latency: 30072
    p95 latency: 30170
    p99 latency: 30372

#测试最佳延迟
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 144.97209178425555
  latency (us):
    avg latency: 6893
    min latency: 6236
    max latency: 7987
    p50 latency: 6908
    p90 latency: 7042
    p95 latency: 7106
    p99 latency: 7386
```
