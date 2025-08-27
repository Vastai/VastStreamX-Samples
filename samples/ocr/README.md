# OCR SAMPLE

本目录提供基于 CRNN 的 OCR sample


## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/doc/doc_ch/algorithm_rec_crnn.md)  [modelzoo](https://github.com/Vastai/VastModelZOO/tree/main/cv/text_recognition/crnn) |
|  输入 shape |   [ (1,3,32,100) ]     |
| INT8量化方式 |   max          |
|  官方精度 | "ACC": 81.04 |
|  VACC FP16  精度 | "ACC": 75.69 |
|  VACC INT8  精度 | "ACC": 74.3 |




## 数据准备

下载模型 resnet34_vd-int8-max-1_3_32_100-vacc 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 CUTE80 到 /opt/vastai/vaststreamx/data/datasets 里



## C++ sample

### crnn 命令行参数说明
```bash
options:
  -m, --model_prefix           model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod])
      --hw_config              hw-config file of the model suite (string [=])
      --vdsp_params            vdsp preprocess parameter file (string [=../data/configs/crnn_rgbplanar.json])
  -d, --device_id              device id to run (unsigned int [=0])
      --label_file             label file (string [=../data/labels/key_37.txt])
      --input_file             input image (string [=../data/images/word_336.png])
      --dataset_filelist       input dataset filelist (string [=])
      --dataset_root           input dataset root (string [=])
      --dataset_output_file    dataset output file (string [=])
  -?, --help                   print this message
```
### crnn 命令示例
在build 目录下运行

单图片示例
```bash
./vaststreamx-samples/bin/crnn \
-m /opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod \
--vdsp_params ../data/configs/crnn_rgbplanar.json \
--device_id 0 \
--label_file ../data/labels/key_37.txt \
--input_file ../data/images/word_336.png 
```
输出如下结果：
```bash
score: 0.999219
text: super
```


数据集示例

```bash
./vaststreamx-samples/bin/crnn \
-m /opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod \
--vdsp_params ../data/configs/crnn_rgbplanar.json \
--device_id 0 \
--label_file ../data/labels/key_37.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/CUTE80/CUTE80_img_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/CUTE80 \
--dataset_output_file cute80_pred.txt

# 结果保存在 cute80_pred.txt

# 统计精度
python3 ../evaluation/crnn/crnn_eval.py \
--gt_file /opt/vastai/vaststreamx/data/datasets/CUTE80/CUTE80_gt.txt \
--output_file cute80_pred.txt


```
精度结果示例
```bash
right_num = 214 all_num=288, acc = 0.7430555555555556
```
### crnn_prof 命令行参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/resnet34_vd])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/crnn_rgbplanar.json])
  -d, --device_ids      device id to run (string [=[0]])
      --label_file      label file (string [=../data/labels/key_37.txt])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        instance number for each device (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (int [=10240])
      --percentiles     percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=1])
  -?, --help            print this message
```
### crnn_prof 命令行示例

```bash
# 测试最大吞吐
./vaststreamx-samples/bin/crnn_prof \
-m /opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod \
--vdsp_params ../data/configs/crnn_rgbplanar.json \
--device_ids [0] \
--label_file ../data/labels/key_37.txt \
--batch_size 8 \
--instance 1 \
--shape "[3,32,100]" \
--iterations 200 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/crnn_prof \
-m /opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod \
--vdsp_params ../data/configs/crnn_rgbplanar.json \
--device_ids [0] \
--label_file ../data/labels/key_37.txt \
--batch_size 1 \
--instance 1 \
--shape "[3,32,100]" \
--iterations 500 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### crnn_prof 结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 8
  throughput (qps): 277.303
  latency (us):
    avg latency: 86061
    min latency: 31820
    max latency: 89070
    p50 latency: 86454
    p90 latency: 86503
    p95 latency: 86512
    p99 latency: 86541


# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 129.401
  latency (us):
    avg latency: 7727
    min latency: 7660
    max latency: 8206
    p50 latency: 7727
    p90 latency: 7743
    p95 latency: 7751
    p99 latency: 7794
```

##  Python sample 


### crnn.py 命令行参数说明
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
  --label_file LABEL_FILE
                        label file
  --input_file INPUT_FILE
                        input file
  --dataset_filelist DATASET_FILELIST
                        dataset filelist
  --dataset_root DATASET_ROOT
                        input dataset root
  --dataset_output_file DATASET_OUTPUT_FILE
                        dataset output file
```

### crnn.py 运行示例

在本目录下运行  
```bash
#单张图片示例
python3 crnn.py \
-m /opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod \
--vdsp_params ../../data/configs/crnn_rgbplanar.json \
--device_id 0 \
--label_file ../../data/labels/key_37.txt \
--input_file ../../data/images/word_336.png 

#数据集示例
python3 crnn.py \
-m /opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod \
--vdsp_params ../../data/configs/crnn_rgbplanar.json \
--device_id 0 \
--label_file ../../data/labels/key_37.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/CUTE80/CUTE80_img_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/CUTE80 \
--dataset_output_file cute80_pred.txt

# 统计精度
python3 ../../evaluation/crnn/crnn_eval.py \
--gt_file /opt/vastai/vaststreamx/data/datasets/CUTE80/CUTE80_gt.txt \
--output_file cute80_pred.txt
```

### crnn.py 运行结果示例

```bash
#单张图片结果示例
[('super', 0.9990234375)]

#统计精度结果示例
right_num = 214 all_num=288, acc = 0.7430555555555556

```

### crnn_prof.py 命令行参数说明

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
  --label_file LABEL_FILE
                        label file
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

### crnn_prof.py 运行示例

在本目录下运行  
```bash

# 测试最大吞吐
python3 crnn_prof.py \
-m /opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod \
--vdsp_params ../../data/configs/crnn_rgbplanar.json \
--device_ids [0] \
--batch_size 8 \
--instance 1 \
--label_file ../../data/labels/key_37.txt \
--shape "[3,32,100]" \
--iterations 200 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 crnn_prof.py \
-m /opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod \
--vdsp_params ../../data/configs/crnn_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--label_file ../../data/labels/key_37.txt \
--shape "[3,32,100]" \
--iterations 500 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0

```

### crnn_prof.py 结果示例

```bash

# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 8
  throughput (qps): 276.69
  latency (us):
    avg latency: 86054
    min latency: 33150
    max latency: 88498
    p50 latency: 86413
    p90 latency: 87051
    p95 latency: 87417
    p99 latency: 88124


# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 124.28
  latency (us):
    avg latency: 8044
    min latency: 7785
    max latency: 9392
    p50 latency: 7988
    p90 latency: 8311
    p95 latency: 8431
    p99 latency: 8683
```    