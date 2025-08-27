# Face alignment sample 

本目录提供基于 hih 模型的 人脸对齐 sample。  
该模型输出98个人脸关键点的热力图，经过后处理，输出98个关键点的二维 (x, y) 坐标。

## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/starhiking/HeatmapInHeatmap)  [modelzoo](-) |
|  输入 shape |   [ (1,3,256,256) ]     |
| INT8量化方式 |   percentile          |
|  官方精度 |  "NME":  0.0408    |
|  VACC FP16  精度 |  "NME":  0.0423   |
|  VACC INT8  精度 |  "NME":  0.0424  |

## 数据准备

下载模型 hih_2s-int8-percentile-1_3_256_256-vacc 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 WFLW 到 /opt/vastai/vaststreamx/data/datasets 里


## C++ sample

### hih 命令行参数说明
```bash
options:
  -m, --model_prefix           model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod])
      --hw_config              hw-config file of the model suite (string [=])
      --vdsp_params            vdsp preprocess parameter file (string [=../data/configs/hih_bgr888.json])
  -d, --device_id              device id to run (unsigned int [=0])
      --input_file             input file (string [=../data/images/face.jpg])
      --dataset_filelist       input dataset filelist (string [=])
      --dataset_root           input dataset root (string [=])
      --dataset_output_file    dataset output file (string [=])
  -?, --help                   print this message
```

### hih 运行示例
在 build 目录里执行  
```bash
#跑单张图片
./vaststreamx-samples/bin/hih \
-m /opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod \
--vdsp_params ../data/configs/hih_bgr888.json \
-d 0 \
--input_file ../data/images/face.jpg

#跑数据集
./vaststreamx-samples/bin/hih \
-m /opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod \
--vdsp_params ../data/configs/hih_bgr888.json \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/WFLW/test_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/WFLW/test/ \
--dataset_output_file ./results.txt
```
数据集文件名和98个点的坐标将保存在 results.txt 文件中。

### hih 运行结果示例
```bash
#跑单张图片, 结果将打印98个特征点二维坐标
Face alignment results: 
(0.0859375, 0.431641)
(0.0859375, 0.478516)
(0.101562, 0.509766)
(0.101562, 0.541016)
(0.101562, 0.587891)
... 
```
测试数据集精度
```bash
python3 ../evaluation/face_alignment/eval_wflw.py \
--result ./results.txt \
--gt /opt/vastai/vaststreamx/data/datasets/WFLW/test.txt
```
```bash
#输出精度：
Face alignment evaluation result:
NME %: 0.04248637550034785
FR_0.1% : 0.032399999999999984
AUC_0.1: 0.597618
``` 


### hih_prof 命令行参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/hih_bgr888.json])
  -d, --device_ids      device id to run (string [=[0]])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        instance number or range for each device (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (int [=10240])
      --percentiles     percentiles of latency (string [=[50,90,95,99]])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=1])
  -?, --help            print this message
```

### hih_prof 运行示例
在 build 目录里执行  
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/hih_prof \
-m /opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod \
--vdsp_params ../data/configs/hih_bgr888.json \
--device_ids [0] \
--batch_size 10 \
--instance 1 \
--iterations 500 \
--shape "[3,256,256]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 2

# 测试最小时延
./vaststreamx-samples/bin/hih_prof \
-m /opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod \
--vdsp_params ../data/configs/hih_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 1024 \
--shape "[3,256,256]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### hih_prof 运行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 2
  batch size: 10
  throughput (qps): 843.012
  latency (us):
    avg latency: 47240
    min latency: 46458
    max latency: 82559
    p50 latency: 46978
    p90 latency: 47697
    p95 latency: 48170
    p99 latency: 48489

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 202.174
  latency (us):
    avg latency: 4945
    min latency: 4870
    max latency: 6419
    p50 latency: 4943
    p90 latency: 4955
    p95 latency: 4959
    p99 latency: 4982
```

## Python sample

### hih.py 命令行参数说明
```bash
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
  --dataset_filelist DATASET_FILELIST
                        dataset file list
  --dataset_root DATASET_ROOT
                        dataset root
  --dataset_output_file DATASET_OUTPUT_FILE
                        dataset output file
```

### hih.py 运行示例

在本目录下运行  
```bash
#单张图片示例
python3 hih.py \
--model_prefix /opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod \
--vdsp_params ../../../data/configs/hih_bgr888.json \
--device_id 0 \
--input_file ../../../data/images/face.jpg

#数据集示例
python3 hih.py \
--model_prefix /opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod \
--vdsp_params ../../../data/configs/hih_bgr888.json \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/WFLW/test_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/WFLW/test/ \
--dataset_output_file ./results.txt
```
结果将保存在文件 results.txt 内

### hih 运行结果示例
```bash
#跑单张图片, 结果将打印98个特征点二维坐标
Face alignment results:
[[0.0859375  0.43164062]
 [0.0859375  0.47851562]
 [0.1015625  0.5097656 ]
 [0.1015625  0.5410156 ]
 [0.1015625  0.5878906 ]
... 
```
测试数据集精度
```bash
python3 ../../../evaluation/face_alignment/eval_wflw.py \
--result ./results.txt \
--gt /opt/vastai/vaststreamx/data/datasets/WFLW/test.txt
```
```bash
#输出精度
Face alignment evaluation result:
NME %: 0.042438748140321024
FR_0.1% : 0.032399999999999984
AUC_0.1: 0.5976196
```

### hih_prof.py 命令行参数说明

```bash
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


### hih_prof.py 运行示例

在本目录下运行  
```bash
# 测试最大吞吐
python3 hih_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod \
--vdsp_params ../../../data/configs/hih_bgr888.json \
--device_ids [0] \
--batch_size 10 \
--instance 1 \
--iterations 500 \
--shape "[3,256,256]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 4

# 测试最小时延
python3 hih_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/models/hih_2s-int8-percentile-1_3_256_256-vacc/mod \
--vdsp_params ../../../data/configs/hih_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 1024 \
--shape "[3,256,256]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### hih_prof.py 运行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 4
  batch size: 10
  throughput (qps): 798.31
  latency (us):
    avg latency: 74190
    min latency: 45514
    max latency: 509402
    p50 latency: 69966
    p90 latency: 71445
    p95 latency: 72218
    p99 latency: 97979

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 238.59
  latency (us):
    avg latency: 4189
    min latency: 4108
    max latency: 8763
    p50 latency: 4181
    p90 latency: 4195
    p95 latency: 4206
    p99 latency: 4276
```
