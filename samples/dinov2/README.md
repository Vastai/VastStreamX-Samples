# DINOV2 Sample

本目录提供基于 dinov2 模型的 sample


## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/facebookresearch/dinov2)  [modelzoo](http://gitlabdev.vastai.com/VastML/algorithm_modelzoo/-/tree/develop/image_retrieval/dinov2) |
|  输入 shape |   [ (1,3,224,224) ]     |
| INT8量化方式 |   -          |
|  官方精度 |  -    |
|  VACC FP16  精度 | mAP M: 79.6025, H: 58.1765 |
|  VACC INT8  精度 | - |



## 数据准备

下载模型 dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc 到 /opt/vastai/vaststreamx/data/models 里  
下载 vdsp 预处理算子 normalize 和 space_to_depth 到 /opt/vastai/vaststreamx/data/elf 里  
下载数据集 oxbuild_images-v1.tgz 到 /opt/vastai/vaststreamx/data/datasets 里  

## C++ sample
### dinov2 命令行参数说明
```bash
options:
  -m, --model_prefix               model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/dinov2-b-fp16-none-1_3_224_224-vacc/mod])
      --norm_elf_file              normalize elf file path (string [=/opt/vastai/vaststreamx/data/elf/normalize])
      --space_to_depth_elf_file    space to depth elf file path (string [=/opt/vastai/vaststreamx/data/elf/space_to_depth])
      --hw_config                  hw-config file of the model suite (string [=])
  -d, --device_id                  device id to run (unsigned int [=0])
      --input_file                 input file (string [=../data/images/oxford_003681.jpg])
      --dataset_root               input dataset root (string [=])
      --dataset_conf               dataset config file (string [=])
  -?, --help                       print this message
```

### dinov2 运行示例

```bash
#单图片示例
./vaststreamx-samples/bin/dinov2 \
-m /opt/vastai/vaststreamx/data/models/dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc/mod \
--norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
--space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_id 0 \
--input_file ../data/images/oxford_003681.jpg


export PYTHONPATH=../samples/dinov2:$PYTHONPATH
./vaststreamx-samples/bin/dinov2 \
-m /opt/vastai/vaststreamx/data/models/dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc/mod \
--norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
--space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_id 0 \
--dataset_root /opt/vastai/vaststreamx/data/datasets/oxbuild_images-v1 \
--dataset_conf ../data/labels/gnd_roxford5k.pkl
```

### dinov2 运行结果示例

```bash
#单张图片结果示例
output:[-0.0480957,0.734375,-1.06543 ... ]

#精度统计结果
mAP M: 79.6025, H: 58.1765
mP@k[ 1 5 10 ], M: [98.5714 94.5238 91.381 ], H: [92.8571 80.0476 70.0476 ]
```


### dinov2_prof 命令行参数说明
```bash
options:
  -m, --model_prefix               model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/dinov2-b-fp16-none-1_3_224_224-vacc/mod])
      --hw_config                  hw-config file of the model suite (string [=])
      --norm_elf_file              normalize elf file path (string [=/opt/vastai/vaststreamx/data/elf/normalize])
      --space_to_depth_elf_file    space to depth elf file path (string [=/opt/vastai/vaststreamx/data/elf/space_to_depth])
  -d, --device_ids                 device id to run (string [=[0]])
  -b, --batch_size                 profiling batch size of the model (unsigned int [=1])
  -i, --instance                   instance number for each device (unsigned int [=1])
  -s, --shape                      model input shape (string [=])
      --iterations                 iterations count for one profiling (int [=10240])
      --percentiles                percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host                 cache input data into host memory (bool [=0])
  -q, --queue_size                 aync wait queue size (unsigned int [=1])
  -?, --help                       print this message
```

### dinov2_prof 命令行示例
在build目录执行
```bash
#测试最大吞吐
./vaststreamx-samples/bin/dinov2_prof  \
-m /opt/vastai/vaststreamx/data/models/dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc/mod \
--norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
--space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 200 \
--shape [3,224,224] \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/dinov2_prof  \
-m /opt/vastai/vaststreamx/data/models/dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc/mod \
--norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
--space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 200 \
--shape [3,224,224] \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 0

```

### dinov2_prof 命令行结果示例

```bash
#测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 26.2984
  latency (us):
    avg latency: 113247
    min latency: 40916
    max latency: 116936
    p50 latency: 113960
    p90 latency: 114061
    p95 latency: 114097
    p99 latency: 114127


# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 25.1516
  latency (us):
    avg latency: 39758
    min latency: 39655
    max latency: 40811
    p50 latency: 39741
    p90 latency: 39806
    p95 latency: 39845
    p99 latency: 39889
```


## Python Sample 

### dinov2.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --norm_elf_file NORM_ELF_FILE
                        normalize op elf file
  --space_to_depth_elf_file SPACE_TO_DEPTH_ELF_FILE
                        space_to_depth op elf files
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input file
  --dataset_root DATASET_ROOT
                        input dataset root
  --dataset_conf DATASET_CONF
                        dataset conf pkl file
```

### dinov2.py 运行示例

在本目录下运行  
```bash
#单张照片示例
python3 dinov2.py \
-m /opt/vastai/vaststreamx/data/models/dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc/mod \
--norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
--space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_id 0 \
--input_file ../../data/images/oxford_003681.jpg

#数据集示例
python3 dinov2.py \
-m /opt/vastai/vaststreamx/data/models/dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc/mod \
--norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
--space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_id 0 \
--dataset_root /opt/vastai/vaststreamx/data/datasets/oxbuild_images-v1 \
--dataset_conf ../../data/labels/gnd_roxford5k.pkl
```


### dinov2.py 运行结果示例

```bash
#单张图片结果示例
output:[-0.0480957   0.734375   -1.0654297  ...  0.39916992  0.20153809
  0.62890625] 

#精度统计结果
mAP M: 79.6, H: 58.18
mP@k[ 1  5 10] M: [98.57 94.52 91.38], H: [92.86 80.05 70.05]
```


## Python sample 性能测试

### dinov2_prof.py 命令行参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --norm_elf_file NORM_ELF_FILE
                        normalize op elf file
  --space_to_depth_elf_file SPACE_TO_DEPTH_ELF_FILE
                        space_to_depth op elf files
  --hw_config HW_CONFIG
                        hw-config file of the model suite
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


### dinov2_prof.py 运行示例

在本目录下运行  
```bash
# 测试最大吞吐
python3 dinov2_prof.py \
-m /opt/vastai/vaststreamx/data/models/dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc/mod \
--norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
--space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 200 \
--shape [3,224,224] \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 dinov2_prof.py \
-m /opt/vastai/vaststreamx/data/models/dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc/mod \
--norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
--space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 200 \
--shape [3,224,224] \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 0
```

### dinov2_prof.py 运行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 26.44
  latency (us):
    avg latency: 113206
    min latency: 45823
    max latency: 118454
    p50 latency: 113346
    p90 latency: 113459
    p95 latency: 113496
    p99 latency: 113542

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 24.81
  latency (us):
    avg latency: 40300
    min latency: 40049
    max latency: 44778
    p50 latency: 40295
    p90 latency: 40372
    p95 latency: 40404
    p99 latency: 40522
```
