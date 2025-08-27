# VIT Sample

本目录提供基于 vit base 16 模型的 Classfication sample


## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | huggingface  [modelzoo](https://github.com/Vastai/VastModelZOO/tree/main/cv/classification/vision_transformer) |
|  输入 shape |   [ (1,3,224,224) ]     |
| INT8量化方式 |   -          |
|  官方精度 |  -      |
|  VACC FP16  精度 | top1_rate: 80.184 top5_rate: 95.398 |
|  VACC INT8  精度 | -  |


## 数据准备

下载模型 vit-b-fp16-none-1_3_224_224-vacc 到 /opt/vastai/vaststreamx/data/models 里  
下载 vdsp 预处理算子 normalize 和 space_to_depth 到 /opt/vastai/vaststreamx/data/elf 里  
下载数据集 ILSVRC2012_img_val 到 /opt/vastai/vaststreamx/data/datasets 里  

## C++ sample

### vit 命令行参数
```bash
options:
  -m, --model_prefix           model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/vit-b-fp16-none-1_3_224_224-vacc/mod])
      --norm_elf_file          normalize vdsp op elf file (string [=/opt/vastai/vaststreamx/data/elf/normalize])
      --space_to_depth_elf_file space_to_depth vdsp op elf file (string [=/opt/vastai/vaststreamx/data/elf/space_to_depth])
      --hw_config              hw-config file of the model suite (string [=])
  -d, --device_id              device id to run (unsigned int [=0])
      --label_file             label file (string [=../data/labels/imagenet.txt])
      --input_file             input file (string [=../data/images/cat.jpg])
      --dataset_filelist       input dataset filelist (string [=])
      --dataset_root           input dataset root (string [=])
      --dataset_output_file    dataset output file (string [=])
  -?, --help                   print this message
```

### vit 运行示例

在 build 目录里执行   
```bash
#单张图片示例
./vaststreamx-samples/bin/vit \
-m /opt/vastai/vaststreamx/data/models/vit-b-fp16-none-1_3_224_224-vacc/mod \
--norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
--space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_id 0 \
--label_file ../data/labels/imagenet.txt \
--input_file ../data/images/cat.jpg 


#数据集示例
./vaststreamx-samples/bin/vit \
-m /opt/vastai/vaststreamx/data/models/vit-b-fp16-none-1_3_224_224-vacc/mod \
--norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
--space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_id 0 \
--label_file ../data/labels/imagenet.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ILSVRC2012_img_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file vit_result.txt

#统计精度
python3 ../evaluation/classification/eval_topk.py  vit_result.txt  
```

### vit 运行结果示例

```bash
#单张图片结果示例
Top5:
0th, score: 8.906, class name: Egyptian cat
1th, score: 8.125, class name: tabby, tabby cat
2th, score: 7.809, class name: lynx, catamount
3th, score: 7.613, class name: tiger cat
4th, score: 5.078, class name: kit fox, Vulpes macrotis

#统计精度结果
[VACC]:  top1_rate: 80.184 top5_rate: 95.398
```


### vit_prof 命令行参数
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/vit-b-fp16-none-1_3_224_224-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/resnet_bgr888.json])
  -d, --device_ids      device id to run (string [=[0]])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        instance number for each device (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (int [=10240])
      --percentiles     percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=1])
  -?, --help            print this message
```

### vit_prof 运行示例

在 build 目录里   
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/vit_prof \
-m /opt/vastai/vaststreamx/data/models/vit-b-fp16-none-1_3_224_224-vacc/mod \
--norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
--space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 700 \
--shape "[3,224,224]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/vit_prof \
-m /opt/vastai/vaststreamx/data/models/vit-b-fp16-none-1_3_224_224-vacc/mod \
--norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
--space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 500 \
--shape "[3,224,224]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### vit_prof 运行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 132.713
  latency (us):
    avg latency: 22391
    min latency: 10601
    max latency: 25366
    p50 latency: 22407
    p90 latency: 22527
    p95 latency: 22558
    p99 latency: 22655

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 109.075
  latency (us):
    avg latency: 9167
    min latency: 9108
    max latency: 9960
    p50 latency: 9163
    p90 latency: 9173
    p95 latency: 9180
    p99 latency: 9250
```

## Python Sample 

### vit.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --norm_elf_file NORMALIZE_ELF_FILE
                        normalize vdsp op elf file
  --space_to_depth_elf_file SPACE_TO_DEPTH_ELF_FILE
                        space_to_depth vdsp op elf file
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --label_file LABEL_FILE
                        label file
  --input_file INPUT_FILE
                        input file
  --dataset_filelist DATASET_FILELIST
                        input dataset filelist
  --dataset_root DATASET_ROOT
                        input dataset root
  --dataset_output_file DATASET_OUTPUT_FILE
                        dataset output file

```

### vit.py 运行示例

在本目录下运行  
```bash
#单张照片示例
python3 vit.py \
-m /opt/vastai/vaststreamx/data/models/vit-b-fp16-none-1_3_224_224-vacc/mod \
--norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
--space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_id 0 \
--label_file ../../../data/labels/imagenet.txt \
--input_file ../../../data/images/cat.jpg

#数据集示例
python3 vit.py \
-m /opt/vastai/vaststreamx/data/models/vit-b-fp16-none-1_3_224_224-vacc/mod \
--norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
--space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_id 0 \
--label_file ../../../data/labels/imagenet.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ILSVRC2012_img_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file vit_result.txt

#统计精度
python3 ../../../evaluation/classification/eval_topk.py  vit_result.txt  
```

### vit.py 运行结果示例

```bash
#单张图片结果示例
Top5:
0th: score: 8.9062, class name: Egyptian cat
1th: score: 8.1250, class name: tabby, tabby cat
2th: score: 7.8086, class name: lynx, catamount
3th: score: 7.6133, class name: tiger cat
4th: score: 5.0781, class name: kit fox, Vulpes macrotis

#精度统计结果
[VACC]:  top1_rate: 80.2 top5_rate: 95.402
```


## Python sample 性能测试

### vit_prof.py 命令行参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --norm_elf_file NORMALIZE_ELF_FILE
                        normalize vdsp op elf file
  --space_to_depth_elf_file SPACE_TO_DEPTH_ELF_FILE
                        space_to_depth vdsp op elf file
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


### vit_prof.py 运行示例

在本目录下运行  
```bash
# 测试最大吞吐
python3 vit_prof.py \
-m /opt/vastai/vaststreamx/data/models/vit-b-fp16-none-1_3_224_224-vacc/mod \
--norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
--space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 700 \
--shape [3,224,224] \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 vit_prof.py \
-m /opt/vastai/vaststreamx/data/models/vit-b-fp16-none-1_3_224_224-vacc/mod \
--norm_elf_file /opt/vastai/vaststreamx/data/elf/normalize \
--space_to_depth_elf_file /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 500 \
--shape [3,224,224] \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 0
```



### vit_prof.py 运行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 132.63
  latency (us):
    avg latency: 22430
    min latency: 11179
    max latency: 23699
    p50 latency: 22445
    p90 latency: 22573
    p95 latency: 22619
    p99 latency: 22728

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 107.18
  latency (us):
    avg latency: 9329
    min latency: 9244
    max latency: 10133
    p50 latency: 9317
    p90 latency: 9369
    p95 latency: 9374
    p99 latency: 9399
```
