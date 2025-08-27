# Classification

本目录提供基于 resnet50 模型的 Classfication  sample

## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/resnet.py)  [modelzoo](https://github.com/Vastai/VastModelZOO/tree/main/cv/classification/resnet) |
|  输入 shape |   [ (1,3,224,224) ]     |
| INT8量化方式 |   -          |
|  官方精度 |  top1_rate:76.130 ; top5_rate: 92.862       |
|  VACC FP16  精度 | top1_rate: 75.936 ; top5_rate: 92.85 |
|  VACC INT8  精度 | top1_rate: 75.816 ; top5_rate: 92.796  |

## 数据准备

下载模型 resnet50-int8-percentile-1_3_224_224-vacc 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 ILSVRC2012_img_val 到 /opt/vastai/vaststreamx/data/datasets 里

## C++ sample

### classification 命令行参数
```bash
options:
  -m, --model_prefix           model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod])
      --hw_config              hw-config file of the model suite (string [=])
      --vdsp_params            vdsp preprocess parameter file (string [=../data/configs/resnet_bgr888.json])
  -d, --device_id              device id to run (unsigned int [=0])
      --label_file             label file (string [=../data/labels/imagenet.txt])
      --input_file             input file (string [=../data/images/cat.jpg])
      --dataset_filelist       input dataset filelist (string [=])
      --dataset_root           input dataset root (string [=])
      --dataset_output_file    dataset output file (string [=])
  -?, --help                   print this message
```

### classification 运行示例

在 build 目录里执行   
```bash
#单张图片示例
./vaststreamx-samples/bin/classification \
-m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
--vdsp_params ../data/configs/resnet_rgbplanar.json \
--device_id 0 \
--label_file ../data/labels/imagenet.txt \
--input_file ../data/images/cat.jpg 


#数据集示例
./vaststreamx-samples/bin/classification \
-m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
--vdsp_params ../data/configs/resnet_rgbplanar.json \
--device_id 0 \
--label_file ../data/labels/imagenet.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ILSVRC2012_img_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file cls_result.txt

#统计精度
python3 ../evaluation/classification/eval_topk.py  cls_result.txt  
```

### classification 运行结果示例

```bash
#单张图片结果示例
Top5:
0th, score: 0.3083, class name: lynx, catamount
1th, score: 0.2401, class name: tabby, tabby cat
2th, score: 0.212, class name: tiger cat
3th, score: 0.1207, class name: Egyptian cat
4th, score: 0.01234, class name: Persian cat


#统计精度结果
[VACC]:  top1_rate: 75.816 top5_rate: 92.796
```


### cls_prof 命令行参数
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod])
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

### cls_prof 运行示例

在 build 目录里   
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/cls_prof \
-m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
--vdsp_params ../data/configs/resnet_rgbplanar.json \
--device_ids [0] \
--batch_size 8 \
--instance 1 \
--iterations 2000 \
--shape "[3,256,256]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/cls_prof \
-m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
--vdsp_params ../data/configs/resnet_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 4000 \
--shape "[3,256,256]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### cls_prof 运行结果示例
以下结果是 880MHz 测试结果
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 8
  throughput (qps): 3231.33
  latency (us):
    avg latency: 7364
    min latency: 4861
    max latency: 9806
    p50 latency: 7362
    p90 latency: 7404
    p95 latency: 7409
    p99 latency: 7418

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 941.4
  latency (us):
    avg latency: 1061
    min latency: 1002
    max latency: 1280
    p50 latency: 1061
    p90 latency: 1065
    p95 latency: 1067
    p99 latency: 1072
```

## Python Sample 

### classification.py 命令行参数说明
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
                        input dataset filelist
  --dataset_root DATASET_ROOT
                        input dataset root
  --dataset_output_file DATASET_OUTPUT_FILE
                        dataset output file

```

### classification.py 运行示例

在本目录下运行  
```bash
#单张照片示例
python3 classification.py \
-m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
--vdsp_params ../../../data/configs/resnet_rgbplanar.json \
--device_id 0 \
--label_file ../../../data/labels/imagenet.txt \
--input_file ../../../data/images/cat.jpg

#数据集示例
python3 classification.py \
-m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
--vdsp_params ../../../data/configs/resnet_rgbplanar.json \
--device_id 0 \
--label_file ../../../data/labels/imagenet.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ILSVRC2012_img_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file cls_result.txt

#统计精度
python3 ../../../evaluation/classification/eval_topk.py  cls_result.txt  
```



### classification.py 运行结果示例

```bash
#单张图片结果示例
Top5:
0th, score: 0.3083, class name: lynx, catamount
1th, score: 0.2401, class name: tabby, tabby cat
2th, score: 0.2120, class name: tiger cat
3th, score: 0.1207, class name: Egyptian cat
4th, score: 0.0123, class name: Persian cat


#精度统计结果
[VACC]:  top1_rate: 75.806 top5_rate: 92.806
```


## Python sample 性能测试

### cls_prof.py 命令行参数说明

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


### cls_prof.py 运行示例

在本目录下运行  
```bash
# 测试最大吞吐
python3 cls_prof.py \
-m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
--vdsp_params ../../../data/configs/resnet_rgbplanar.json \
--device_ids [0] \
--batch_size 8 \
--instance 1 \
--iterations 4000 \
--shape [3,256,256] \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 cls_prof.py \
-m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
--vdsp_params ../../../data/configs/resnet_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 4000 \
--shape [3,256,256] \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 0
```



### cls_prof.py 运行结果示例

以下结果是 880MHz 测试结果
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 8
  throughput (qps): 3231.27
  latency (us):
    avg latency: 7391
    min latency: 5140
    max latency: 8204
    p50 latency: 7388
    p90 latency: 7426
    p95 latency: 7433
    p99 latency: 7443


# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 932.25
  latency (us):
    avg latency: 1071
    min latency: 1065
    max latency: 1363
    p50 latency: 1070
    p90 latency: 1075
    p95 latency: 1077
    p99 latency: 1086

```