# Mask2Fomer Segmentation Sample

本 sample 基于 mask2former 模型实现 目标检测与分割 功能

## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/facebookresearch/Mask2Former)  [modelzoo](https://github.com/Vastai/VastModelZOO/tree/main/cv/segmentation/mask2former) |
|  输入 shape |   [ (1,3,1024,1024) ]     |
| INT8量化方式 |   -         |
|  官方精度 | Detection  "mAP@.5:.95": - ;  Segmentation  "mAP@.5:.95":  43.7   |
|  VACC FP16  精度 | Detection  "mAP@.5:.95": 42.1 ;  Segmentation  "mAP@.5:.95":  42.1  |
|  VACC INT8  精度 |  -  |


## 数据准备

下载模型 mask2former-fp16-none-1_3_1024_1024-vacc 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 det_coco_val 到 /opt/vastai/vaststreamx/data/datasets 里


## C++ Sample 

### mask2former 命令行参数说明
```bash
options:
  -m, --model_prefix           model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod])
      --hw_config              hw-config file of the model suite (string [=])
      --vdsp_params            vdsp preprocess parameter file (string [=../data/configs/mask2former_rgbplanar.json])
  -d, --device_id              device id to run (unsigned int [=0])
      --threshold              threshold for detection (float [=0.5])
      --label_file             label file (string [=../data/labels/coco2id.txt])
      --input_file             input image file (string [=../data/images/cycling.jpg])
      --output_file            output image file (string [=])
      --dataset_filelist       input dataset filelist (string [=])
      --dataset_root           input dataset root (string [=])
      --dataset_output_file    dataset output file (string [=])
  -?, --help                   print this message
```

### mask2former 命令行示例
```bash
# 测试单张图片，分割结果保存到 mask2former_result.jpg
./vaststreamx-samples/bin/mask2former  \
--model_prefix /opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod \
--vdsp_params ../data/configs/mask2former_rgbplanar.json \
--device_id 0 \
--threshold 0.6 \
--label_file ../data/labels/coco2id.txt \
--input_file ../data/images/cycling.jpg \
--output_file mask2former_result.jpg


# 测试数据集
./vaststreamx-samples/bin/mask2former  \
--model_prefix /opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod \
--vdsp_params ../data/configs/mask2former_rgbplanar.json \
--device_id 0 \
--threshold 0.001 \
--label_file  ../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file mask2former_predictions.json


# 统计精度
python3 ../evaluation/coco_seg/coco_seg_eval.py \
--prediction_file mask2former_predictions.json \
--gt ../evaluation/coco_seg/instances_val2017.json
```

### mask2former 命令行结果示例

```bash
# 测试单张图片
Object class: person, score: 0.96586, bbox: [0, 0, 60, 364]
Object class: bicycle, score: 0.942803, bbox: [83, 200, 252, 499]
Object class: person, score: 0.939369, bbox: [71, 0, 271, 459]
Object class: bicycle, score: 0.923028, bbox: [271, 242, 333, 499]
Object class: person, score: 0.919342, bbox: [245, 19, 333, 435]
Object class: bicycle, score: 0.817861, bbox: [0, 147, 47, 434]

# 精度统计结果示例
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=36.81s).
Accumulating evaluation results...
DONE (t=6.41s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.629
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.446
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.226
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.458
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.609
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.349
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.567
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.603
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.381
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.644
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.812
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=40.88s).
Accumulating evaluation results...
DONE (t=6.34s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.639
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.235
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.594
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.339
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.546
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.575
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.370
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.608
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.769
```

### mask2former_prof 命令行参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/mask2former_rgbplanar.json])
  -d, --device_ids      device id to run (string [=[0]])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        instance number for each device (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (int [=1024])
      --percentiles     percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=1])
  -?, --help            print this message
```


```bash
# 测试最大吞吐
./vaststreamx-samples/bin/mask2former_prof \
--model_prefix /opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod \
--vdsp_params ../data/configs/mask2former_rgbplanar.json \
--device_ids [0] \
--shape "[3,1024,1024]" \
--batch_size 1 \
--instance 1 \
--iterations 10 \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/mask2former_prof \
--model_prefix /opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod \
--vdsp_params ../data/configs/mask2former_rgbplanar.json \
--device_ids [0] \
--shape "[3,1024,1024]" \
--batch_size 1 \
--instance 1 \
--iterations 10 \
--input_host 1 \
--queue_size 0
```

### mask2former_prof 命令行结果示例
以下结果为 x86_linux  OCLK=835MHz条件下测试所得
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 0.3592
  latency (us):
    avg latency: 7517587
    min latency: 2830881
    max latency: 8392207
    p50 latency: 8334797
    p90 latency: 8346086
    p95 latency: 8392207
    p99 latency: 8392207

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 0.353967
  latency (us):
    avg latency: 2825122
    min latency: 2819540
    max latency: 2830540
    p50 latency: 2826895
    p90 latency: 2827612
    p95 latency: 2830540
    p99 latency: 2830540
```

以下结果为 VS1000 aarch64_linux VE1M OCLK=1250MHz条件下测试所得
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 0.358114
  latency (us):
    avg latency: 6155742
    min latency: 2874985
    max latency: 8419721
    p50 latency: 5672085
    p90 latency: 8411426
    p95 latency: 8419721
    p99 latency: 8419721

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 0.348005
  latency (us):
    avg latency: 2873516
    min latency: 2859541
    max latency: 2893751
    p50 latency: 2872828
    p90 latency: 2893539
    p95 latency: 2893751
    p99 latency: 2893751
```


## Python Sample

### mask2former.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  --label_file LABEL_FILE
                        label file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --threshold THRESHOLD
                        detection threshold
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file
  --dataset_filelist DATASET_FILELIST
                        input dataset image list
  --dataset_root DATASET_ROOT
                        input dataset root
  --dataset_output_file DATASET_OUTPUT_FILE
                        dataset output file
```

### mask2former.py 命令行示例

```bash
#测试单张图片
python3 mask2former.py \
--model_prefix /opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod \
--vdsp_params ../../../data/configs/mask2former_rgbplanar.json \
--device_id 0 \
--threshold 0.6 \
--label_file ../../../data/labels/coco2id.txt \
--input_file ../../../data/images/cycling.jpg \
--output_file mask2former_result.jpg

# 测试数据集
python3 mask2former.py  \
--model_prefix /opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod \
--vdsp_params ../../../data/configs/mask2former_rgbplanar.json \
--device_id 0 \
--threshold 0.001 \
--label_file  ../../../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file mask2former_predictions.json

# 统计精度
python3 ../../../evaluation/coco_seg/coco_seg_eval.py \
--prediction_file mask2former_predictions.json \
--gt ../../../evaluation/coco_seg/instances_val2017.json
```

### mask2former.py 命令行结果示例
```bash
#测试单张图片结果
Object class: person, score: 0.9659856557846069, bbox: [0, 0, 60, 364]
Object class: bicycle, score: 0.9422694444656372, bbox: [83, 200, 252, 499]
Object class: person, score: 0.9388745427131653, bbox: [71, 0, 271, 459]
Object class: person, score: 0.9240907430648804, bbox: [245, 19, 333, 435]
Object class: bicycle, score: 0.9223312139511108, bbox: [272, 242, 333, 499]
Object class: bicycle, score: 0.8169117569923401, bbox: [0, 147, 47, 434]
# 精度统计结果
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=36.81s).
Accumulating evaluation results...
DONE (t=6.41s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.629
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.446
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.226
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.458
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.609
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.349
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.567
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.603
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.381
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.644
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.812
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=40.88s).
Accumulating evaluation results...
DONE (t=6.34s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.639
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.235
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.594
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.339
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.546
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.575
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.370
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.608
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.769
```

### mask2former_prof.py 命令行参数说明
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
### mask2former_prof.py 命令行示例

```bash
#测试最大吞吐
python3 mask2former_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod \
--vdsp_params ../../../data/configs/mask2former_rgbplanar.json \
--device_ids [0] \
--shape "[3,1024,1024]" \
--batch_size 1 \
--instance 1 \
--iterations 10 \
--input_host 1 \
--queue_size 1


#测试最小时延 
python3 mask2former_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/models/mask2former-fp16-none-1_3_1024_1024-vacc/mod \
--vdsp_params ../../../data/configs/mask2former_rgbplanar.json \
--device_ids [0] \
--shape "[3,1024,1024]" \
--batch_size 1 \
--instance 1 \
--iterations 10 \
--input_host 1 \
--queue_size 0
```

### mask2former_prof.py 命令行结果示例
```bash
#测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 0.36
  latency (us):
    avg latency: 7652739
    min latency: 3083156
    max latency: 8616688
    p50 latency: 8330389
    p90 latency: 8353901
    p95 latency: 8485294
    p99 latency: 8590409

#测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 0.32
  latency (us):
    avg latency: 3081544
    min latency: 3073319
    max latency: 3101773
    p50 latency: 3080806
    p90 latency: 3084219
    p95 latency: 3092996
    p99 latency: 3100018
```