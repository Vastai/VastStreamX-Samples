# Rt-Detr Sample

本 目录 提供基于 rt-detr 模型的目标检测 sample


## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/lyuwenyu/RT-DETR)  [modelzoo](https://github.com/Vastai/VastModelZOO/tree/main/cv/detection/rtdetr) |
|  输入 shape |   [ (1,3,1066,800) ]     |
| INT8量化方式 |   -          |
|  官方精度 |  "mAP@.5":   63.8 ;     "mAP@.5:.95":  46.5    |
|  VACC FP16  精度 | "mAP@.5":  61.9 ;  "mAP@.5:.95":  45.1  |
|  VACC INT8  精度 | - |

## 数据准备
下载模型 rtdetr-fp16-none-1_3_640_640-vacc 到 /opt/vastai/vaststreamx/data/models/
下载数据集 det_coco_val 到 /opt/vastai/vaststreamx/data/datasets


## C++ Sample

### rtdetr 命令行参数说明

```bash
usage: ./vaststreamx-samples/bin/rtdetr [options] ... 
options:
  -m, --model_prefix             model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod])
      --hw_config                hw-config file of the model suite (string [=])
      --vdsp_params              vdsp preprocess parameter file (string [=./data/configs/rtdetr_bgr888.json])
  -d, --device_id                device id to run (unsigned int [=0])
      --threshold                threshold for detection (float [=0.5])
      --label_file               label file (string [=./data/labels/coco2id.txt])
      --input_file               input file (string [=./data/images/dog.jpg])
      --output_file              output file (string [=result.png])
      --dataset_filelist         dataset filename list (string [=])
      --dataset_root             dataset root (string [=])
      --dataset_output_folder    dataset output folder path (string [=])
  -?, --help                     print this message
```
### rtdetr 命令行示例
```bash
# 测试单张图片
./vaststreamx-samples/bin/rtdetr \
-m /opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod \
--vdsp_params ../data/configs/rtdetr_bgr888.json  \
--device_id 0 \
--threshold 0.5 \
--label_file ../data/labels/coco2id.txt \
--input_file ../data/images/dog.jpg \
--output_file rtdetr_result.jpg


#检测结果如下
Detection objects:
Object class: dog, score: 0.930711, bbox: [131.344, 222.328, 179.062, 319.219]
Object class: truck, score: 0.622459, bbox: [467.25, 74.4961, 224.25, 96.5391]
Object class: bicycle, score: 0.931213, bbox: [125.812, 135.281, 443.25, 285.469]
#检测框绘制于rtdetr_result.jpg

# 测试数据集
mkdir -p ./rtdetr_out
./vaststreamx-samples/bin/rtdetr \
-m /opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod \
--vdsp_params ../data/configs/rtdetr_bgr888.json  \
--device_id 0  \
--threshold 0.001 \
--label_file ../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder ./rtdetr_out


# 精度统计
python3 ../evaluation/detection/eval_map.py \
--gt ../evaluation/detection/instances_val2017.json \
--txt ./rtdetr_out

# 精度结果示例
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.451
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.619
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.488
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.268
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.483
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.609
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.641
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.431
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.681
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.812
{'bbox_mAP': 0.451, 'bbox_mAP_50': 0.619, 'bbox_mAP_75': 0.488, 'bbox_mAP_s': 0.268, 'bbox_mAP_m': 0.483, 'bbox_mAP_l': 0.609, 'bbox_mAP_copypaste': '0.451 0.619 0.488 0.268 0.483 0.609'}

```
### rtdetr_prof 命令行参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=./data/configs/rtdetr_bgr888.json])
  -d, --device_ids      device id to run (string [=[0]])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        instance number or range for each device (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (int [=10240])
      --percentiles     percentiles of latency (string [=[50,90,95,99]])
      --threshold       threshold for detection (float [=0.5])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=2])
  -?, --help            print this message

```
### rtdetr_prof 命令行示例
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/rtdetr_prof \
-m /opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod \
--vdsp_params ../data/configs/rtdetr_bgr888.json  \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 100 \
--shape "[3,640,640]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/rtdetr_prof \
-m /opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod \
--vdsp_params ../data/configs/rtdetr_bgr888.json  \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 100 \
--shape "[3,640,640]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0

```
### rtdetr_prof 命令行结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 16.2462
  latency (us):
    avg latency: 182127
    min latency: 65145
    max latency: 188043
    p50 latency: 184479
    p90 latency: 184846
    p95 latency: 184907
    p99 latency: 185203

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 15.5863
  latency (us):
    avg latency: 64158
    min latency: 63852
    max latency: 65046
    p50 latency: 64139
    p90 latency: 64359
    p95 latency: 64401
    p99 latency: 64483
```

## Python Sample

### rtdetr.py 命令行参数说明
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
  --threshold THRESHOLD
                        device id to run
  --label_file LABEL_FILE
                        label file
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file
  --dataset_filelist DATASET_FILELIST
                        dataset filename list
  --dataset_root DATASET_ROOT
                        dataset root
  --dataset_output_folder DATASET_OUTPUT_FOLDER
                        dataset output folder path
```
### rtdetr.py 命令行示例
```bash
# 测试单张图片
python3 rtdetr.py \
-m /opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod \
--vdsp_params ../../../data/configs/rtdetr_bgr888.json  \
--device_id 0 \
--threshold 0.5 \
--label_file ../../../data/labels/coco2id.txt \
--input_file ../../../data/images/dog.jpg \
--output_file rtdetr_result.jpg
# 单张图片测试结果示例
Detection objects:
Object class: dog, label:16, score: 0.9307105541229248, bbox: [131, 222, 179, 319]
Object class: truck, label:7, score: 0.622459352016449, bbox: [467, 74, 224, 96]
Object class: bicycle, label:1, score: 0.9312127232551575, bbox: [125, 135, 443, 285]
#检测框绘制于 rtdetr_result.jpg


# 测试数据集
mkdir -p ./rtdetr_out
python3 rtdetr.py \
-m /opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod \
--vdsp_params ../../../data/configs/rtdetr_bgr888.json  \
--device_id 0  \
--threshold 0.001 \
--label_file ../../../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder ./rtdetr_out

# 精度统计
python3 ../../../evaluation/detection/eval_map.py \
--gt ../../../evaluation/detection/instances_val2017.json \
--txt ./rtdetr_out

# 精度结果示例
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.451
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.619
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.488
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.268
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.483
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.609
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.576
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.641
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.430
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.681
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.812
{'bbox_mAP': 0.451, 'bbox_mAP_50': 0.619, 'bbox_mAP_75': 0.488, 'bbox_mAP_s': 0.268, 'bbox_mAP_m': 0.483, 'bbox_mAP_l': 0.609, 'bbox_mAP_copypaste': '0.451 0.619 0.488 0.268 0.483 0.609'}
```


### rtdetr_prof.py 命令行参数说明
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
                        device id to run
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

### rtdetr_prof.py 命令行示例
```bash
# 测试最大吞吐
python3 rtdetr_prof.py \
-m /opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod \
--vdsp_params ../../../data/configs/rtdetr_bgr888.json  \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 100 \
--shape "[3,640,640]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1



# 测试最小时延
python3 rtdetr_prof.py \
-m /opt/vastai/vaststreamx/data/models/rtdetr-fp16-none-1_3_640_640-vacc/mod \
--vdsp_params ../../../data/configs/rtdetr_bgr888.json  \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 100 \
--shape "[3,640,640]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0

```


### rtdetr_prof.py 命令行结果示例
```bash
# 测试最大吞吐

- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 16.25
  latency (us):
    avg latency: 182759
    min latency: 64487
    max latency: 185917
    p50 latency: 184503
    p90 latency: 184847
    p95 latency: 184977
    p99 latency: 185176
# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 15.75
  latency (us):
    avg latency: 63477
    min latency: 63128
    max latency: 64112
    p50 latency: 63451
    p90 latency: 63693
    p95 latency: 63758
    p99 latency: 64080

```