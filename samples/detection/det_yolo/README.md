# Object Detection

本目录提供基于 yolo 模型的 Detection  sample

## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/ultralytics/yolov5)  [modelzoo](https://github.com/Vastai/VastModelZOO/tree/main/cv/detection/yolov5) |
|  输入 shape |   [ (1,3,640,640) ]     |
| INT8量化方式 |   percentile          |
|  官方精度 |  "mAP@.5":   64.1  ;     "mAP@.5:.95":  45.4    |
|  VACC FP16  精度 | "mAP@.5":   63.4 ;  "mAP@.5:.95":  44.8  |
|  VACC INT8  精度 | "mAP@.5":   63.0 ;  "mAP@.5:.95":   43.3  |

## 数据准备

下载模型 yolov5m-int8-percentile-1_3_640_640-vacc-pipeline 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 det_coco_val 到 /opt/vastai/vaststreamx/data/datasets 里

## C++ sample

### detection 命令行参数说明
```bash
options:
  -m, --model_prefix             model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/yolov5m-int8-max-1_3_640_640-vacc-pipeline/mod])
      --hw_config                hw-config file of the model suite (string [=])
      --vdsp_params              vdsp preprocess parameter file (string [=./data/configs/yolo_div255_bgr888.json])
  -d, --device_id                device id to run (unsigned int [=0])
      --threshold                threshold for detection (float [=0.5])
      --label_file               label file (string [=../data/labels/coco2id.txt])
      --input_file               input file (string [=../data/images/dog.jpg])
      --output_file              output file (string [=result.png])
      --dataset_filelist         dataset filename list (string [=])
      --dataset_root             dataset root (string [=])
      --dataset_output_folder    dataset output folder path (string [=])
  -?, --help                     print this message
```
### detection 运行示例
在 build 目录里执行  

```bash
#跑单张图片
./vaststreamx-samples/bin/detection \
-m /opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../data/configs/yolo_div255_bgr888.json  \
--device_id 0 \
--threshold 0.5 \
--label_file ../data/labels/coco2id.txt \
--input_file ../data/images/dog.jpg \
--output_file result.png


#跑数据集
mkdir -p ./det_out
./vaststreamx-samples/bin/detection \
-m /opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../data/configs/yolo_div255_bgr888.json  \
--device_id 0 \
--threshold 0.01 \
--label_file ../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder ./det_out


# 统计精度
python3 ../evaluation/detection/eval_map.py \
--gt ../evaluation/detection/instances_val2017.json \
--txt ./det_out


```
### detection 运行结果示例
```bash
# 单张图片结果示例，检测框绘制在 result.png 图片里
Detection objects:
Object class: bicycle, score: 0.926758, bbox: [121.8, 130.8, 443.4, 288.6]
Object class: dog, score: 0.912109, bbox: [131.7, 224.7, 177, 318.9]
Object class: truck, score: 0.728027, bbox: [472.8, 72.3, 217.2, 99]


# 精度统计结果示例
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.433
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.630
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.474
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.266
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.484
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.338
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.551
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.591
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.398
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.644
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.745
{'bbox_mAP': 0.433, 'bbox_mAP_50': 0.63, 'bbox_mAP_75': 0.474, 'bbox_mAP_s': 0.266, 'bbox_mAP_m': 0.484, 'bbox_mAP_l': 0.565, 'bbox_mAP_copypaste': '0.433 0.630 0.474 0.266 0.484 0.565'}


```
### det_prof 命令行参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/yolov5m-int8-max-1_3_640_640-vacc-pipeline/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=./data/configs/yolo_div255_bgr888.json])
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
### det_prof 运行示例
在 build 目录里执行  
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/det_prof \
-m /opt/vastai/vaststreamx/data/models/yolov5m-int8-max-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../data/configs/yolo_div255_bgr888.json  \
--device_ids [0] \
--batch_size 2 \
--instance 1 \
--iterations 2000 \
--shape "[3,640,640]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/det_prof \
-m /opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../data/configs/yolo_div255_bgr888.json  \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 2000 \
--shape "[3,640,640]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```
### det_prof 运行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 2
  throughput (qps): 471.332
  latency (us):
    avg latency: 12662
    min latency: 10385
    max latency: 17695
    p50 latency: 12658
    p90 latency: 12699
    p95 latency: 12712
    p99 latency: 12732

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 223.296
  latency (us):
    avg latency: 4477
    min latency: 4405
    max latency: 5048
    p50 latency: 4484
    p90 latency: 4493
    p95 latency: 4496
    p99 latency: 4517
```

## Python sample 功能测试

### detection.py 命令行参数说明
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

### detection.py 运行示例

在本目录下运行  
```bash
# 单张图片测试
python3 detection.py \
-m /opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../../../data/configs/yolo_div255_bgr888.json  \
--device_id 0 \
--threshold 0.5 \
--label_file ../../../data/labels/coco2id.txt \
--input_file ../../../data/images/dog.jpg \
--output_file detection_result.jpg

#数据集测试
mkdir -p ./det_out
python3 detection.py \
-m /opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../../../data/configs/yolo_div255_bgr888.json  \
--device_id 0  \
--threshold 0.01 \
--label_file ../../../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder ./det_out

# 精度统计
python3 ../../../evaluation/detection/eval_map.py \
--gt ../../../evaluation/detection/instances_val2017.json \
--txt ./det_out


```

### detection.py 运行结果示例

```bash
#单张图片结果示例,如果指定了输出文件，则可以在输出文件 detection_result.jpg 中看到检测框
Detection objects:
Object class: bicycle, score: 0.9268, bbox: [121.80, 130.80, 443.40, 288.60]
Object class: dog, score: 0.9121, bbox: [131.70, 224.70, 177.00, 318.90]
Object class: truck, score: 0.7280, bbox: [472.80, 72.30, 217.20, 99.00]



#精度统计结果示例
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.432
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.630
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.473
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.266
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.483
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.563
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.550
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.397
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.643
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.743
{'bbox_mAP': 0.432, 'bbox_mAP_50': 0.63, 'bbox_mAP_75': 0.473, 'bbox_mAP_s': 0.266, 'bbox_mAP_m': 0.483, 'bbox_mAP_l': 0.563, 'bbox_mAP_copypaste': '0.432 0.630 0.473 0.266 0.483 0.563'}

```


## Python sample 性能测试

### det_prof.py 命令行参数说明

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


### det_prof.py 运行示例

在本目录下运行  
```bash
# 测试最大吞吐
python3 det_prof.py \
-m /opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../../../data/configs/yolo_div255_bgr888.json  \
--device_ids [0] \
--batch_size 2 \
--instance 1 \
--iterations 1000 \
--shape "[3,640,640]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 det_prof.py \
-m /opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../../../data/configs/yolo_div255_bgr888.json  \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 1000 \
--shape "[3,640,640]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```



### det_prof.py 运行结果示例

```bash

# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 2
  throughput (qps): 470.64
  latency (us):
    avg latency: 12697
    min latency: 10604
    max latency: 18169
    p50 latency: 12690
    p90 latency: 12740
    p95 latency: 12758
    p99 latency: 12785



# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 218.98
  latency (us):
    avg latency: 4565
    min latency: 4497
    max latency: 5226
    p50 latency: 4570
    p90 latency: 4589
    p95 latency: 4594
    p99 latency: 4636


```
