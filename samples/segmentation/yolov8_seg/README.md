# YoloV8 Segmentation Sample

本 sample 基于 yolov8 模型实现 目标检测与分割 功能


## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/ultralytics/ultralytics)  [modelzoo](http://gitlabdev.vastai.com/VastML/algorithm_modelzoo/-/tree/develop/segmentation/yolov8_seg) |
|  输入 shape |   [ (1,3,1024,1024) ]     |
| INT8量化方式 |   -         |
|  官方精度 | Detection  "mAP@.5:.95": - ;  Segmentation  "mAP@.5:.95":  40.8   |
|  VACC FP16  精度 | Detection  "mAP@.5:.95": 48.4 ;  Segmentation  "mAP@.5:.95":  38.6  |
|  VACC INT8  精度 | Detection  "mAP@.5:.95": 48.0 ;  Segmentation  "mAP@.5:.95":  38.3   |


## 数据准备

下载模型 yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 det_coco_val 到 /opt/vastai/vaststreamx/data/datasets 里


## C++ Sample 

### yolov8_seg 命令行参数说明
```bash
options:
  -m, --model_prefix             model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/linux_models/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod])
      --hw_config                hw-config file of the model suite (string [=])
      --vdsp_params              vdsp preprocess parameter file (string [=../data/configs/yolov8seg_bgr888.json])
  -d, --device_id                device id to run (unsigned int [=0])
      --elf_file                 elf file path (string [=/opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc])
      --threshold                threshold for detection (float [=0.5])
      --label_file               label file (string [=../data/labels/coco2id.txt])
      --input_file               input image file (string [=../data/images/detect.jpg])
      --output_file              output image file (string [=])
      --dataset_filelist         input dataset filelist (string [=])
      --dataset_root             input dataset root (string [=])
      --dataset_output_folder    dataset output folder (string [=])
  -?, --help                     print this message
```
### yolov8_seg 命令行示例
```bash
# 测试单张图片，分割结果保存到 yolov8_seg_result.jpg
./vaststreamx-samples/bin/yolov8_seg \
-m /opt/vastai/vaststreamx/data/models/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../data/configs/yolov8seg_bgr888.json \
--device_id 0 \
--elf_file /opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc \
--threshold 0.4 \
--label_file ../data/labels/coco2id.txt \
--input_file ../data/images/cycling.jpg \
--output_file yolov8_seg_result.jpg



./vaststreamx-samples/bin/yolov8_seg \
-m  /work/yolov8m-seg-fp16-none-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../data/configs/yolov8seg_bgr888.json \
--device_id 0 \
--elf_file /opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc \
--threshold 0.4 \
--label_file ../data/labels/coco2id.txt \
--input_file ../data/images/cycling.jpg \
--output_file yolov8_seg_result.jpg


# 测试数据集
mkdir -p yolov8_seg_out
./vaststreamx-samples/bin/yolov8_seg \
-m /opt/vastai/vaststreamx/data/models/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../data/configs/yolov8seg_bgr888.json \
--device_id 0 \
--elf_file /opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc \
--threshold 0.01 \
--label_file ../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder ./yolov8_seg_out

# 统计精度
python3 ../evaluation/yolov8_seg/yolov8_seg_eval.py \
--output_path yolov8_seg_out \
--gt ../evaluation/yolov8_seg/instances_val2017.json
```
### yolov8_seg 命令行结果示例

```bash
# 测试单张图片
Object class: bicycle, score: 0.875, bbox: [74.3125, 201.5, 253.875, 499.5]
Object class: person, score: 0.855469, bbox: [0.390625, 0.390625, 63.4688, 361.75]
Object class: person, score: 0.800781, bbox: [71.6875, 0, 275, 461.75]
Object class: person, score: 0.492188, bbox: [254.5, 17.6719, 334, 369.5]
Object class: bicycle, score: 0.4375, bbox: [243, 233.75, 334, 500]

# 精度统计结果示例
Evaluate annotation type *bbox*
DONE (t=28.19s).
Accumulating evaluation results...
DONE (t=4.75s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.480
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.643
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.522
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.283
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.534
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.641
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.363
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.582
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.610
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.670
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.768
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=31.43s).
Accumulating evaluation results...
DONE (t=4.68s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.383
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.608
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.406
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.176
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.435
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.556
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.305
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.488
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.258
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.553
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.674
```
分割结果保存于 yolov8_seg_result.jpg

### yolov8_seg_prof 命令行参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/yolov8seg_bgr888.json])
  -d, --device_ids      device id to run (string [=[0]])
      --elf_file        elf file path (string [=/opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        instance number for each device (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (int [=1024])
      --percentiles     percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=1])
  -?, --help            print this message
```


### yolov8_seg_prof 命令行示例
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/yolov8_seg_prof \
-m /opt/vastai/vaststreamx/data/models/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../data/configs/yolov8seg_bgr888.json \
--elf_file /opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc \
--device_ids [0] \
--shape "[3,640,640]" \
--batch_size 1 \
--instance 6 \
--iterations 600 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/yolov8_seg_prof \
-m /opt/vastai/vaststreamx/data/models/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../data/configs/yolov8seg_bgr888.json \
--elf_file /opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc \
--device_ids [0] \
--shape "[3,640,640]" \
--batch_size 1 \
--instance 1 \
--iterations 100 \
--queue_size 0
```
### yolov8_seg_prof 命令行结果示例
```bash
# 测试最大吞吐
- number of instances: 6
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 100.836
  latency (us):
    avg latency: 171527
    min latency: 24346
    max latency: 331879
    p50 latency: 174224
    p90 latency: 229919
    p95 latency: 245406
    p99 latency: 272607

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 18.297
  latency (us):
    avg latency: 54652
    min latency: 54378
    max latency: 58446
    p50 latency: 54538
    p90 latency: 54782
    p95 latency: 55037
    p99 latency: 57628
```


## Python Sample

### yolov8_seg.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  --elf_file ELF_FILE   input file
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
  --dataset_output_folder DATASET_OUTPUT_FOLDER
                        dataset output folder
```
### yolov8_seg.py 命令行示例
```bash
#测试单张图片
python3 yolov8_seg.py \
-m /opt/vastai/vaststreamx/data/models/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../../../data/configs/yolov8seg_bgr888.json \
--device_id 0 \
--elf_file /opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc \
--threshold 0.4 \
--label_file ../../../data/labels/coco2id.txt \
--input_file ../../../data/images/cycling.jpg \
--output_file yolov8_seg_result.jpg

# 测试数据集
mkdir -p yolov8_seg_out
python3 yolov8_seg.py \
-m /opt/vastai/vaststreamx/data/models/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../../../data/configs/yolov8seg_bgr888.json \
--device_id 0 \
--elf_file /opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc \
--threshold 0.01 \
--label_file ../../../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder ./yolov8_seg_out

# 统计精度
python3 ../../../evaluation/yolov8_seg/yolov8_seg_eval.py \
--output_path yolov8_seg_out \
--gt ../../../evaluation/yolov8_seg/instances_val2017.json
```

### yolov8_seg.py 命令行结果示例
```bash
#测试单张图片结果
Object class: bicycle, score: 0.875, bbox: [74.3125, 201.5, 253.875, 499.5]
Object class: person, score: 0.85546875, bbox: [0.390625, 0.390625, 63.46875, 361.75]
Object class: person, score: 0.80078125, bbox: [71.6875, 0.0, 275.0, 461.75]
Object class: person, score: 0.4921875, bbox: [254.5, 17.671875, 334.0, 369.5]
Object class: bicycle, score: 0.4375, bbox: [243.0, 233.75, 334.0, 500.0]

# 精度统计结果
Evaluate annotation type *bbox*
DONE (t=27.24s).
Accumulating evaluation results...
DONE (t=4.75s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.480
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.643
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.522
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.283
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.534
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.641
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.363
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.582
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.610
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.670
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.768
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=30.79s).
Accumulating evaluation results...
DONE (t=4.53s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.383
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.608
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.406
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.176
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.435
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.556
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.305
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.488
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.258
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.553
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.674

```


### yolov8_seg_prof.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  --elf_file ELF_FILE   input file
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

### yolov8_seg_prof.py 命令行示例

```bash
#测试最大吞吐
python3 yolov8_seg_prof.py \
-m /opt/vastai/vaststreamx/data/models/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../../../data/configs/yolov8seg_bgr888.json \
--elf_file /opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc \
--device_ids [0] \
--shape "[3,640,640]" \
--batch_size 1 \
--instance 6 \
--iterations 600 \
--queue_size 1


#测试最小时延 
python3 yolov8_seg_prof.py \
-m /opt/vastai/vaststreamx/data/models/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../../../data/configs/yolov8seg_bgr888.json \
--elf_file /opt/vastai/vaststreamx/data/elf/yolov8_seg_post_proc \
--device_ids [0] \
--shape "[3,640,640]" \
--batch_size 1 \
--instance 1 \
--iterations 600 \
--queue_size 0
```


### yolov8_seg_prof.py 命令行结果示例
```bash
#测试最大吞吐
- number of instances: 6
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 86.42
  latency (us):
    avg latency: 203965
    min latency: 107201
    max latency: 331215
    p50 latency: 206207
    p90 latency: 241031
    p95 latency: 251699
    p99 latency: 277311

#测试最小时延 
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 15.57
  latency (us):
    avg latency: 64231
    min latency: 63545
    max latency: 66051
    p50 latency: 64215
    p90 latency: 64528
    p95 latency: 64624
    p99 latency: 65188
```
