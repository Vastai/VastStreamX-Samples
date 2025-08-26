# DETR Sample

本 目录 提供基于 detr_r50 模型的 目标检测 sample

## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/facebookresearch/detr)  [modelzoo](http://gitlabdev.vastai.com/VastML/algorithm_modelzoo/-/tree/develop/detection/detr) |
|  输入 shape |   [ (1,3,1066,800) ]     |
| INT8量化方式 |   -          |
|  官方精度 |  "mAP@.5":   58.3 ;     "mAP@.5:.95":  37.8    |
|  VACC FP16  精度 | "mAP@.5":  58.3 ;  "mAP@.5:.95":  37.8  |
|  VACC INT8  精度 | - |

## 数据准备
下载模型 detr_res50-fp16-none-1_3_1066_800-vacc 到 /opt/vastai/vaststreamx/data/models/
下载数据集 det_coco_val 到 /opt/vastai/vaststreamx/data/datasets

## C++ Sample

### detr 命令行参数说明
```bash
options:
  -m, --model_prefix             model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod])
      --hw_config                hw-config file of the model suite (string [=])
      --vdsp_params              vdsp preprocess parameter file (string [=./data/configs/detr_bgr888.json])
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
### detr 命令行示例

```bash
# 测试单张图片
./vaststreamx-samples/bin/detr \
-m /opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod \
--vdsp_params ../data/configs/detr_bgr888.json  \
--device_id 0 \
--threshold 0.5 \
--label_file ../data/labels/coco91.txt \
--input_file ../data/images/dog.jpg \
--output_file detr_result.jpg

#检测结果如下
Detection objects:
Object class: cup, score: 0.670273, bbox: [426.188, 106.27, 22.125, 30.6683]
Object class: potted plant, score: 0.536071, bbox: [57.1172, 91.8727, 27.3281, 47.4703]
Object class: dog, score: 0.999476, bbox: [131.391, 217.169, 179.719, 324.547]
Object class: bicycle, score: 0.999589, bbox: [124.5, 126.851, 440.625, 288.819]
Object class: truck, score: 0.995513, bbox: [469.969, 80.2549, 211.313, 89.6939]
#检测框绘制于detr_result.jpg

# 数据集测试
mkdir -p ./detr_out
vaststreamx-samples/bin/detr \
-m /opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod \
--vdsp_params ../data/configs/detr_bgr888.json  \
--device_id 0 \
--threshold 0.01 \
--label_file ../data/labels/coco91.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder ./detr_out
#结果保存于文件夹 detr_out


# 统计精度
python3 ../evaluation/detection/eval_map.py \
--gt ../evaluation/detection/instances_val2017.json \
--txt ./detr_out

# 精度统计结果
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.378
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.583
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.395
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.143
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.413
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.594
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.309
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.478
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.506
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.227
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.556
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.749
{'bbox_mAP': 0.378, 'bbox_mAP_50': 0.583, 'bbox_mAP_75': 0.395, 'bbox_mAP_s': 0.143, 'bbox_mAP_m': 0.413, 'bbox_mAP_l': 0.594, 'bbox_mAP_copypaste': '0.378 0.583 0.395 0.143 0.413 0.594'}
```

### detr_prof 命令行参数说明

```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=./data/configs/detr_bgr888.json])
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

### detr_prof 命令行示例
```bash
# 测试最大吞吐
vaststreamx-samples/bin/detr_prof \
-m /opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod \
--vdsp_params ../data/configs/detr_bgr888.json  \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 200 \
--shape "[3,1066,800]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
vaststreamx-samples/bin/detr_prof \
-m /opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod \
--vdsp_params ../data/configs/detr_bgr888.json  \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 150 \
--shape "[3,1066,800]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0

```

### detr_prof 命令行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 29.4654
  latency (us):
    avg latency: 101071
    min latency: 37345
    max latency: 104893
    p50 latency: 101705
    p90 latency: 101851
    p95 latency: 101888
    p99 latency: 101956


# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 27.4558
  latency (us):
    avg latency: 36421
    min latency: 36261
    max latency: 37348
    p50 latency: 36409
    p90 latency: 36530
    p95 latency: 36610
    p99 latency: 36714
```

## Python Sample

### detr.py 命令行参数说明
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

### detr.py 命令行示例
```bash
# 测试单张图片
python3  detr.py \
-m /opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod \
--vdsp_params ../../../data/configs/detr_bgr888.json  \
--device_id 0 \
--threshold 0.5 \
--label_file ../../../data/labels/coco91.txt \
--input_file ../../../data/images/dog.jpg \
--output_file detr_result.jpg
# 单张图片测试结果示例
Detection objects:
Object class: cup, score: 0.670272707939148, bbox: [426, 106, 22, 30]
Object class: potted plant, score: 0.5360713005065918, bbox: [57, 91, 27, 47]
Object class: dog, score: 0.9994760155677795, bbox: [131, 217, 179, 324]
Object class: bicycle, score: 0.9995890259742737, bbox: [124, 126, 440, 288]
Object class: truck, score: 0.9955134987831116, bbox: [469, 80, 211, 89]
#检测框绘制于 detr_result.jpg

# 测试数据集
mkdir -p ./detr_out
python3 detr.py \
-m /opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod \
--vdsp_params ../../../data/configs/detr_bgr888.json  \
--device_id 0  \
--threshold 0.01 \
--label_file ../../../data/labels/coco91.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder ./detr_out

# 精度统计
python3 ../../../evaluation/detection/eval_map.py \
--gt ../../../evaluation/detection/instances_val2017.json \
--txt ./detr_out

# 精度结果示例
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.378
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.583
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.395
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.143
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.412
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.593
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.309
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.506
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.227
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.555
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.748
{'bbox_mAP': 0.378, 'bbox_mAP_50': 0.583, 'bbox_mAP_75': 0.395, 'bbox_mAP_s': 0.143, 'bbox_mAP_m': 0.412, 'bbox_mAP_l': 0.593, 'bbox_mAP_copypaste': '0.378 0.583 0.395 0.143 0.412 0.593'}

```




### detr_prof.py 命令行参数说明
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
### detr_prof.py 命令行示例
```bash
# 测试最大吞吐
python3 detr_prof.py \
-m /opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod \
--vdsp_params ../../../data/configs/detr_bgr888.json  \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 200 \
--shape "[3,1066,800]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 detr_prof.py \
-m /opt/vastai/vaststreamx/data/models/detr_res50-fp16-none-1_3_1066_800-vacc/mod \
--vdsp_params ../../../data/configs/detr_bgr888.json  \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 100 \
--shape "[3,1066,800]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### detr_prof.py 命令行结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 29.47
  latency (us):
    avg latency: 101243
    min latency: 37701
    max latency: 102691
    p50 latency: 101719
    p90 latency: 101889
    p95 latency: 101922
    p99 latency: 102028


# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 27.58
  latency (us):
    avg latency: 36258
    min latency: 36069
    max latency: 37441
    p50 latency: 36218
    p90 latency: 36415
    p95 latency: 36467
    p99 latency: 36623
```