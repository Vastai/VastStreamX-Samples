# DYNAMIC MODEL SAMPLE

本目录提供基于 yolov5s dynamic shape 模型的 sample


## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/ultralytics/yolov5)  [modelzoo](http://gitlabdev.vastai.com/VastML/algorithm_modelzoo/-/tree/develop/detection/yolov5) |
|  输入 shape |   [ (1,3,320,320) - (1,3,640,640) ]     |
| INT8量化方式 |   percentile          |
|  官方精度 |  "mAP@.5":   56.8  ;     "mAP@.5:.95":  37.4	   |
|  VACC FP16  精度 | "mAP@.5":   55.3 ;  "mAP@.5:.95":  36.8  |
|  VACC INT8  精度 | "mAP@.5":   54.6 ;  "mAP@.5:.95":  36.0  |

## 数据准备

下载模型 torch-yolov5s_coco-int8-percentile-Y-Y-2-none 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 det_coco_val 到 /opt/vastai/vaststreamx/data/datasets 里

## C++ Sample 

### dynamic_yolo 命令行参数说明
```bash
options:
  -m, --module_info              model info json files (string [=/opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json])
      --vdsp_params              vdsp preprocess parameter file (string [=./data/configs/yolo_div255_bgr888.json])
  -d, --device_id                device id to run (unsigned int [=0])
      --max_input_shape          model max input shape (string [=[1,3,640,640]])
      --threshold                threshold for detection (float [=0.5])
      --label_file               label file (string [=../data/labels/coco2id.txt])
      --input_file               input file (string [=../data/images/dog.jpg])
      --output_file              output file (string [=dynamic_result.jpg])
      --dataset_filelist         dataset filename list (string [=])
      --dataset_root             dataset root (string [=])
      --dataset_output_folder    dataset output folder path (string [=])
  -?, --help                     print this message
```
### dynamic_yolo 命令行示例
在build 目录执行
```bash
./vaststreamx-samples/bin/dynamic_yolo \
-m /opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json \
--vdsp_params ../data/configs/yolo_div255_bgr888.json \
--device_id 0 \
--max_input_shape "[1,3,640,640]" \
--threshold 0.5 \
--label_file ../data/labels/coco2id.txt \
--input_file ../data/images/dog.jpg \
--output_file dynamic_model_result.jpg



#跑数据集
mkdir -p ./dynamic_out
./vaststreamx-samples/bin/dynamic_yolo \
-m /opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json \
--vdsp_params ../data/configs/yolo_div255_bgr888.json \
--device_id 0 \
--max_input_shape "[1,3,640,640]" \
--threshold 0.01 \
--label_file ../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder ./dynamic_out

#统计精度
python3 ../evaluation/detection/eval_map.py \
--gt ../evaluation/detection/instances_val2017.json \
--txt ./dynamic_out


```
### dynamic_yolo 单张图片结果示例
```bash
# 单张图片结果示例，检测框绘制于 dynamic_model_result.jpg
Detection objects:
Object class: dog, score: 0.930664, bbox: [129.675, 222.6, 181.125, 325.2]
Object class: truck, score: 0.664062, bbox: [469.8, 81.6, 218.4, 89.7]
Object class: bicycle, score: 0.552734, bbox: [161.4, 120.9, 403.8, 309]
Object class: car, score: 0.502441, bbox: [463.2, 76.05, 229.8, 96.3]



#精度测试结果示例
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.360
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.546
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.392
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.197
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.412
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.465
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.297
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.492
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.343
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.592
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.679
{'bbox_mAP': 0.36, 'bbox_mAP_50': 0.546, 'bbox_mAP_75': 0.392, 'bbox_mAP_s': 0.197, 'bbox_mAP_m': 0.412, 'bbox_mAP_l': 0.465, 'bbox_mAP_copypaste': '0.360 0.546 0.392 0.197 0.412 0.465'}


```



### dynamic_yolo_prof 命令行参数说明
```bash
options:
  -m, --module_info        model info json files (string [=/opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json])
      --vdsp_params        vdsp preprocess parameter file (string [=./data/configs/yolo_div255_bgr888.json])
  -d, --device_ids         device id to run (string [=[0]])
      --max_input_shape    model max input shape (string [=[1,3,640,640]])
  -b, --batch_size         profiling batch size of the model (unsigned int [=1])
  -i, --instance           instance number or range for each device (unsigned int [=1])
  -s, --shape              model input shape (string [=])
      --iterations         iterations count for one profiling (int [=10240])
      --percentiles        percentiles of latency (string [=[50,90,95,99]])
      --threshold          threshold for detection (float [=0.5])
      --input_host         cache input data into host memory (bool [=0])
  -q, --queue_size         aync wait queue size (unsigned int [=2])
  -?, --help               print this message
```
### dynamic_yolo_prof 命令行示例
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/dynamic_yolo_prof \
-m /opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json \
--vdsp_params ../data/configs/yolo_div255_bgr888.json \
--device_ids [0] \
--max_input_shape "[1,3,640,640]" \
--threshold 0.5 \
--batch_size 1 \
--instance 2 \
--iterations 5000 \
--shape "[1,3,640,640]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/dynamic_yolo_prof \
-m /opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json \
--vdsp_params ../data/configs/yolo_div255_bgr888.json \
--device_ids [0] \
--max_input_shape "[1,3,640,640]" \
--threshold 0.5 \
--batch_size 1 \
--instance 1 \
--iterations 2000 \
--shape "[1,3,640,640]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```
### dynamic_yolo_prof 命令行结果示例
```bash
# 测试最大吞吐
- number of instances: 2
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 906.531
  latency (us):
    avg latency: 6551
    min latency: 3784
    max latency: 9650
    p50 latency: 6551
    p90 latency: 6579
    p95 latency: 6593
    p99 latency: 6606

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 309.41
  latency (us):
    avg latency: 3231
    min latency: 3213
    max latency: 3846
    p50 latency: 3224
    p90 latency: 3251
    p95 latency: 3286
    p99 latency: 3322
```

## Python Sample

### dynamic_yolo.py  命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODULE_INFO, --module_info MODULE_INFO
                        model prefix of the model suite files
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --threshold THRESHOLD
                        device id to run
  --max_input_shape MAX_INPUT_SHAPE
                        model max input shape
  --label_file LABEL_FILE
                        label file
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file
```

### dynamic_yolo.py  命令行示例
在当前目录下执行
```bash
python3 dynamic_yolo.py \
-m /opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json \
--vdsp_params ../../data/configs/yolo_div255_bgr888.json \
--device_id 0 \
--max_input_shape "[1,3,640,640]" \
--threshold 0.5 \
--label_file ../../data/labels/coco2id.txt \
--input_file ../../data/images/dog.jpg \
--output_file dynamic_model_result.jpg


#测试数据集
mkdir -p dynamic_out
python3 dynamic_yolo.py \
-m /opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json \
--vdsp_params ../../data/configs/yolo_div255_bgr888.json \
--device_id 0 \
--max_input_shape "[1,3,640,640]" \
--threshold 0.01 \
--label_file ../../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder ./dynamic_out

#统计精度
python3 ../../evaluation/detection/eval_map.py \
--gt ../../evaluation/detection/instances_val2017.json \
--txt ./dynamic_out


```
### dynamic_yolo.py 结果示例
```bash
# 单张图片结果示例
Detection objects:
Object class: dog, score: 0.9306640625, bbox: [129, 222, 181, 325]
Object class: truck, score: 0.6640625, bbox: [469, 81, 218, 89]
Object class: bicycle, score: 0.552734375, bbox: [161, 120, 403, 309]
Object class: car, score: 0.50244140625, bbox: [463, 76, 229, 96]

# 精度统计结果示例
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.359
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.546
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.201
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.411
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.463
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.296
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.590
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.675
{'bbox_mAP': 0.359, 'bbox_mAP_50': 0.546, 'bbox_mAP_75': 0.391, 'bbox_mAP_s': 0.201, 'bbox_mAP_m': 0.411, 'bbox_mAP_l': 0.463, 'bbox_mAP_copypaste': '0.359 0.546 0.391 0.201 0.411 0.463'}

```

### dynamic_yolo_prof.py  命令行参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODULE_INFO, --module_info MODULE_INFO
                        model prefix of the model suite files
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  --max_input_shape MAX_INPUT_SHAPE
                        model max input shape
  -d DEVICE_IDS, --device_ids DEVICE_IDS
                        device id to run
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        profiling batch size of the model
  -i INSTANCE, --instance INSTANCE
                        instance number for each device
  -s SHAPE, --shape SHAPE
                        data input shape
  --model_input_shape MODEL_INPUT_SHAPE
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

### dynamic_yolo_prof.py  命令行示例

```bash
#测试最大吞吐
python3 dynamic_yolo_prof.py \
-m /opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json \
--vdsp_params ../../data/configs/yolo_div255_bgr888.json \
--max_input_shape "[1,3,640,640]" \
--model_input_shape "[1,3,640,640]" \
--device_ids [0] \
--batch_size 1 \
--instance 2 \
--iterations 5000 \
--shape "[3,640,640]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

#测试最小时延
python3 dynamic_yolo_prof.py \
-m /opt/vastai/vaststreamx/data/models/torch-yolov5s_coco-int8-percentile-Y-Y-2-none/yolov5s_coco_module_info.json \
--vdsp_params ../../data/configs/yolo_div255_bgr888.json \
--max_input_shape "[1,3,640,640]" \
--model_input_shape "[1,3,640,640]" \
--device_ids [2] \
--batch_size 1 \
--instance 1 \
--iterations 2000 \
--shape "[3,640,640]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0

```

### dynamic_yolo_prof.py 命令行结果示例
```bash
# 测试最大吞吐
- number of instances: 2
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 907.00
  latency (us):
    avg latency: 6579
    min latency: 4228
    max latency: 9249
    p50 latency: 6584
    p90 latency: 6603
    p95 latency: 6609
    p99 latency: 6623

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 299.14
  latency (us):
    avg latency: 3341
    min latency: 3321
    max latency: 3921
    p50 latency: 3335
    p90 latency: 3354
    p95 latency: 3395
    p99 latency: 3450
```