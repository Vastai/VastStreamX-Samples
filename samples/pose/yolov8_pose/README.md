# YoloV8 Pose Sample

本 sample 基于 yolov8 模型实现 Pose 功能

## 模型信息

| 模型信息       | 值                                                                                                                                              |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 来源           | [github](https://github.com/ultralytics/ultralytics) [modelzoo](http://192.168.20.70/VastML/algorithm_modelzoo/-/tree/develop/pose/yolov8_pose) |
| 输入 shape     | [ (1,3,640,640) ]                                                                                                                               |
| INT8量化方式   | percentile                                                                                                                                      |
| 官方精度       | Pose "mAP@.5:.95":60.0                                                                                                                          |
| VACC FP16 精度 | Pose "mAP@.5:.95": 59.4                                                                                                                         |
| VACC INT8 精度 | Pose "mAP@.5:.95": 57.1                                                                                                                         |

## 数据准备

下载模型 yolov8s-pose-int8-percentile-1_3_640_640-vacc-pipeline 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 det_coco_val 到 /opt/vastai/vaststreamx/data/datasets 里

## Python Sample

### yolov8_pose.py 命令行参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  --file_path FILE_PATH
                        dir path
  --model_prefix_path MODEL_PREFIX_PATH
                        model info
  --vdsp_params_info VDSP_PARAMS_INFO
                        vdsp op info
  --label_txt LABEL_TXT
                        label txt
  --draw_output         save output image or not
  --threashold THREASHOLD
                        threashold for postprocess
  --device_id DEVICE_ID
                        device id
  --batch BATCH         bacth size
  --save_dir SAVE_DIR   save_dir
```

### yolov8_pose.py 命令行示例

```bash
#测试单张图片
python3 yolov8_pose.py \
--file_path /opt/vastai/vaststreamx/data/datasets/det_coco_val/000000342128.jpg \
--model_prefix_path /opt/vastai/vaststreamx/data/models/yolov8s-pose-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params_info ../../../data/configs/yolov8s_pose.json \
--label_txt ../../../data/labels/coco2id.txt \
--device_id 0 \
--save_dir ./yolov8_pose_out \
--draw_output

# 测试数据集
python3 yolov8_pose.py \
--file_path /opt/vastai/vaststreamx/data/datasets/det_coco_val \
--model_prefix_path /opt/vastai/vaststreamx/data/models/yolov8s-pose-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params_info ../../../data/configs/yolov8s_pose.json \
--label_txt ../../../data/labels/coco2id.txt \
--device_id 0 \
--save_dir ./yolov8_pose_out

# 统计精度
python3 ../../../evaluation/yolov8_pose/yolov8_pose_eval.py \
--gt ../../../evaluation/yolov8_pose/person_keypoints_val2017.json \
--pred yolov8_pose_out/predictions.json
```

### yolov8_pose.py 命令行结果示例

```bash
# 单张图片结果
图片结果将保存到 yolov8_pose_out 目录下

# 精度统计结果
Accumulating evaluation results...
DONE (t=0.76s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.488
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.650
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.538
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.173
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.672
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.799
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.194
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.515
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.576
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.226
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.765
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.868
Running per image evaluation...
Evaluate annotation type *keypoints*
DONE (t=6.00s).
Accumulating evaluation results...
DONE (t=0.23s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.570
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.847
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.630
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.500
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.680
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.640
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.894
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.703
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.560
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.752
```

### yolov8_pose_prof.py 命令行参数说明

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

### yolov8_pose_prof.py 命令行示例

```bash
#测试最大吞吐
python3 yolov8_pose_prof.py \
-m /opt/vastai/vaststreamx/data/models/yolov8s-pose-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../../../data/configs/yolov8s_pose.json \
--device_ids [0] \
--shape "[3,640,640]" \
--batch_size 1 \
--instance 5 \
--iterations 4000 \
--queue_size 1


#测试最小时延
python3 yolov8_pose_prof.py \
-m /opt/vastai/vaststreamx/data/models/yolov8s-pose-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../../../data/configs/yolov8s_pose.json \
--device_ids [0] \
--shape "[3,640,640]" \
--batch_size 1 \
--instance 1 \
--iterations 1000 \
--queue_size 0
```

### yolov8_pose_prof.py 命令行结果示例

```bash
#测试最大吞吐
- number of instances: 5
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 591.13
  latency (us):
    avg latency: 25153
    min latency: 15173
    max latency: 49770
    p50 latency: 24982
    p90 latency: 28864
    p95 latency: 29930
    p99 latency: 32118

#测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 79.79
  latency (us):
    avg latency: 12530
    min latency: 11662
    max latency: 18061
    p50 latency: 12432
    p90 latency: 13105
    p95 latency: 13163
    p99 latency: 13268
```
