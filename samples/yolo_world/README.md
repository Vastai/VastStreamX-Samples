# Yolo World Sample

本目录提供以及 yolo-world 模型的目标检测 sample


## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/AILab-CVC/YOLO-World)  [modelzoo](https://github.com/Vastai/VastModelZOO/tree/main/cv/detection/yolo_world) |
|  输入 shape |  "text": [(1,16),(1,16)], "image":[(1,3,1280,1280),(1203,512)]  |
| INT8量化方式 |   -          |
|  官方精度 |  "mAP@.5":   45.5 ;     "mAP@.5:.95": 34.6  |
|  VACC FP16  精度 | "mAP@.5":  45.9 ;  "mAP@.5:.95":  34.8  |
|  VACC INT8  精度 | - |


## 数据准备

下载 yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc 模型到 /opt/vastai/vaststreamx/data/models/
下载 yolo_world_text-fp16-none-1_16_1_16-vacc 模型到 /opt/vastai/vaststreamx/data/models/
下载数据集 det_coco_val 到 /opt/vastai/vaststreamx/data/datasets 里
下载 tokenizer  到 /opt/vastai/vaststreamx/data/tokenizer 里

## C++ Sample

### yolo_world 命令参数说明
```bash
options:
      --imgmod_prefix          image model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod])
      --imgmod_hw_config       hw-config file of the model suite (string [=])
      --imgmod_vdsp_params     vdsp preprocess parameter file (string [=../data/configs/yolo_world_1280_1280_bgr888.json])
      --txtmod_prefix          model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod])
      --txtmod_hw_config       hw-config file of the model suite (string [=])
      --txtmod_vdsp_params     vdsp preprocess parameter file (string [=../data/configs/clip_txt_vdsp.json])
  -d, --device_id              device id to run (unsigned int [=0])
      --max_per_image          max_per_image (unsigned int [=300])
      --score_thresh           threshold for detection (float [=0.5])
      --iou_thresh             iou threshold (float [=0.7])
      --nms_pre                nms_pre (unsigned int [=30000])
      --label_file             npz filelist of input strings (string [=])
      --npz_files_path         npz filelist of input strings (string [=])
      --input_file             input file (string [=../data/images/CLIP.png])
      --output_file            output file (string [=yolo_world_result.jpg])
      --dataset_filelist       input dataset filelist (string [=])
      --dataset_root           input dataset root (string [=])
      --dataset_output_file    dataset output file (string [=yolo_world_dataset_output.json])
  -?, --help                   print this message
```



### yolo_world 命令运行示例

```bash
在build目录里执行

# 用python 脚本生成token
mkdir -p tokens
python3 ../samples/yolo_world/make_tokens.py \
--class_text ../data/labels/lvis_v1_class_texts.json \
--tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/clip-vit-base-patch32 \
--save_path tokens

# 测试单张图片
./vaststreamx-samples/bin/yolo_world \
--imgmod_prefix  /opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod \
--imgmod_vdsp_params ../data/configs/yolo_world_1280_1280_bgr888.json \
--txtmod_prefix /opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod \
--txtmod_vdsp_params  ../data/configs/clip_txt_vdsp.json \
--device_id  0 \
--label_file ../data/labels/lvis_v1_class_texts.json \
--npz_files_path tokens \
--input_file ../data/images/dog.jpg \
--output_file yolo_world_result.jpg


# 测试数据集
./vaststreamx-samples/bin/yolo_world \
--imgmod_prefix  /opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod \
--imgmod_vdsp_params ../data/configs/yolo_world_1280_1280_bgr888.json \
--txtmod_prefix /opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod \
--txtmod_vdsp_params  ../data/configs/clip_txt_vdsp.json \
--device_id  0 \
--label_file ../data/labels/lvis_v1_class_texts.json \
--npz_files_path tokens \
--max_per_image 300 \
--score_thresh  0.001 \
--iou_thresh  0.7 \
--nms_pre  30000 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file yoloworld_dataset_result.json

# 统计精度
python3 ../evaluation/yolo_world/eval_lvis.py  \
--path_res yoloworld_dataset_result.json \
--path_ann_file ../evaluation/yolo_world/lvis_v1_minival_inserted_image_name.json

```
### yolo_world 命令运行结果示例
```bash
# 单张图片测试结果
Detection objects:
Object class: bicycle, score: 0.836795, bbox: [125.25, 132.3, 568.5, 421.95]
Object class: canister, score: 0.613604, bbox: [689.1, 128.981, 716.062, 154.341]
Object class: pickup truck, score: 0.612216, bbox: [465.6, 72.825, 689.85, 170.738]
Object class: trash can, score: 0.610639, bbox: [427.575, 109.069, 447.705, 134.405]
Object class: pug-dog, score: 0.58237, bbox: [131.325, 219.6, 311.587, 541.5]
Object class: dog, score: 0.575108, bbox: [131.325, 219.6, 311.587, 541.5]
Object class: canister, score: 0.510268, bbox: [427.734, 109.209, 447.539, 134.438]

# 精度统计结果
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=all] = 0.348
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=300 catIds=all] = 0.459
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=300 catIds=all] = 0.379
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=     s | maxDets=300 catIds=all] = 0.254
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=     m | maxDets=300 catIds=all] = 0.456
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=     l | maxDets=300 catIds=all] = 0.540
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=  r] = 0.292
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=  c] = 0.331
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=  f] = 0.373
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=all] = 0.450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=     s | maxDets=300 catIds=all] = 0.291
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=     m | maxDets=300 catIds=all] = 0.564
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=     l | maxDets=300 catIds=all] = 0.669

```
### yolo_world_text 模型性能测试

yolo_world_text_prof 命令参数 

```bash
options:
  -m, --model_prefix     model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod])
      --hw_config        hw-config file of the model suite (string [=])
      --vdsp_params      vdsp preprocess parameter file (string [=../data/configs/clip_txt_vdsp.json])
  -d, --device_ids       device id to run (string [=[0]])
  -b, --batch_size       profiling batch size of the model (unsigned int [=1])
  -i, --instance         instance number or range for each device (unsigned int [=1])
      --iterations       iterations count for one profiling (int [=1024])
      --percentiles      percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host       cache input data into host memory (bool [=0])
  -q, --queue_size       aync wait queue size (unsigned int [=2])
      --test_npz_file    npz_file for test (string [=])
  -?, --help             print this message
```

```bash

# 测试 yolo_world_text 模型最大吞吐
./vaststreamx-samples/bin/yolo_world_text_prof \
--model_prefix /opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod \
--vdsp_params  ../data/configs/clip_txt_vdsp.json \
--test_npz_file ./tokens/Bible.npz  \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 2000 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试 yolo_world_text 模型最小时延
./vaststreamx-samples/bin/yolo_world_text_prof \
--model_prefix /opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod \
--vdsp_params  ../data/configs/clip_txt_vdsp.json \
--test_npz_file ./tokens/Bible.npz  \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 2000 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### yolo_world_text 模型性能测试结果

以下结果为 x86_linux  OCLK=835MHz条件下测试所得
```bash
# 测试 yolo_world_text 模型最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 461.997
  latency (us):
    avg latency: 6428
    min latency: 3189
    max latency: 7328
    p50 latency: 6432
    p90 latency: 6464
    p95 latency: 6494
    p99 latency: 6514


# 测试 yolo_world_text 模型最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 378.999
  latency (us):
    avg latency: 2637
    min latency: 2629
    max latency: 2955
    p50 latency: 2636
    p90 latency: 2640
    p95 latency: 2644
    p99 latency: 2699
```
以下结果为 VS1000 aarch64_linux VE1M OCLK=1250MHz条件下测试所得
```bash
# 测试 yolo_world_text 模型最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 485.404
  latency (us):
    avg latency: 5843
    min latency: 2986
    max latency: 26265
    p50 latency: 5788
    p90 latency: 5967
    p95 latency: 6035
    p99 latency: 7924


# 测试 yolo_world_text 模型最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 255.245
  latency (us):
    avg latency: 3916
    min latency: 2834
    max latency: 25528
    p50 latency: 3786
    p90 latency: 4231
    p95 latency: 4391
    p99 latency: 7427
```

### yolo_world_image 模型性能测试

yolo_world_image 命令参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/yolo_world_1280_1280_bgr888.json])
  -d, --device_ids      device id to run (string [=[0]])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        instance number for each device (unsigned int [=1])
      --iterations      iterations count for one profiling (int [=10240])
      --percentiles     percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=1])
  -?, --help            print this message

```
yolo_world_image 命令示例
```bash

# 测试 yolo_world_image 模型最大吞吐
./vaststreamx-samples/bin/yolo_world_image_prof \
--model_prefix /opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod  \
--vdsp_params ../data/configs/yolo_world_1280_1280_bgr888.json \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 20 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1


# 测试 yolo_world_image 模型最小时延
./vaststreamx-samples/bin/yolo_world_image_prof \
--model_prefix /opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod  \
--vdsp_params ../data/configs/yolo_world_1280_1280_bgr888.json \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 20 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```
### yolo_world_image 模型性能测试结果
以下结果为 x86_linux  OCLK=835MHz条件下测试所得
后处理时延:350ms左右
```bash
# 测试 yolo_world_image 模型最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 2.96069
  latency (us):
    avg latency: 930305
    min latency: 452515
    max latency: 1118020
    p50 latency: 994318
    p90 latency: 1000310
    p95 latency: 1001156
    p99 latency: 1118020

# 测试 yolo_world_image 模型最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 2.24123
  latency (us):
    avg latency: 446181
    min latency: 443401
    max latency: 452529
    p50 latency: 445603
    p90 latency: 450141
    p95 latency: 450995
    p99 latency: 452529
```

以下结果为 VS1000 aarch64_linux VE1M OCLK=1250MHz条件下测试所得
后处理时延:120s左右
```bash
# 测试 yolo_world_image 模型最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 2.98702
  latency (us):
    avg latency: 895330
    min latency: 530363
    max latency: 1144797
    p50 latency: 922202
    p90 latency: 943437
    p95 latency: 1119705
    p99 latency: 1144797

# 测试 yolo_world_image 模型最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 1.89199
  latency (us):
    avg latency: 528541
    min latency: 519631
    max latency: 556078
    p50 latency: 524637
    p90 latency: 548540
    p95 latency: 551595
    p99 latency: 556078
```
## Python Sample

### yolo_world.py 脚本参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  --imgmod_prefix IMGMOD_PREFIX
                        image model prefix of the model suite files
  --imgmod_hw_config IMGMOD_HW_CONFIG
                        image model hw-config file of the model suite
  --imgmod_vdsp_params IMGMOD_VDSP_PARAMS
                        vdsp preprocess parameter file
  --txtmod_prefix TXTMOD_PREFIX
                        text model prefix of the model suite files
  --txtmod_hw_config TXTMOD_HW_CONFIG
                        text model hw-config file of the model suite
  --txtmod_vdsp_params TXTMOD_VDSP_PARAMS
                        text model vdsp preprocess parameter file
  --tokenizer_path TOKENIZER_PATH
                        tokenizer path
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --max_per_image MAX_PER_IMAGE
                        max objects detected per image
  --score_thres SCORE_THRES
                        object confidence threshold
  --iou_thres IOU_THRES
                        iou threshold
  --nms_pre NMS_PRE     nms_pre
  --label_file LABEL_FILE
                        label file
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file
  --dataset_filelist DATASET_FILELIST
                        input dataset filelist
  --dataset_root DATASET_ROOT
                        input dataset root
  --dataset_output_file DATASET_OUTPUT_FILE
                        dataset output file
```

### yolo_world.py 脚本运行示例

```bash
# 单张图片示例
python3 yolo_world.py \
--imgmod_prefix  /opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod \
--imgmod_vdsp_params ../../data/configs/yolo_world_1280_1280_bgr888.json \
--txtmod_prefix /opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod \
--txtmod_vdsp_params  ../../data/configs/clip_txt_vdsp.json \
--tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/clip-vit-base-patch32 \
--device_id  0 \
--max_per_image 300 \
--score_thres  0.5 \
--iou_thres  0.7 \
--nms_pre  30000 \
--label_file ../../data/labels/lvis_v1_class_texts.json \
--input_file ../../data/images/dog.jpg \
--output_file yolo_world_result.jpg


# 测试数据集
python3 yolo_world.py \
--imgmod_prefix  /opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod \
--imgmod_vdsp_params ../../data/configs/yolo_world_1280_1280_bgr888.json \
--txtmod_prefix /opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod \
--txtmod_vdsp_params  ../../data/configs/clip_txt_vdsp.json \
--tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/clip-vit-base-patch32 \
--device_id  0 \
--max_per_image 300 \
--score_thres  0.001 \
--iou_thres  0.7 \
--nms_pre  30000 \
--label_file ../../data/labels/lvis_v1_class_texts.json \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file yoloworld_dataset_result.json

# 统计精度
python3 ../../evaluation/yolo_world/eval_lvis.py  \
--path_res yoloworld_dataset_result.json \
--path_ann_file ../../evaluation/yolo_world/lvis_v1_minival_inserted_image_name.json

```

### yolo_world.py 脚本运行结果示例

```bash
# 单张图片测试结果
Object class: bicycle, score: 0.8368, bbox: [125.25 132.3  568.5  421.95]
Object class: canister, score: 0.6136, bbox: [689.1     128.98125 716.0625  154.34062]
Object class: pickup truck, score: 0.6122, bbox: [465.6     72.825  689.85   170.7375]
Object class: trash can, score: 0.6106, bbox: [427.575   109.06875 447.70547 134.4047 ]
Object class: pug-dog, score: 0.5824, bbox: [131.325  219.6    311.5875 541.5   ]
Object class: dog, score: 0.5751, bbox: [131.325  219.6    311.5875 541.5   ]
Object class: canister, score: 0.5103, bbox: [427.73438 109.20937 447.53906 134.4375 ]

# 精度统计结果
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=all] = 0.348
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=300 catIds=all] = 0.457
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=300 catIds=all] = 0.379
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=     s | maxDets=300 catIds=all] = 0.255
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=     m | maxDets=300 catIds=all] = 0.456
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=     l | maxDets=300 catIds=all] = 0.540
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=  r] = 0.292
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=  c] = 0.331
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=  f] = 0.372
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=all] = 0.452
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=     s | maxDets=300 catIds=all] = 0.293
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=     m | maxDets=300 catIds=all] = 0.566
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=     l | maxDets=300 catIds=all] = 0.673
```
### yolo_world_text 模型性能测试

```bash
# 测试 yolo_world_text 模型最大吞吐
python3 yolo_world_text_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod \
--vdsp_params  ../../data/configs/clip_txt_vdsp.json \
--tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/clip-vit-base-patch32 \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 2000 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试 yolo_world_text 模型最小时延
python3 yolo_world_text_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/models/yolo_world_text-fp16-none-1_16_1_16-vacc/mod \
--vdsp_params  ../../data/configs/clip_txt_vdsp.json \
--tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/clip-vit-base-patch32 \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 1500 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```
### yolo_world_text 模型性能测试结果

```bash
# 测试 yolo_world_text 模型最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 449.86
  latency (us):
    avg latency: 6621
    min latency: 3745
    max latency: 7227
    p50 latency: 6623
    p90 latency: 6675
    p95 latency: 6691
    p99 latency: 6714


# 测试 yolo_world_text 模型最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 357.71
  latency (us):
    avg latency: 2794
    min latency: 2746
    max latency: 3847
    p50 latency: 2799
    p90 latency: 2807
    p95 latency: 2811
    p99 latency: 2854
```


### yolo_world_image 模型性能测试
```bash
# 测试 yolo_world_image 模型最大吞吐
python3 yolo_world_image_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod  \
--vdsp_params ../../data/configs/yolo_world_1280_1280_bgr888.json \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 20 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1


# 测试 yolo_world_image 模型最小时延
python3 yolo_world_image_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/models/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc/mod  \
--vdsp_params ../../data/configs/yolo_world_1280_1280_bgr888.json \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 20 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0

```

### yolo_world_image 模型性能测试结果

```bash
# 测试 yolo_world_image 模型最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 2.96
  latency (us):
    avg latency: 962423
    min latency: 427634
    max latency: 1088937
    p50 latency: 997505
    p90 latency: 1001379
    p95 latency: 1001482
    p99 latency: 1071446


# 测试 yolo_world_image 模型最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 2.42
  latency (us):
    avg latency: 413146
    min latency: 408812
    max latency: 418684
    p50 latency: 412929
    p90 latency: 415679
    p95 latency: 416207
    p99 latency: 418189
```


