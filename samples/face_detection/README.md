# FACE DETECTION SAMPLE

本目录提供基于 retinaface-resnet50 模型的 人脸检测 sample. 该模型除了检测人脸框，还检测 5个 face landmarks.


## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/biubug6/Pytorch_Retinaface)  [modelzoo](http://gitlabdev.vastai.com/VastML/algorithm_modelzoo/-/tree/develop/face_detection/retinaface) |
|  输入 shape |   [ (1,3,640,640) ]     |
| INT8量化方式 |   percentile          |
|  官方精度 | "Easy": 94.33, "Medium": 90.90, "Hard": 66.40 |
|  VACC FP16  精度 | "Easy": 94.316, "Medium": 90.814, "Hard": 65.69   |
|  VACC INT8  精度 | "Easy": 94.199, "Medium": 90.45, "Hard": 62.585   |


## 数据准备

下载模型 retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 widerface_val 到 /opt/vastai/vaststreamx/data/datasets 里


## C++ Sample

### face_detection 命令行参数说明
```bash
options:
  -m, --model_prefix             model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod])
      --hw_config                hw-config file of the model suite (string [=])
      --vdsp_params              vdsp preprocess parameter file (string [=./data/configs/retinaface_rgbplanar.json])
  -d, --device_id                device id to run (unsigned int [=0])
      --threshold                threshold for detection (float [=0.5])
      --input_file               input file (string [=../data/images/face.jpg])
      --output_file              output file (string [=face_det_result.jpg])
      --dataset_filelist         dataset filename list (string [=])
      --dataset_root             dataset root (string [=])
      --dataset_output_folder    dataset output folder path (string [=])
  -?, --help                     print this message
```

### face_detection 命令行示例
在build目录下执行
```bash
# 单图片测试示例
./vaststreamx-samples/bin/face_detection \
-m /opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../data/configs/retinaface_rgbplanar.json \
--device_id 0 \
--threshold 0.5 \
--input_file ../data/images/face.jpg \
--output_file face_det_result.jpg

# 数据集测试示例
mkdir -p facedet_out
./vaststreamx-samples/bin/face_detection \
-m /opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../data/configs/retinaface_rgbplanar.json \
--device_id 0 \
--threshold 0.001 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/widerface_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder facedet_out

# 统计精度
python3 ../evaluation/face_detection/evaluation.py \
-g ../evaluation/face_detection/ground_truth \
-p ./facedet_out

```
### face_detection 命令结果示例

```bash
# 单张图片测试结果示例，检测结果保存在face_det_result.jpg里
Face bboxes and landmarks:
Index:0, score: 0.999023, bbox: [270, 144, 524, 820], landmarks: [ [398.4,479.6] [633.6,485.6] [501.6,634.4] [419.6,764.8] [601.6,768.8] ]

# 精度统计结果示例
Easy   Val AP: 0.9419949307634238
Medium Val AP: 0.9045098397938389
Hard   Val AP: 0.6258506557055097
```
### face_detection_prof 命令行参数说明

```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/retinaface_rgbplanar.json])
  -d, --device_ids      device id to run (string [=[0]])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        instance number or range for each device (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (int [=10240])
      --percentiles     percentiles of latency (string [=[50, 90, 95, 99]])
      --threshold       threshold for detection (float [=0.01])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=1])
  -?, --help            print this message
```

### face_detection_prof 命令行示例
在build目录里执行
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/face_detection_prof \
-m /opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../data/configs/retinaface_rgbplanar.json \
--device_ids [0] \
--batch_size  2 \
--instance  1 \
--shape "[3,640,640]" \
--iterations 1000 \
--percentiles "[50,90,95,99]" \
--threshold 0.01 \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/face_detection_prof \
-m /opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../data/configs/retinaface_rgbplanar.json \
--device_ids [0] \
--batch_size  1 \
--instance  1 \
--shape "[3,640,640]" \
--iterations 1500 \
--percentiles "[50,90,95,99]" \
--threshold 0.01 \
--input_host 1 \
--queue_size 0
```
### face_detection_prof 命令结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 2
  throughput (qps): 440.745
  latency (us):
    avg latency: 13542
    min latency: 9371
    max latency: 19661
    p50 latency: 13535
    p90 latency: 13578
    p95 latency: 13585
    p99 latency: 13597

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 280.041
  latency (us):
    avg latency: 3570
    min latency: 3551
    max latency: 4120
    p50 latency: 3570
    p90 latency: 3575
    p95 latency: 3577
    p99 latency: 3604
```
## Python Sample

### face_detection.py 命令行参数说明

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
  --input_file INPUT_FILE
                        input file
  --dataset_filelist DATASET_FILELIST
                        dataset filename list
  --dataset_root DATASET_ROOT
                        dataset root
  --dataset_output_folder DATASET_OUTPUT_FOLDER
                        dataset output folder path
  --output_file OUTPUT_FILE
                        output file
```
### face_detection.py 命令行示例
在本目录下运行
```bash
# 单张图片测试示例
python3 face_detection.py \
-m /opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../../data/configs/retinaface_rgbplanar.json \
--device_id 0 \
--threshold 0.5 \
--input_file ../../data/images/face.jpg \
--output_file face_det_result.jpg

# 数据集测试示例
mkdir -p facedet_out
python3 face_detection.py \
-m /opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../../data/configs/retinaface_rgbplanar.json \
--device_id 0 \
--threshold 0.001 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/widerface_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder ./facedet_out


# 统计精度
python3 ../../evaluation/face_detection/evaluation.py \
-g ../../evaluation/face_detection/ground_truth \
-p ./facedet_out

```
### face_detection.py 命令结果示例

检测结果保存在face_det_result.jpg里

```bash
# 单张图片测试结果示例
Face bboxes and landmarks:
Index:0,score: 0.9990234375, bbox: [270.0, 144.0, 524.0, 820.0], landmarks:[[398.4, 479.6], [633.6, 485.6], [501.6, 634.4], [419.6, 764.8], [601.6, 768.8]]

#精度统计结果示例
Easy   Val AP: 0.9397643155944696
Medium Val AP: 0.9020290739151974
Hard   Val AP: 0.6239435243984959

```

### face_detection_prof.py 命令行参数说明

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


### face_detection_prof.py 命令行示例
在本目录运行
```bash
# 测试最大吞吐
python3 face_detection_prof.py \
-m /opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../../data/configs/retinaface_rgbplanar.json \
--device_ids [0] \
--batch_size  2 \
--instance  1 \
--shape "[3,640,640]" \
--iterations 1500 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 face_detection_prof.py \
-m /opt/vastai/vaststreamx/data/models/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../../data/configs/retinaface_rgbplanar.json \
--device_ids [0] \
--batch_size  1 \
--instance  1 \
--shape "[3,640,640]" \
--iterations 1500 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```
### face_detection_prof.py 命令结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 2
  throughput (qps): 439.31
  latency (us):
    avg latency: 13606
    min latency: 9869
    max latency: 18987
    p50 latency: 13605
    p90 latency: 13648
    p95 latency: 13660
    p99 latency: 13683

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 272.87
  latency (us):
    avg latency: 3663
    min latency: 3647
    max latency: 4293
    p50 latency: 3662
    p90 latency: 3670
    p95 latency: 3673
    p99 latency: 3682
```
