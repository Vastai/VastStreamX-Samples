# FACE RECOGNITION SAMPLE

本目录提供基于 facenet 的人脸特征提取 sample 


## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/timesler/facenet-pytorch)  [modelzoo](http://gitlabdev.vastai.com/VastML/algorithm_modelzoo/-/tree/develop/face_recognize/facenet) |
|  输入 shape |   [ (1,3,160,160) ]     |
| INT8量化方式 |   percentile          |
|  官方精度 | "ACC":0.9965, "AUC":0.9997,   "ERR":0.0050   |
|  VACC FP16  精度 |  "ACC":0.99367, "AUC":0.99959,   "ERR":0.00556   |
|  VACC INT8  精度 | "ACC":0.99317, "AUC":0.99957,   "ERR":0.00633   |



## 数据准备

下载模型 facenet_vggface2-int8-percentile-1_3_160_160-vacc 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 lfw_mtcnnpy_160 到 /opt/vastai/vaststreamx/data/datasets 里


## C++ Sample 

### face_recognition 命令行参数说明
```bash
options:
  -m, --model_prefix             model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod])
      --hw_config                hw-config file of the model suite (string [=])
      --vdsp_params              vdsp preprocess parameter file (string [=../data/configs/facenet_bgr888.json])
  -d, --device_id                device id to run (unsigned int [=0])
      --input_file               input file (string [=../data/images/face.jpg])
      --dataset_filelist         input dataset image list (string [=])
      --dataset_root             input dataset root (string [=])
      --dataset_output_folder    dataset output folder (string [=])
  -?, --help                     print this message
```
### face_recognition 命令行示例
在build目录里执行

单个人脸图片示例
```bash
./vaststreamx-samples/bin/face_recognition \
-m /opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod \
--vdsp_params ../data/configs/facenet_bgr888.json \
--device_id  0 \
--input_file ../data/images/face.jpg
```
结果将输出 512 维人脸特征


跑人脸数据集示例   
在build 目录下执行   
```bash
mkdir -p facenet_output

./vaststreamx-samples/bin/face_recognition \
-m /opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod \
--vdsp_params ../data/configs/facenet_bgr888.json \
--device_id  0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/lfw_mtcnnpy_160_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder facenet_output
```
结果保存在 facenet_output 文件夹

测数据集精度    
在build目录下执行
```bash
python3 ../evaluation/face_recognition/facenet_eval.py \
--gt_dir /opt/vastai/vaststreamx/data/datasets/lfw_mtcnnpy_160 \
--gt_pairs_path /opt/vastai/vaststreamx/data/datasets/lfw_mtcnnpy_160_pairs.txt \
--input_npz_path /opt/vastai/vaststreamx/data/datasets/lfw_mtcnnpy_160_filelist.txt \
--out_npz_dir ./facenet_output
```

精度输出为
```bash
Accuracy: 0.99317+-0.00369
Validation rate: 0.98200+-0.01087 @ FAR=0.00100
Area Under Curve (AUC): 0.99957
Equal Error Rate (EER): 0.00633
```

### face_recognition_prof 命令行参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/facenet_bgr888.json])
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

### face_recognition_prof 命令行示例

在build 目录里执行
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/face_recognition_prof \
-m /opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod \
--vdsp_params ../data/configs/facenet_bgr888.json \
--device_ids [0] \
--batch_size 64 \
--instance 2 \
--iterations 300 \
--shape "[3,160,160]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/face_recognition_prof \
-m /opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod \
--vdsp_params ../data/configs/facenet_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 5000 \
--shape "[3,160,160]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### face_recognition_prof 结果示例

```bash
# 测试最大吞吐
- number of instances: 2
  devices: [ 0 ]
  queue size: 1
  batch size: 64
  throughput (qps): 2758.58
  latency (us):
    avg latency: 138454
    min latency: 44469
    max latency: 160124
    p50 latency: 138992
    p90 latency: 139377
    p95 latency: 139412
    p99 latency: 139786

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 863.868
  latency (us):
    avg latency: 1156
    min latency: 1143
    max latency: 2948
    p50 latency: 1156
    p90 latency: 1159
    p95 latency: 1160
    p99 latency: 1166
```
## Python Sample

### face_recognition.py 命令行参数说明
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
  --input_file INPUT_FILE
                        input file
  --dataset_filelist DATASET_FILELIST
                        input dataset image list
  --dataset_root DATASET_ROOT
                        input dataset root
  --dataset_output_folder DATASET_OUTPUT_FOLDER
                        dataset output folder
```
### face_recognition.py 命令行示例
在当前目录执行
```bash
# 测试单张图片
python3 face_recognition.py \
-m /opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod \
--vdsp_params ../../data/configs/facenet_bgr888.json \
--device_id  0 \
--input_file ../../data/images/face.jpg
# 结果将输出 512 维人脸特征



# 测试数据集
mkdir -p facenet_output
python3 face_recognition.py \
-m /opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod \
--vdsp_params ../../data/configs/facenet_bgr888.json \
--device_id  0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/lfw_mtcnnpy_160_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder facenet_output
#结果保存在 facenet_output 文件夹
```

测数据集精度    
```bash
python3 ../../evaluation/face_recognition/facenet_eval.py \
--gt_dir /opt/vastai/vaststreamx/data/datasets/lfw_mtcnnpy_160 \
--gt_pairs_path /opt/vastai/vaststreamx/data/datasets/lfw_mtcnnpy_160_pairs.txt \
--input_npz_path /opt/vastai/vaststreamx/data/datasets/lfw_mtcnnpy_160_filelist.txt \
--out_npz_dir ./facenet_output
```
输出精度
```bash
Accuracy: 0.99317+-0.00369
Validation rate: 0.98200+-0.01087 @ FAR=0.00100
Area Under Curve (AUC): 0.99957
Equal Error Rate (EER): 0.00633
```

### face_recognition_prof.py 命令行参数说明
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



### face_recognition_prof.py 命令行示例
在当前目录执行
```bash
# 测试最大吞吐
python3 face_recognition_prof.py \
-m /opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod \
--vdsp_params ../../data/configs/facenet_bgr888.json \
--device_ids [0] \
--batch_size 64 \
--instance 2 \
--iterations 300 \
--shape "[3,160,160]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 face_recognition_prof.py \
-m /opt/vastai/vaststreamx/data/models/facenet_vggface2-int8-percentile-1_3_160_160-vacc/mod \
--vdsp_params ../../data/configs/facenet_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 5000 \
--shape "[3,160,160]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```
### face_recognition_prof.py 结果示例

```bash
# 测试最大吞吐
- number of instances: 2
  devices: [0]
  queue size: 1
  batch size: 64
  throughput (qps): 2759.46
  latency (us):
    avg latency: 138470
    min latency: 41007
    max latency: 142401
    p50 latency: 139037
    p90 latency: 139164
    p95 latency: 139471
    p99 latency: 139777

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 846.47
  latency (us):
    avg latency: 1180
    min latency: 1152
    max latency: 2232
    p50 latency: 1179
    p90 latency: 1183
    p95 latency: 1185
    p99 latency: 1190
```