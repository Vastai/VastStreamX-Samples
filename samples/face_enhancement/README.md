# Face Enhancement sample

本目录提供基于 gpen 模型的 人脸增强 sample. 

## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/yangxy/GPEN)  [modelzoo](http://gitlabdev.vastai.com/VastML/algorithm_modelzoo/-/tree/develop/super_resolution/gpen) |
|  输入 shape |   [ (1,3,512,512) ]     |
| INT8量化方式 |   mse          |
|  官方精度 | "PSNR": - , "SSIM": - |
|  VACC FP16  精度 | "PSNR": 26.114 , "SSIM":  0.691  |
|  VACC INT8  精度 | "PSNR":  25.856 , "SSIM": 0.666  |


## 数据准备

下载模型 gpen-int8-mse-1_3_512_512-vacc 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 GPEN 到 /opt/vastai/vaststreamx/data/datasets 里


## C++ sample

### face_enhancement 命令行参数说明
```bash
options:
  -m, --model_prefix             model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models//gpen-int8-mse-1_3_512_512-vacc/mod])
      --hw_config                hw-config file of the model suite (string [=])
      --vdsp_params              vdsp preprocess parameter file (string [=../data/configs/gpen_bgr888.json])
  -d, --device_id                device id to run (unsigned int [=0])
      --input_file               input image (string [=../data/images/face.jpg])
      --output_file              output image (string [=face_result.jpg])
      --dataset_filelist         input dataset filelist (string [=])
      --dataset_root             input dataset root (string [=])
      --dataset_output_folder    dataset output folder (string [=])
  -?, --help                     print this message
```

### face_enhancement 命令行示例
在build目录里执行

```bash
#单张照片示例
./vaststreamx-samples/bin/face_enhancement \
-m /opt/vastai/vaststreamx/data/models/gpen-int8-mse-1_3_512_512-vacc/mod \
--vdsp_params ../data/configs/gpen_bgr888.json \
--device_id 0 \
--input_file  ../data/images/face.jpg \
--output_file gpen_result.jpg
#结果将保存为 gpen_result.jpg

#数据集示例
mkdir -p gpen_output
./vaststreamx-samples/bin/face_enhancement \
-m /opt/vastai/vaststreamx/data/models/gpen-int8-mse-1_3_512_512-vacc/mod \
--vdsp_params ../data/configs/gpen_bgr888.json \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/GPEN/filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/GPEN/lq/ \
--dataset_output_folder gpen_output
#结果将保存在 gpen_output 目录里
```

精度测试
```bash
python3 ../evaluation/face_enhancement/eval_celeb.py \
--result ./gpen_output \
--gt /opt/vastai/vaststreamx/data/datasets/GPEN/hq
```
精度输出为
```bash
mean psnr: 25.856292241565264, mean ssim: 0.6661125573509248
```

### face_enhancement_prof 命令行参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/gpen-int8-mse-1_3_512_512-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/gpen_bgr888.json])
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

### face_enhancement_prof 运行示例
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/face_enhancement_prof \
-m /opt/vastai/vaststreamx/data/models/gpen-int8-mse-1_3_512_512-vacc/mod \
--vdsp_params ../data/configs/gpen_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 120 \
--shape "[3,512,512]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/face_enhancement_prof \
-m /opt/vastai/vaststreamx/data/models/gpen-int8-mse-1_3_512_512-vacc/mod \
--vdsp_params ../data/configs/gpen_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 120 \
--shape "[3,512,512]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### face_enhancement_prof 运行结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 20.545
  latency (us):
    avg latency: 145594
    min latency: 50895
    max latency: 147976
    p50 latency: 145711
    p90 latency: 146507
    p95 latency: 146754
    p99 latency: 147314

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 19.8255
  latency (us):
    avg latency: 50436
    min latency: 49349
    max latency: 52056
    p50 latency: 50413
    p90 latency: 50924
    p95 latency: 51089
    p99 latency: 51536
```


## Python sample

### face_enhancement.py 命令行参数说明
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
  --output_file OUTPUT_FILE
                        output file
  --dataset_filelist DATASET_FILELIST
                        dataset filelist
  --dataset_root DATASET_ROOT
                        input dataset root
  --dataset_output_folder DATASET_OUTPUT_FOLDER
                        dataset output folder
```

### face_enhancement.py 命令行示例

```bash
#单张图片测试，结果将保存为 gpen_result.jpg
python3 face_enhancement.py \
-m /opt/vastai/vaststreamx/data/models/gpen-int8-mse-1_3_512_512-vacc/mod \
--vdsp_params ../../data/configs/gpen_bgr888.json \
--device_id 0 \
--input_file ../../data/images/face.jpg \
--output_file gpen_result.jpg

#数据集测试，结果保存于 gpen_output 文件夹里
mkdir -p gpen_output
python3 face_enhancement.py \
-m /opt/vastai/vaststreamx/data/models/gpen-int8-mse-1_3_512_512-vacc/mod \
--vdsp_params ../../data/configs/gpen_bgr888.json \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/GPEN/filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/GPEN/lq \
--dataset_output_folder gpen_output
```

精度测试
```bash
python3 ../../evaluation/face_enhancement/eval_celeb.py \
--result ./gpen_output \
--gt /opt/vastai/vaststreamx/data/datasets/GPEN/hq
```
输出精度为
```bash
mean psnr: 25.856292241565264, mean ssim: 0.6661125573509248
```

### face_enhancement_prof.py 命令行参数说明

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


### face_enhancement_prof.py 运行示例

在本目录下运行  
```bash
# 测试最大吞吐
python3 face_enhancement_prof.py \
-m /opt/vastai/vaststreamx/data/models/gpen-int8-mse-1_3_512_512-vacc/mod \
--vdsp_params ../../data/configs/gpen_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,512,512]" \
--iterations 120 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 face_enhancement_prof.py \
-m /opt/vastai/vaststreamx/data/models/gpen-int8-mse-1_3_512_512-vacc/mod \
--vdsp_params ../../data/configs/gpen_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,512,512]" \
--iterations 120 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```


### face_enhancement_prof.py 运行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 20.53
  latency (us):
    avg latency: 145777
    min latency: 53108
    max latency: 148966
    p50 latency: 145883
    p90 latency: 146692
    p95 latency: 146975
    p99 latency: 147402

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 19.35
  latency (us):
    avg latency: 51650
    min latency: 50532
    max latency: 53277
    p50 latency: 51629
    p90 latency: 52154
    p95 latency: 52273
    p99 latency: 52585
```

