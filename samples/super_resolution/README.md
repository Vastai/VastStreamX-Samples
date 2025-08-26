# Super Resolution sample

本目录提供基于 rcan, edsr 模型的 图像超分 sample。  
当前 sample 提供的 rcan 三件套的输入是 (h x w) = (1080 x 1920) ，输出是 (h x w) = (2160 x 3840),即 SR4K 。  
edsr 三件套的输入是 (h x w) = (256 x 256) ，输出是 (h x w) = (512 x 512)。

## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/yulunzhang/RCAN)  [modelzoo](http://gitlabdev.vastai.com/VastML/algorithm_modelzoo/-/tree/develop/super_resolution/rcan) |
|  输入 shape |   [ (1,3,1080,1920) ]     |
| INT8量化方式 |   max        |
|  官方精度 | "PSNR":  32.785, "SSIM": 0.776 |
|  VACC FP16  精度 | "PSNR": 32.90 , "SSIM":  0.901 |
|  VACC INT8  精度 |  "PSNR": 32.339 , "SSIM": 0.884   |

## 数据准备

下载模型 rcan-int8-max-1_3_1080_1920-vacc 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 sr4k 到 /opt/vastai/vaststreamx/data/datasets 里




## C++ sample

### super_resulotion 命令行参数说明
```bash
options:
  -m, --model_prefix             model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod])
      --hw_config                hw-config file of the model suite (string [=])
      --vdsp_params              vdsp preprocess parameter file (string [=../data/configs/rcan_bgr888.json])
  -d, --device_id                device id to run (unsigned int [=0])
      --denorm                   denormalization paramsters [mean, std, scale] (string [=[0, 1, 1]])
      --input_file               input image (string [=../data/images/hd_1920x1080.png])
      --output_file              output image (string [=sr_result.jpg])
      --dataset_filelist         input dataset filelist (string [=])
      --dataset_root             input dataset root (string [=])
      --dataset_output_folder    dataset output folder (string [=])
  -?, --help                     print this message
```

### super_resulotion 命令行示例
在build目录里执行

单张照片示例
```bash
./vaststreamx-samples/bin/super_resolution \
-m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
--vdsp_params ../data/configs/rcan_bgr888.json \
--device_id 0 \
--denorm "[0,1,1]" \
--input_file  ../data/images/hd_1920x1080.png \
--output_file sr_result.png
```
结果将保存为 sr_result.png


数据集示例
```bash
mkdir -p sr_output
./vaststreamx-samples/bin/super_resolution \
-m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
--vdsp_params ../data/configs/rcan_bgr888.json \
--device_id 0 \
--denorm "[0,1,1]" \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/sr4k/DIV2K_valid_LR_bicubic_X2_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/sr4k/ \
--dataset_output_folder sr_output
```
结果将保存在 sr_output 目录里

精度测试
```bash
python3 ../evaluation/super_resolution/evaluation.py \
--gt_dir /opt/vastai/vaststreamx/data/datasets/sr4k/DIV2K_valid_HR \
--input_filelist /opt/vastai/vaststreamx/data/datasets/sr4k/DIV2K_valid_LR_bicubic_X2_filelist.txt  \
--output_dir sr_output
```
输出精度
```bash
#最后一行
mean psnr: 32.33915235662851, mean ssim: 0.8842343995396744
```

### sr_prof 命令行参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/rcan_bgr888.json])
  -d, --device_ids      device id to run (string [=[0]])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        instance number for each device (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (int [=20])
      --percentiles     percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=1])
  -?, --help            print this message
```

### sr_prof 运行示例
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/sr_prof \
-m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
--vdsp_params ../data/configs/rcan_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,256,256]" \
--iterations 100 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

#测试最小时延
./vaststreamx-samples/bin/sr_prof \
-m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
--vdsp_params ../data/configs/rcan_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,256,256]" \
--iterations 100 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### sr_prof 运行结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 21.5976
  latency (us):
    avg latency: 135762
    min latency: 74602
    max latency: 167593
    p50 latency: 136609
    p90 latency: 138379
    p95 latency: 138608
    p99 latency: 167530

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 13.1654
  latency (us):
    avg latency: 75954
    min latency: 75426
    max latency: 78427
    p50 latency: 75777
    p90 latency: 76540
    p95 latency: 76788
    p99 latency: 77112
```


## Python sample

### super_resulotion.py 命令行参数说明
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
  --denorm DENORM
                        denormalization parameters [mean, std, scale]
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file
  --dataset_filelist DATASET_FILELIST
                        dataset filelst
  --dataset_root DATASET_ROOT
                        input dataset root
  --dataset_output_folder DATASET_OUTPUT_FOLDER
                        dataset output folder
```
### super_resulotion.py 命令行示例

```bash
#单张图片测试，结果将保存为 sr_result.png
python3 super_resolution.py \
-m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
--vdsp_params ../../data/configs/rcan_bgr888.json \
--device_id 0 \
--denorm "[0,1,1]" \
--input_file ../../data/images/hd_1920x1080.png \
--output_file sr_result.png

#数据集测试，结果保存于 sr_output 文件夹里
mkdir -p sr_output
python3 super_resolution.py \
-m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
--vdsp_params ../../data/configs/rcan_bgr888.json \
--device_id 0 \
--denorm "[0,1,1]" \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/sr4k/DIV2K_valid_LR_bicubic_X2_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/sr4k/ \
--dataset_output_folder sr_output
```

精度测试
```bash
python3 ../../evaluation/super_resolution/evaluation.py \
--gt_dir /opt/vastai/vaststreamx/data/datasets/sr4k/DIV2K_valid_HR \
--input_filelist /opt/vastai/vaststreamx/data/datasets/sr4k/DIV2K_valid_LR_bicubic_X2_filelist.txt  \
--output_dir sr_output
```
输出精度
```bash
#最后一行
mean psnr: 32.33915235662851, mean ssim: 0.8842343995396744
```

## Python sample

### sr_prof.py 命令行参数说明

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


### sr_prof.py 运行示例

在本目录下运行  
```bash
#
# 测试最大吞吐
python3 sr_prof.py \
-m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
--vdsp_params ../../data/configs/rcan_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,256,256]" \
--iterations 100 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 sr_prof.py \
-m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
--vdsp_params ../../data/configs/rcan_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,256,256]" \
--iterations 100 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### sr_prof.py 运行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 21.72
  latency (us):
    avg latency: 136165
    min latency: 77509
    max latency: 165931
    p50 latency: 136693
    p90 latency: 137119
    p95 latency: 137162
    p99 latency: 137487

#测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 13.01
  latency (us):
    avg latency: 76887
    min latency: 73031
    max latency: 78491
    p50 latency: 77129
    p90 latency: 78255
    p95 latency: 78443
    p99 latency: 78485

```
