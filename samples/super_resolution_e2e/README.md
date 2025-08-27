# Super Resolution E2D sample

本 sample 与 Super Resolution sample 的区别在与 本sample后处理使用了vdsp 算子

本目录提供基于 rcan, edsr 模型的 图像超分 sample。  
当前 sample 提供的 rcan 三件套的输入是 (h x w) = (1080 x 1920) ，输出是 (h x w) = (2160 x 3840),即 SR4K 。  
edsr 三件套的输入是 (h x w) = (256 x 256) ，输出是 (h x w) = (512 x 512)。

## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/yulunzhang/RCAN)  [modelzoo](https://github.com/Vastai/VastModelZOO/tree/main/cv/super_resolution/rcan) |
|  输入 shape |   [ (1,3,1080,1920) ]     |
| INT8量化方式 |   max        |
|  官方精度 | "PSNR":  32.785, "SSIM": 0.776 |
|  VACC FP16  精度 | "PSNR": 32.90 , "SSIM":  0.901 |
|  VACC INT8  精度 |  "PSNR": 32.339 , "SSIM": 0.884   |

## 数据准备

下载模型 rcan-int8-max-1_3_1080_1920-vacc 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 sr4k 到 /opt/vastai/vaststreamx/data/datasets 里




## C++ sample

### super_resolution_e2e 命令行参数说明
```bash
options:
  -m, --model_prefix             model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod])
      --hw_config                hw-config file of the model suite (string [=])
      --vdsp_params              vdsp preprocess parameter file (string [=../data/configs/rcan_bgr888.json])
  -d, --device_id                device id to run (unsigned int [=0])
      --postproc_elf             post process elf file (string [=/opt/vastai/vaststreamx/data/elf/postprocessimage])
      --denorm                   denormalization paramsters [mean, std, scale] (string [=[0, 1, 1]])
      --input_file               input image (string [=../data/images/hd_1920x1080.png])
      --output_file              output image (string [=sr_result.jpg])
      --dataset_filelist         input dataset filelist (string [=])
      --dataset_root             input dataset root (string [=])
      --dataset_output_folder    dataset output folder (string [=])
  -?, --help                     print this message
```

### super_resolution_e2e 命令行示例
在build目录里执行

单张照片示例
```bash
./vaststreamx-samples/bin/super_resolution_e2e \
-m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
--vdsp_params ../data/configs/rcan_bgr888.json \
--device_id 0 \
--postproc_elf /opt/vastai/vaststreamx/data/elf/postprocessimage \
--denorm "[0,1,1]" \
--input_file  ../data/images/hd_1920x1080.png \
--output_file sr_result.png
```
结果将保存为 sr_result.png


数据集示例
```bash
mkdir -p sr_output
./vaststreamx-samples/bin/super_resolution_e2e \
-m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
--vdsp_params ../data/configs/rcan_bgr888.json \
--device_id 0 \
--postproc_elf /opt/vastai/vaststreamx/data/elf/postprocessimage \
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
mean psnr: 32.278302721721005, mean ssim: 0.8838268926851219
```

### sr_e2e_prof 命令行参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/rcan_bgr888.json])
  -d, --device_ids      device id to run (string [=[0]])
      --postproc_elf    post process elf file (string [=/opt/vastai/vaststreamx/data/elf/postprocessimage])
      --denorm          denormalization paramsters [mean, std, scale] (string [=[0, 1, 1]])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        instance number for each device (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (int [=20])
      --percentiles     percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=1])
  -?, --help            print this message
```

### sr_e2e_prof 运行示例
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/sr_e2e_prof \
-m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
--vdsp_params ../data/configs/rcan_bgr888.json \
--postproc_elf /opt/vastai/vaststreamx/data/elf/postprocessimage \
--denorm "[0,1,1]" \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,256,256]" \
--iterations 100 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

#测试最小时延
./vaststreamx-samples/bin/sr_e2e_prof \
-m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
--vdsp_params ../data/configs/rcan_bgr888.json \
--postproc_elf /opt/vastai/vaststreamx/data/elf/postprocessimage \
--denorm "[0,1,1]" \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,256,256]" \
--iterations 100 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### sr_e2e_prof 运行结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 23.4213
  latency (us):
    avg latency: 126325
    min latency: 66550
    max latency: 150669
    p50 latency: 127494
    p90 latency: 128751
    p95 latency: 129075
    p99 latency: 129716

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 16.3081
  latency (us):
    avg latency: 61317
    min latency: 60991
    max latency: 66121
    p50 latency: 61157
    p90 latency: 61613
    p95 latency: 62054
    p99 latency: 64311
```


## Python sample

### super_e2e_resulotion.py 命令行参数说明
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
  --postproc_elf POSTPROC_ELF
  --denorm DENORM       denormalization params [mean, std, scale]
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
### super_e2e_resulotion.py 命令行示例

```bash
#单张图片测试，结果将保存为 sr_result.png
python3 super_resolution_e2e.py \
-m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
--vdsp_params ../../data/configs/rcan_bgr888.json \
--postproc_elf /opt/vastai/vaststreamx/data/elf/postprocessimage \
--device_id 0 \
--denorm "[0,1,1]" \
--input_file ../../data/images/hd_1920x1080.png \
--output_file sr_result.png

#数据集测试，结果保存于 sr_output 文件夹里
mkdir -p sr_output
python3 super_resolution_e2e.py \
-m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
--vdsp_params ../../data/configs/rcan_bgr888.json \
--device_id 0 \
--postproc_elf /opt/vastai/vaststreamx/data/elf/postprocessimage \
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
mean psnr: 32.278302721721005, mean ssim: 0.8838268926851219
```

## Python sample

### sr_e2e_prof.py 命令行参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  --postproc_elf POSTPROC_ELF
  --denorm DENORM       denormalization params [mean, std, scale]
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


### sr_e2e_prof.py 运行示例

在本目录下运行  
```bash
#
# 测试最大吞吐
python3 sr_e2e_prof.py \
-m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
--vdsp_params ../../data/configs/rcan_bgr888.json \
--postproc_elf /opt/vastai/vaststreamx/data/elf/postprocessimage \
--denorm "[0,1,1]" \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,256,256]" \
--iterations 100 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 sr_e2e_prof.py \
-m /opt/vastai/vaststreamx/data/models/rcan-int8-max-1_3_1080_1920-vacc/mod \
--vdsp_params ../../data/configs/rcan_bgr888.json \
--postproc_elf /opt/vastai/vaststreamx/data/elf/postprocessimage \
--denorm "[0,1,1]" \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,256,256]" \
--iterations 100 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### sr_e2e_prof.py 运行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 23.39
  latency (us):
    avg latency: 126944
    min latency: 67391
    max latency: 150057
    p50 latency: 127662
    p90 latency: 127849
    p95 latency: 128036
    p99 latency: 128329

#测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 16.78
  latency (us):
    avg latency: 59595
    min latency: 59268
    max latency: 64405
    p50 latency: 59411
    p90 latency: 59803
    p95 latency: 60550
    p99 latency: 63271
```
