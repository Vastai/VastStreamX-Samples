# ELIC Sample

本目录提供基于 elic 模型的图像压缩与解压sample 

## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/VincentChandelier/ELiC-ReImplemetation)  [modelzoo](-) |
|  输入 shape |   [ (1,3,512,512) - (1,3,1280,2048)]     |
| INT8量化方式 |   -          |
|  官方精度 |  "PSNR":   37.8   |
|  VACC FP16  精度 | "PSNR":   37.7    |
|  VACC INT8  精度 | - |


## 数据集准备

下载 elic 模型 elic.tar.gz 到 /opt/vastai/vaststreamx/data/models 里
下载 图片集 Kodak-512.tar.gz Kodak.tar.gz kodak_1280_2048.tar.gz 到 /opt/vastai/vaststreamx/data/datasets 里

## Python Sample

依赖: compressai==1.1.5

### elic_inference.py 脚本命令行说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  --gaha_model_prefix GAHA_MODEL_PREFIX
                        model prefix of the model suite files
  --gaha_hw_config GAHA_HW_CONFIG
                        hw-config file of the model suite
  --gaha_vdsp_params GAHA_VDSP_PARAMS
                        vdsp preprocess parameter file
  --hs_model_prefix HS_MODEL_PREFIX
                        h_s model prefix of the model suite files
  --hs_hw_config HS_HW_CONFIG
                        hs_hw-config file of the model suite
  --gs_model_prefix GS_MODEL_PREFIX
                        g_s model prefix of the model suite files
  --gs_hw_config GS_HW_CONFIG
                        gs_hw-config file of the model suite
  --torch_model TORCH_MODEL
                        torch model file
  --tensorize_elf_path TENSORIZE_ELF_PATH
                        tensorize elf file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file
  --dataset_path DATASET_PATH
                        input dataset path
  --dataset_output_path DATASET_OUTPUT_PATH
                        dataset output path
  --patch PATCH         padding patch size (default: 256)
```

### elic_inference.py 脚本命令示例

```bash
#测试单张图片
python3 elic_inference.py  \
--gaha_model_prefix  /opt/vastai/vaststreamx/data/models/elic/gaha-fp16-none-1_3_512_512-vacc/mod \
--gaha_vdsp_params  ../../../data/configs/elic_compress_gaha_rgbplanar.json \
--hs_model_prefix  /opt/vastai/vaststreamx/data/models/elic/hs_chunk-fp16-none-1_192_8_8/mod \
--gs_model_prefix   /opt/vastai/vaststreamx/data/models/elic/gs-fp16-none-1_320_32_32/mod \
--torch_model  /opt/vastai/vaststreamx/data/pre-trained/ELIC_0450_ft_3980_Plateau.pth.tar \
--tensorize_elf_path /opt/vastai/vaststreamx/data/elf/tensorize_ext_op \
--device_id  0 \
--input_file  /opt/vastai/vaststreamx/data/datasets/Kodak-512/kodim01.png \
--output_file  elic_result.png
#结果保存为 elic_result.png


# 测试数据集
python3 elic_inference.py  \
--gaha_model_prefix  /opt/vastai/vaststreamx/data/models/elic/gaha-fp16-none-1_3_512_512-vacc/mod \
--gaha_vdsp_params  ../../../data/configs/elic_compress_gaha_rgbplanar.json \
--hs_model_prefix  /opt/vastai/vaststreamx/data/models/elic/hs_chunk-fp16-none-1_192_8_8/mod \
--gs_model_prefix /opt/vastai/vaststreamx/data/models/elic/gs-fp16-none-1_320_32_32/mod \
--torch_model  /opt/vastai/vaststreamx/data/pre-trained/ELIC_0450_ft_3980_Plateau.pth.tar \
--tensorize_elf_path /opt/vastai/vaststreamx/data/elf/tensorize_ext_op \
--device_id  0 \
--dataset_path /opt/vastai/vaststreamx/data/datasets/Kodak-512/ \
--dataset_output_path dataset_outputs

#结果保存到 dataset_outputs 文件夹，并输出PSNR。压缩与解压时间受CPU影响较大，重点关注psnr值
    Ave Compress time:317.88142522176105 ms
    Ave Decompress time:233.75619451204935 ms
    Ave PNSR:37.708291421801384

```




### dynamic_elic_inference.py 脚本命令行说明 (注意dynamic 最大支持输入图片分辨率 1024x1024)

```bash
optional arguments:
  -h, --help            show this help message and exit
  --gaha_model_info GAHA_MODEL_INFO
                        model prefix of the model suite files
  --gaha_hw_config GAHA_HW_CONFIG
                        hw-config file of the model suite
  --gaha_vdsp_params GAHA_VDSP_PARAMS
                        vdsp preprocess parameter file
  --max_input_shape MAX_INPUT_SHAPE
                        model max input shape, max supported shape: [1,3,1024,1024]
  --hs_model_info HS_MODEL_INFO
                        h_s model prefix of the model suite files
  --hs_hw_config HS_HW_CONFIG
                        hs_hw-config file of the model suite
  --gs0_model_info GS0_MODEL_INFO
                        g_s model prefix of the model suite files
  --gs_model_info GS_MODEL_INFO
                        g_s model prefix of the model suite files
  --gs0_hw_config GS0_HW_CONFIG
                        gs0_hw-config file of the model suite
  --gs_hw_config GS_HW_CONFIG
                        gs_hw-config file of the model suite
  --torch_model TORCH_MODEL
                        torch model file
  --tensorize_elf_path TENSORIZE_ELF_PATH
                        tensorize elf file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file
  --dataset_path DATASET_PATH
                        input dataset path
  --dataset_output_path DATASET_OUTPUT_PATH
                        dataset output path
  --patch PATCH         padding patch size (default: 256)
```

### dynamic_elic_inference.py 脚本命令示例

```bash
#测试单张图片
python3 dynamic_elic_inference.py  \
--gaha_model_info /opt/vastai/vaststreamx/data/models/elic/gaha-dynamic/gaha-dynamic_module_info.json \
--gaha_vdsp_params  ../../../data/configs/elic_compress_gaha_rgbplanar.json \
--hs_model_info /opt/vastai/vaststreamx/data/models/elic/hs_chunk-dynamic/hs_chunk-dynamic_module_info.json \
--gs0_model_info /opt/vastai/vaststreamx/data/models/elic/gs0-dynamic/gs0-dynamic_module_info.json \
--gs_model_info /opt/vastai/vaststreamx/data/models/elic/gs-dynamic/gs-dynamic_module_info.json \
--torch_model  /opt/vastai/vaststreamx/data/pre-trained/ELIC_0450_ft_3980_Plateau.pth.tar \
--tensorize_elf_path /opt/vastai/vaststreamx/data/elf/tensorize_ext_op \
--device_id  0 \
--input_file  /opt/vastai/vaststreamx/data/datasets/Kodak/kodim01.png \
--output_file  elic_dynamic_result.png
#结果保存为 elic_dynamic_result.png


# 测试数据集
python3 dynamic_elic_inference.py  \
--gaha_model_info /opt/vastai/vaststreamx/data/models/elic/gaha-dynamic/gaha-dynamic_module_info.json \
--gaha_vdsp_params  ../../../data/configs/elic_compress_gaha_rgbplanar.json \
--hs_model_info /opt/vastai/vaststreamx/data/models/elic/hs_chunk-dynamic/hs_chunk-dynamic_module_info.json \
--gs0_model_info /opt/vastai/vaststreamx/data/models/elic/gs0-dynamic/gs0-dynamic_module_info.json \
--gs_model_info /opt/vastai/vaststreamx/data/models/elic/gs-dynamic/gs-dynamic_module_info.json \
--torch_model  /opt/vastai/vaststreamx/data/pre-trained/ELIC_0450_ft_3980_Plateau.pth.tar \
--tensorize_elf_path /opt/vastai/vaststreamx/data/elf/tensorize_ext_op \
--device_id  0 \
--dataset_path /opt/vastai/vaststreamx/data/datasets/Kodak/ \
--dataset_output_path dataset_outputs

#结果保存到 dataset_outputs 文件夹，并输出PSNR。压缩与解压时间受CPU影响较大，重点关注psnr值
    Ave Compress time:404.6524564425151 ms
    Ave Decompress time:297.8130678335826 ms
    Ave PNSR:37.521121727924076
```


### elic_no_entropy_inference.py 脚本命令行说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  --model_prefix ELIC_NOENTROPY_MODEL_PREFIX
                        model prefix of the model suite files
  --hw_config ELIC_NOENTROPY_HW_CONFIG
                        hw-config file of the model suite
  --vdsp_params ELIC_NOENTROPY_VDSP_PARAMS
                        vdsp preprocess parameter file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file
  --dataset_path DATASET_PATH
                        input dataset path
  --dataset_output_path DATASET_OUTPUT_PATH
                        dataset output path
  --patch PATCH         padding patch size (default: 256)
```

### elic_no_entropy_inference.py 脚本命令示例

```bash
#测试单张图片
##512x512 模型
python3 elic_no_entropy_inference.py  \
--model_prefix /opt/vastai/vaststreamx/data/models/elic/elic_no_entropy-512_512/mod \
--vdsp_params  ../../../data/configs/elic_compress_gaha_rgbplanar.json \
--device_id  0 \
--input_file  /opt/vastai/vaststreamx/data/datasets/Kodak-512/kodim01.png \
--output_file  elic_no_entropy_512x512.png
#结果保存为 elic_no_entropy_512x512.png


##1280x2048 模型
python3 elic_no_entropy_inference.py  \
--model_prefix /opt/vastai/vaststreamx/data/models/elic/elic_no_entropy-1280_2048/mod \
--vdsp_params  ../../../data/configs/elic_compress_gaha_rgbplanar.json \
--device_id  0 \
--input_file  /opt/vastai/vaststreamx/data/datasets/Kodak_1280_2048/kodim01.png \
--output_file  elic_no_entropy_1280x2048.png
#结果保存为 elic_no_entropy_1280x2048.png

# 测试数据集

##512x512 模型
python3 elic_no_entropy_inference.py  \
--model_prefix /opt/vastai/vaststreamx/data/models/elic/elic_no_entropy-512_512/mod \
--vdsp_params  ../../../data/configs/elic_compress_gaha_rgbplanar.json \
--device_id  0 \
--dataset_path /opt/vastai/vaststreamx/data/datasets/Kodak-512/ \
--dataset_output_path dataset_outputs_512x512

#结果保存到 dataset_outputs_512x512 文件夹，并输出PSNR
    Ave Compress time:107.34399159749348 ms
    Ave PNSR:37.706234893066075
    Ave bbp:0.843717660754919

##1280x2048 模型
python3 elic_no_entropy_inference.py  \
--model_prefix /opt/vastai/vaststreamx/data/models/elic/elic_no_entropy-1280_2048/mod \
--vdsp_params  ../../../data/configs/elic_compress_gaha_rgbplanar.json \
--device_id  0 \
--dataset_path /opt/vastai/vaststreamx/data/datasets/Kodak_1280_2048/ \
--dataset_output_path dataset_outputs_1280x2048

#结果保存到 dataset_outputs_1280x2048 文件夹，并输出PSNR。压缩与解压时间受CPU影响较大，重点关注psnr值
    Ave Compress time:1303.3234576384227 ms
    Ave PNSR:43.009992020884276
    Ave bbp:0.262036203717192  
```


### elic_noentropy_prof.py 脚本命令行说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  --hw_config HW_CONFIG
                        hw-config file of the model suite
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
  --patch PATCH         padding patch size (default: 256)
  
```

### elic_noentropy_prof.py 脚本命令示例

```bash

# 测试最大吞吐
##512x512 模型
python ./elic_noentropy_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/models/elic/elic_no_entropy-512_512/mod \
--vdsp_params  ../../../data/configs/elic_compress_gaha_rgbplanar.json \
--device_ids [0] \
--instance 1 \
--iterations 100 \
--queue_size 1 \
--batch_size 1

##1280x2048 模型
python ./elic_noentropy_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/models/elic/elic_no_entropy-1280_2048/mod \
--vdsp_params  ../../../data/configs/elic_compress_gaha_rgbplanar.json \
--device_ids [0] \
--instance 1 \
--iterations 10 \
--batch_size 1  \
--queue_size 1

# 测试最小时延
##512x512 模型
python ./elic_noentropy_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/models/elic/elic_no_entropy-512_512/mod \
--vdsp_params  ../../../data/configs/elic_compress_gaha_rgbplanar.json \
--device_ids [0] \
--instance 1 \
--iterations 100 \
--batch_size 1  \
--queue_size 0

##1280x2048 模型
python ./elic_noentropy_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/models/elic/elic_no_entropy-1280_2048/mod \
--vdsp_params  ../../../data/configs/elic_compress_gaha_rgbplanar.json \
--device_ids [0] \
--instance 1 \
--iterations 10 \
--batch_size 1  \
--queue_size 0
```


### elic_noentropy_prof.py 运行结果示例
```bash
# 测试最大吞吐
##512x512 模型
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 2
  throughput (qps): 9.65
  latency (us):
    avg latency: 207147
    min latency: 204326
    max latency: 210392
    p50 latency: 206909
    p90 latency: 208871
    p95 latency: 209142
    p99 latency: 209844
   
##1280x2048 模型 
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 0.82
  latency (us):
    avg latency: 3310918
    min latency: 1334366
    max latency: 3736934
    p50 latency: 3599276
    p90 latency: 3614015
    p95 latency: 3675474
    p99 latency: 3724642

    
# 测试最小时延
##512x512 模型
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 9.32
  latency (us):
    avg latency: 107344
    min latency: 104584
    max latency: 114090
    p50 latency: 107284
    p90 latency: 108189
    p95 latency: 108360
    p99 latency: 110198

##1280x2048 模型
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 0.76
  latency (us):
    avg latency: 1312423
    min latency: 1306063
    max latency: 1335950
    p50 latency: 1308474
    p90 latency: 1325153
    p95 latency: 1330551
    p99 latency: 1334871
```

