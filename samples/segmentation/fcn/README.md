# FCN Model Sample

本例程提供基于 FCN 模型的图像分割 sample 

## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/core/models/fcn.py)  [modelzoo](https://github.com/Vastai/VastModelZOO/tree/main/cv/segmentation/fcn) |
|  输入 shape |   [ (1,3,320,320) ]     |
| INT8量化方式 |   kl_divergence         |
|  官方精度 |  "mIOU": 53.339   |
|  VACC FP16  精度 |  "mIOU": 52.990  |
|  VACC INT8  精度 |  "mIOU": 52.989   |


## 数据准备

下载模型 fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 segmentation 到 /opt/vastai/vaststreamx/data/datasets 里


## C++ Sample

### fcn 命令行参数说明
```bash
options:
  -m, --model_prefix             model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod])
      --hw_config                hw-config file of the model suite (string [=])
      --vdsp_params              vdsp preprocess parameter file (string [=../data/configs/fcn_bgr888.json])
  -d, --device_id                device id to run (unsigned int [=0])
      --input_file               input file (string [=../data/images/dog.jpg])
      --output_file              output file (string [=fcn_result.jpg])
      --dataset_filelist         input dataset file list (string [=])
      --dataset_root             dataset root (string [=])
      --dataset_output_folder    dataset output folder (string [=])
  -?, --help                     print this message
```
### fcn 命令行示例
在build目录下执行   

单张图片示例，结果保存为 fcn_result.jpg
```bash
./vaststreamx-samples/bin/fcn \
-m /opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod \
--vdsp_params ../data/configs/fcn_bgr888.json \
--device_id 0 \
--input_file ../data/images/cycling.jpg \
--output_file fcn_result.jpg
```


数据集测试示例
```bash
mkdir -p fcn_out
./vaststreamx-samples/bin/fcn \
-m /opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod \
--vdsp_params ../data/configs/fcn_bgr888.json \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/segmentation/JPEGImages_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/segmentation/ \
--dataset_output_folder fcn_out


# 统计精度
python3 ../evaluation/fcn/awesome_vamp_eval.py \
--src_dir /opt/vastai/vaststreamx/data/datasets/segmentation/JPEGImages_val \
--gt_dir /opt/vastai/vaststreamx/data/datasets/segmentation/SegmentationClass \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/segmentation/JPEGImages_val_filelist.txt \
--out_npz_dir ./fcn_out \
--input_shape 320 320 \
--vamp_flag
```
精度结果示例
```bash
看最后一行
validation pixAcc: 88.257, mIoU: 52.952
```


### fcn_prof 命令行参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/fcn_bgr888.json])
  -d, --device_ids      device id to run (string [=[0]])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        instance number for each device (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (int [=10240])
      --percentiles     percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=2])
  -?, --help            print this message
```

### fcn_prof 命令行示例
在build目录下执行   
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/fcn_prof  \
-m /opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod \
--vdsp_params ../data/configs/fcn_bgr888.json \
--device_ids [0] \
--batch_size 6 \
--instance 1 \
--shape "[3,320,320]" \
--iterations 600 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/fcn_prof  \
-m /opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod \
--vdsp_params ../data/configs/fcn_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,320,320]" \
--iterations 1300 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### fcn_prof 命令行结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 6
  throughput (qps): 662.88
  latency (us):
    avg latency: 27045
    min latency: 23957
    max latency: 45346
    p50 latency: 26798
    p90 latency: 28591
    p95 latency: 28855
    p99 latency: 29389

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 261.488
  latency (us):
    avg latency: 3823
    min latency: 3751
    max latency: 5196
    p50 latency: 3828
    p90 latency: 3852
    p95 latency: 3888
    p99 latency: 4052
```

## Python Sample

### fcn.py 命令行参数说明
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
                        dataset image file list
  --dataset_root DATASET_ROOT
                        dataset root
  --dataset_output_folder DATASET_OUTPUT_FOLDER
                        dataset output folder
```
### fcn.py 命令行示例
在当前目录执行
```bash
python3 fcn.py \
-m /opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod \
--vdsp_params ../../../data/configs/fcn_bgr888.json \
--device_id 0 \
--input_file ../../../data/images/cycling.jpg \
--output_file fcn_result.jpg
```
结果保存到  fcn_result.jpg


测试数据集
```bash 
mkdir -p fcn_out
python3 fcn.py \
-m /opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod \
--vdsp_params ../../../data/configs/fcn_bgr888.json \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/segmentation/JPEGImages_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/segmentation/ \
--dataset_output_folder fcn_out

# 统计精度
python3 ../../../evaluation/fcn/awesome_vamp_eval.py \
--src_dir /opt/vastai/vaststreamx/data/datasets/segmentation/JPEGImages_val \
--gt_dir /opt/vastai/vaststreamx/data/datasets/segmentation/SegmentationClass \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/segmentation/JPEGImages_val_filelist.txt \
--out_npz_dir ./fcn_out \
--input_shape 320 320 \
--vamp_flag
```
精度结果示例
```bash
看最后一行
validation pixAcc: 88.257, mIoU: 52.952
```





### fcn_prof.py 命令行参数说明
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

### fcn_prof.py 命令行示例
在当前目录执行
```bash
# 测试最大吞吐
python3 fcn_prof.py \
-m /opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod \
--vdsp_params ../../../data/configs/fcn_bgr888.json \
--device_ids [0] \
--batch_size 6 \
--instance 1 \
--shape "[3,320,320]" \
--iterations 600 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 fcn_prof.py \
-m /opt/vastai/vaststreamx/data/models/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc/mod \
--vdsp_params ../../../data/configs/fcn_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,320,320]" \
--iterations 1300 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0

```

### fcn_prof.py 命令行结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 6
  throughput (qps): 633.16
  latency (us):
    avg latency: 28343
    min latency: 21853
    max latency: 44837
    p50 latency: 28422
    p90 latency: 31900
    p95 latency: 32908
    p99 latency: 34044

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 281.65
  latency (us):
    avg latency: 3549
    min latency: 3474
    max latency: 5232
    p50 latency: 3546
    p90 latency: 3562
    p95 latency: 3568
    p99 latency: 3608
```