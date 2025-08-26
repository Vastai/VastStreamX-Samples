# BiSeNet 

本目录提供基于 bisenet 模型的 人脸分割   sample


## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/zllrunning/face-parsing.PyTorch)  [modelzoo](http://gitlabdev.vastai.com/VastML/algorithm_modelzoo/-/tree/develop/segmentation/bisenet) |
|  输入 shape |   [ (1,3,512,512) ]     |
| INT8量化方式 |   kl_divergence         |
|  官方精度 |  "mIOU": 0.744   |
|  VACC FP16  精度 |  "mIOU": 0.733  |
|  VACC INT8  精度 |  "mIOU": 0.7377   |



## 数据准备

下载模型 bisenet-int8-kl_divergence-1_3_512_512-vacc 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 bisenet 到 /opt/vastai/vaststreamx/data/datasets 里

## C++ sample

### bisenet 命令行参数说明
```bash
options:
  -m, --model_prefix             model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod])
      --hw_config                hw-config file of the model suite (string [=])
      --vdsp_params              vdsp preprocess parameter file (string [=../data/configs/bisenet_bgr888.json])
  -d, --device_id                device id to run (unsigned int [=0])
      --input_file               input file (string [=../data/images/face.jpg])
      --output_file              output file (string [=bisenet_result.jpg])
      --dataset_filelist         input dataset filelist (string [=])
      --dataset_root             dataset root (string [=])
      --dataset_output_folder    dataset output folder (string [=])
  -?, --help                     print this message
```


### bisenet 运行示例
在build目录里执行

单张人脸照片分割示例
```bash
./vaststreamx-samples/bin/bisenet \
--model_prefix /opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod \
--vdsp_params ../data/configs/bisenet_bgr888.json \
--device_id 0 \
--input_file ../data/images/face.jpg \
--output_file bisenet_result.jpg 
```
人脸分割结果将展示在 bisenet_result.jpg 里

人脸数据集示例
```bash
mkdir -p bisenet_output
./vaststreamx-samples/bin/bisenet \
--model_prefix /opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod \
--vdsp_params ../data/configs/bisenet_bgr888.json \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/bisenet/bisegnet_test_img_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/bisenet/ \
--dataset_output_folder ./bisenet_output
#结果将保存在 bisenet_output 目录


# 统计精度
python3 ../evaluation/bisenet/zllrunning_vamp_eval.py \
--src_dir /opt/vastai/vaststreamx/data/datasets/bisenet/bisegnet_test_img \
--gt_dir /opt/vastai/vaststreamx/data/datasets/bisenet/bisegnet_test_mask \
--input_npz_path /opt/vastai/vaststreamx/data/datasets/bisenet/bisegnet_test_img_filelist.txt \
--out_npz_dir ./bisenet_output \
--input_shape 512 512 \
--vamp_flag
```

统计精度结果示例
```bash
----------------- Total Performance --------------------
Overall Acc:     0.9561228425633737
Mean Acc :       0.8350860147133995
FreqW Acc :      0.9174207989307908
Mean IoU :       0.744108556563433
Overall F1:      0.842063201903483
----------------- Class IoU Performance ----------------
background      : 0.9384490212744376
skin    : 0.9286972981269527
nose    : 0.6125526265047803
eyeglass        : 0.6047313080934851
left_eye        : 0.6506479371850342
right_eye       : 0.6431100473011616
left_brow       : 0.8365007166025495
right_brow      : 0.6695612151061148
left_ear        : 0.6559966087000867
right_ear       : 0.42227149051906654
mouth   : 0.8716355466964373
upper_lip       : 0.8360681048965977
lower_lip       : 0.7626009735382941
hair    : 0.799124760847228
hat     : 0.8631400202928009
earring : 0.362178884825004
necklace        : 0.8412048354353484
neck    : 0.9303956158261196
cloth   : 0.9091955629337264
--------------------------------------------------------
```

### bisenet_prof 命令行参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/bisenet_bgr888.json])
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
### bisenet_prof 运行示例
在build目录下执行
```bash
#测试最大吞吐
./vaststreamx-samples/bin/bisenet_prof  \
--model_prefix /opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod \
--vdsp_params ../data/configs/bisenet_bgr888.json \
--device_ids [0] \
--batch_size 2 \
--instance 1 \
--shape "[3,512,512]" \
--iterations 2000 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1


#测试最小时延
./vaststreamx-samples/bin/bisenet_prof  \
--model_prefix /opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod \
--vdsp_params ../data/configs/bisenet_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,512,512]" \
--iterations 1000 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0

```
### bisenet_prof 运行结果示例
```bash
#测试最大吞吐
- number of instances: 2
  devices: [ 0 ]
  queue size: 1
  batch size: 2
  throughput (qps): 727.665
  latency (us):
    avg latency: 16407
    min latency: 13138
    max latency: 29803
    p50 latency: 16364
    p90 latency: 17108
    p95 latency: 17364
    p99 latency: 18455


#测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 201.538
  latency (us):
    avg latency: 4960
    min latency: 4864
    max latency: 7498
    p50 latency: 4940
    p90 latency: 5007
    p95 latency: 5025
    p99 latency: 5092
```

## Python sample 功能测试

### bisenet.py 命令行参数说明
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
                        dataset root
  --dataset_output_folder DATASET_OUTPUT_FOLDER
                        dataset output folder

```

### bisenet.py 运行示例

在本目录下运行  
```bash
#单张照片示例，结果将展示在 bisenet_result.jpg 里
python3 bisenet.py \
--model_prefix /opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod \
--vdsp_params ../../../data/configs/bisenet_bgr888.json \
--device_id 0 \
--input_file ../../../data/images/face.jpg \
--output_file bisenet_result.jpg


#测试数据集,结果保存在 bisenet_output
mkdir -p bisenet_output
python3 bisenet.py \
--model_prefix /opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod \
--vdsp_params ../../../data/configs/bisenet_bgr888.json \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/bisenet/bisegnet_test_img_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/bisenet/ \
--dataset_output_folder ./bisenet_output

# 统计精度
python3 ../../../evaluation/bisenet/zllrunning_vamp_eval.py \
--src_dir /opt/vastai/vaststreamx/data/datasets/bisenet/bisegnet_test_img \
--gt_dir /opt/vastai/vaststreamx/data/datasets/bisenet/bisegnet_test_mask \
--input_npz_path /opt/vastai/vaststreamx/data/datasets/bisenet/bisegnet_test_img_filelist.txt \
--out_npz_dir ./bisenet_output \
--input_shape 512 512 \
--vamp_flag
```

统计精度结果示例
```bash
----------------- Total Performance --------------------
Overall Acc:     0.9561228425633737
Mean Acc :       0.8350860147133995
FreqW Acc :      0.9174207989307908
Mean IoU :       0.744108556563433
Overall F1:      0.842063201903483
----------------- Class IoU Performance ----------------
background      : 0.9384490212744376
skin    : 0.9286972981269527
nose    : 0.6125526265047803
eyeglass        : 0.6047313080934851
left_eye        : 0.6506479371850342
right_eye       : 0.6431100473011616
left_brow       : 0.8365007166025495
right_brow      : 0.6695612151061148
left_ear        : 0.6559966087000867
right_ear       : 0.42227149051906654
mouth   : 0.8716355466964373
upper_lip       : 0.8360681048965977
lower_lip       : 0.7626009735382941
hair    : 0.799124760847228
hat     : 0.8631400202928009
earring : 0.362178884825004
necklace        : 0.8412048354353484
neck    : 0.9303956158261196
cloth   : 0.9091955629337264
--------------------------------------------------------
```


## Python sample 性能测试

### bisenet_prof.py 命令行参数说明
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


### bisenet_prof.py 运行示例

在本目录下运行  
```bash
#测试最大吞吐
python3 bisenet_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod \
--vdsp_params ../../../data/configs/bisenet_bgr888.json \
--device_ids [0] \
--batch_size 2 \
--instance 2 \
--shape "[3,512,512]" \
--iterations 2000 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

#测试最小时延
python3 bisenet_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/models/bisenet-int8-kl_divergence-1_3_512_512-vacc/mod \
--vdsp_params ../../../data/configs/bisenet_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,512,512]" \
--iterations 1000 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```


### bisenet_prof.py 运行结果示例

```bash
#测试最大吞吐
- number of instances: 2
  devices: [0]
  queue size: 1
  batch size: 2
  throughput (qps): 661.38
  latency (us):
    avg latency: 17952
    min latency: 12128
    max latency: 32663
    p50 latency: 17629
    p90 latency: 21279
    p95 latency: 22577
    p99 latency: 26036

#测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 178.64
  latency (us):
    avg latency: 5596
    min latency: 5519
    max latency: 8249
    p50 latency: 5586
    p90 latency: 5644
    p95 latency: 5660
    p99 latency: 5817
```
