# Swin Transformer Sample

本目录提供基于 movile_vit 模型的 image classification sample   
本Sample 同样支持 swin-transformer 模型，只需将命令行的 movile_vit 模型替换为 swin-transformer 模型, vdsp json文件替换为 swin-transformer 对应的 vdsp json文件即可
## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/microsoft/Swin-Transformer)  [modelzoo](http://gitlabdev.vastai.com/Algorithm/algorithm_modelzoo/-/tree/develop/classification/swin_transformer) |
|  输入 shape |   [ (1,3,224,224) ]     |
| INT8量化方式 |   -          |
|  官方精度 |  top1_rate:83.2    top5_rate:96.2         |
|  VACC FP16  精度 | top1_rate: 81.846 top5_rate: 95.442  |
|  VACC INT8  精度 |  -    |


## 数据准备

下载模型 swin-b-fp16-none-1_3_224_224-vacc 到 /opt/vastai/vaststreamx/data/models 里  
下载数据集 ILSVRC2012_img_val 到 /opt/vastai/vaststreamx/data/datasets 里  

需要 odsp_plugin： 
下载 odsp_plugin***.tar.gz,然后解压，编译，配置LD_LIBRARY_PATH
```bash
mkdir -p /opt/vastai/odsp_plugin
tar xf odsp_plugin***.tar.gz -C /opt/vastai/odsp_plugin
cd /opt/vastai/odsp_plugin/vastai
bash build.sh
export LD_LIBRARY_PATH=/opt/vastai/odsp_plugin/vastai/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/vastai/odsp_plugin/protobuf/lib:$LD_LIBRARY_PATH
```

## C++ sample

### mobilevit 命令行参数
```bash
options:
  -m, --model_prefix           model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/mobilevit_s-fp16-none-1_3_224_224-vacc/mod])
      --hw_config              hw-config file of the model suite (string [=])
      --vdsp_params            vdsp preprocess parameter file (string [=../data/configs/mobilevit_rgbplanar.json])
  -d, --device_id              device id to run (unsigned int [=0])
      --label_file             label file (string [=../data/labels/imagenet.txt])
      --input_file             input file (string [=../data/images/cat.jpg])
      --dataset_filelist       input dataset filelist (string [=])
      --dataset_root           input dataset root (string [=])
      --dataset_output_file    dataset output file (string [=])
  -?, --help                   print this message
```

### mobilevit 运行示例

在 build 目录里执行   
```bash
#单张图片示例
./vaststreamx-samples/bin/mobilevit \
-m /opt/vastai/vaststreamx/data/models/swin-b-fp16-none-1_3_224_224-vacc/mod \
--vdsp_params ../data/configs/swin_rgbplanar.json \
--device_id 0 \
--label_file ../data/labels/imagenet.txt \
--input_file ../data/images/cat.jpg 


#数据集示例
./vaststreamx-samples/bin/mobilevit \
-m /opt/vastai/vaststreamx/data/models/swin-b-fp16-none-1_3_224_224-vacc/mod \
--vdsp_params ../data/configs/swin_rgbplanar.json \
--device_id 0 \
--label_file ../data/labels/imagenet.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ILSVRC2012_img_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file swin_result.txt

#统计精度
python3 ../evaluation/classification/eval_topk.py  swin_result.txt  
```

### mobilevit 运行结果示例

```bash
#单张图片结果示例
Top5:
0th, score: 7.957, class name: Egyptian cat
1th, score: 6.98, class name: tabby, tabby cat
2th, score: 6.691, class name: tiger cat
3th, score: 6.191, class name: weasel
4th, score: 4.848, class name: lynx, catamount

#精度统计结果
[VACC]:  top1_rate: 81.846 top5_rate: 95.442
```


### mobilevit_prof 命令行参数
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/mobilevit_s-fp16-none-1_3_224_224-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/mobilevit_rgbplanar.json])
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

### mobilevit_prof 运行示例

在 build 目录里   
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/mobilevit_prof \
-m /opt/vastai/vaststreamx/data/models/swin-b-fp16-none-1_3_224_224-vacc/mod \
--vdsp_params ../data/configs/swin_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 200 \
--shape "[3,224,224]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/mobilevit_prof \
-m /opt/vastai/vaststreamx/data/models/swin-b-fp16-none-1_3_224_224-vacc/mod \
--vdsp_params ../data/configs/swin_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 200 \
--shape "[3,224,224]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### mobilevit_prof 运行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 29.3533
  latency (us):
    avg latency: 101688
    min latency: 36862
    max latency: 104828
    p50 latency: 102112
    p90 latency: 102189
    p95 latency: 102217
    p99 latency: 102257

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 28.7977
  latency (us):
    avg latency: 34724
    min latency: 34634
    max latency: 36781
    p50 latency: 34719
    p90 latency: 34768
    p95 latency: 34790
    p99 latency: 34812
```

## Python Sample 

### mobilevit.py 命令行参数说明
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
  --label_file LABEL_FILE
                        label file
  --input_file INPUT_FILE
                        input file
  --dataset_filelist DATASET_FILELIST
                        input dataset filelist
  --dataset_root DATASET_ROOT
                        input dataset root
  --dataset_output_file DATASET_OUTPUT_FILE
                        dataset output file

```

### mobilevit.py 运行示例

在本目录下运行  
```bash
#单张照片示例
python3 mobilevit.py \
-m /opt/vastai/vaststreamx/data/models/swin-b-fp16-none-1_3_224_224-vacc/mod \
--vdsp_params ../../../data/configs/swin_rgbplanar.json \
--device_id 0 \
--label_file ../../../data/labels/imagenet.txt \
--input_file ../../../data/images/cat.jpg

#数据集示例
python3 mobilevit.py \
-m /opt/vastai/vaststreamx/data/models/swin-b-fp16-none-1_3_224_224-vacc/mod \
--vdsp_params ../../../data/configs/swin_rgbplanar.json \
--device_id 0 \
--label_file ../../../data/labels/imagenet.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ILSVRC2012_img_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file swin_result.txt

#统计精度
python3 ../../../evaluation/classification/eval_topk.py  swin_result.txt  
```

### mobilevit.py 运行结果示例

```bash
#单张图片结果示例
Top5:
0th, score: 7.7422, class name: Egyptian cat
1th, score: 6.6758, class name: weasel
2th, score: 6.6562, class name: tabby, tabby cat
3th, score: 6.4141, class name: tiger cat
4th, score: 4.6797, class name: lynx, catamount

#精度统计结果
[VACC]: top1_rate: 81.834 top5_rate: 95.448
```

### mobilevit_prof.py 命令行参数说明

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


### mobilevit_prof.py 运行示例

在本目录下运行  
```bash
# 测试最大吞吐
python3 mobilevit_prof.py \
-m /opt/vastai/vaststreamx/data/models/swin-b-fp16-none-1_3_224_224-vacc/mod \
--vdsp_params ../../../data/configs/swin_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 200 \
--shape [3,224,224] \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 mobilevit_prof.py \
-m /opt/vastai/vaststreamx/data/models/swin-b-fp16-none-1_3_224_224-vacc/mod \
--vdsp_params ../../../data/configs/swin_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 200 \
--shape [3,224,224] \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 0
```


### mobilevit_prof.py 运行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 29.35
  latency (us):
    avg latency: 101678
    min latency: 37337
    max latency: 104610
    p50 latency: 102144
    p90 latency: 102230
    p95 latency: 102256
    p99 latency: 102281

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 28.75
  latency (us):
    avg latency: 34785
    min latency: 34671
    max latency: 36973
    p50 latency: 34779
    p90 latency: 34836
    p95 latency: 34860
    p99 latency: 34919
```



