# Run Stream Async Sample

本目录提供模型异步推理 sample。异步推理主要流程为：     
- 初始化模型并创建stream
- 创建接收线程，调用 get_output 获取输出，get_output在没有数据返回时，会阻塞
- 调用 process_async发送数据
- 调用 close_input， 这样 get_output 在取完数据后会返回 false 或者 exception
- 等待 接收线程结束
- 调用 wait_until_done 等待 stream 结束。


## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/pytorch/vision/blob/v0.9.0/torchvision/models/resnet.py)  [modelzoo](https://github.com/Vastai/VastModelZOO/tree/main/cv/classification/resnet) |
|  输入 shape |   [ (1,3,224,224) ]     |
| INT8量化方式 |   -          |
|  官方精度 |  top1_rate:76.130 ; top5_rate: 92.862       |
|  VACC FP16  精度 | top1_rate: 75.936 ; top5_rate: 92.85 |
|  VACC INT8  精度 | top1_rate: 75.816 ; top5_rate: 92.796  |

## 数据准备

下载模型 resnet50-int8-percentile-1_3_224_224-vacc 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 ILSVRC2012_img_val 到 /opt/vastai/vaststreamx/data/datasets 里



## C++ Sample

### run_stream_async 命令行参数说明
```bash
options:
  -m, --model_prefix           model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod])
      --hw_config              hw-config file of the model suite (string [=])
      --vdsp_params            vdsp preprocess parameter file (string [=../data/configs/resnet_bgr888.json])
  -d, --device_id              device id to run (unsigned int [=0])
      --label_file             label file (string [=../data/labels/imagenet.txt])
      --input_file             input file (string [=../data/images/cat.jpg])
      --dataset_filelist       input dataset filelist (string [=])
      --dataset_root           input dataset root (string [=])
      --dataset_output_file    dataset output file (string [=])
  -?, --help                   print this message
```
### run_stream_async 命令行示例
在build目录下运行
```bash
./vaststreamx-samples/bin/run_stream_async \
-m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
--vdsp_params ../data/configs/resnet_rgbplanar.json \
--device_id 0 \
--label_file ../data/labels/imagenet.txt \
--input_file ../data/images/cat.jpg

#测试数据集
./vaststreamx-samples/bin/run_stream_async \
-m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
--vdsp_params ../data/configs/resnet_rgbplanar.json \
-d 0 \
--label_file ../data/labels/imagenet.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ILSVRC2012_img_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file cls_result.txt

#统计精度
python3 ../evaluation/classification/eval_topk.py  cls_result.txt  
```
### run_stream_async 命令行结果示例
```bash
# 单张图片结果示例
Top5:
0th, class name: lynx, catamount, score: 0.30835
1th, class name: tabby, tabby cat, score: 0.240112
2th, class name: tiger cat, score: 0.212036
3th, class name: Egyptian cat, score: 0.120728
4th, class name: Persian cat, score: 0.0123367

# 统计精度结果示例
[VACC]:  top1_rate: 75.816 top5_rate: 92.796
```


### run_stream_async_prof 命令行参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/resnet_bgr888.json])
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

### run_stream_async_prof 命令行示例
在build目录下执行
```bash
./vaststreamx-samples/bin/run_stream_async_prof \
-m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
--vdsp_params ../data/configs/resnet_rgbplanar.json \
--device_ids [0] \
--batch_size 8 \
--instance 2 \
--iterations 2000 \
--shape "[3,256,256]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1
```
### run_stream_async_prof 命令行结果示例
```bash
- number of instances: 2
  devices: [ 0 ]
  queue size: 1
  batch size: 8
  throughput (qps): 3079.19
  latency (us):
    avg latency: 5186
    min latency: 5123
    max latency: 8046
    p50 latency: 5183
    p90 latency: 5209
    p95 latency: 5214
    p99 latency: 5262
```

## Python Sample

### run_stream_async.py 命令行参数说明
```bash
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
```
### run_stream_async.py 命令行示例
在当前目录下执行
```bash
python3 run_stream_async.py \
-m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
--vdsp_params ../../data/configs/resnet_rgbplanar.json \
--device_id 0 \
--label_file ../../data/labels/imagenet.txt \
--input_file ../../data/images/cat.jpg


# 跑数据集
python3 run_stream_async.py \
-m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
--vdsp_params ../../data/configs/resnet_rgbplanar.json \
--device_id 0 \
--label_file ../../data/labels/imagenet.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ILSVRC2012_img_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file cls_result.txt

#统计精度
python3 ../../evaluation/classification/eval_topk.py  cls_result.txt  
```

### run_stream_async.py 命令行结果示例

```bash
# 单张图片结果示例
Top5:
0th, class name: lynx, catamount, score: 0.308349609375
1th, class name: tabby, tabby cat, score: 0.2401123046875
2th, class name: tiger cat, score: 0.2120361328125
3th, class name: Egyptian cat, score: 0.1207275390625
4th, class name: Persian cat, score: 0.01233673095703125

# 统计精度结果示例
[VACC]:  top1_rate: 75.806 top5_rate: 92.806
```


### run_stream_async_prof.py 命令行参数说明

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


### run_stream_async_prof.py 命令行示例

```bash
python3 run_stream_async_prof.py \
-m /opt/vastai/vaststreamx/data/models/resnet50-int8-percentile-1_3_224_224-vacc/mod \
--vdsp_params ../../data/configs/resnet_rgbplanar.json \
--device_ids [0] \
--batch_size 8 \
--instance 2 \
--iterations 2000 \
--shape "[3,256,256]" \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1
```

### run_stream_async_prof.py 命令行结果示例
```bash
- number of instances: 2
  devices: [0]
  batch size: 8
  queue size: 1
  throughput (qps): 3170.66
  latency (us):
    avg latency: 10062
    min latency: 6075
    max latency: 10399
    p50 latency: 10064
    p90 latency: 10107
    p95 latency: 10124
    p99 latency: 10157
```


