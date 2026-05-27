# NLP Sample

本目录提供基于 bert 模型的 NLP sample

## 数据准备
下载模型 bert_base_en_qa-384-int8-max-1_384_1_384_1_384-vacc 到 /opt/vastai/vastpipe/data/models 里
下载数据集 SQuAD_1.1 到 /opt/vastai/vastpipe/data/datasets 里

## C++ sample

### bert 命令行参数说明
```bash
options:
  -m, --model_prefix             model prefix of the model suite files (string [=/opt/vastai/vastpipe/data/models/bert_base_en_qa-384-int8-max-1_384_1_384_1_384-vacc/mod])
      --hw_config                hw-config file of the model suite (string [=])
      --vdsp_params              vdsp preprocess parameter file (string [=../data/configs/bert_vdsp.json])
  -d, --device_id                device id to run (unsigned int [=0])
      --input_file               input npz file (string [=])
      --output_file              output npz file (string [=])
      --dataset_filelist         dataset npz filename list (string [=])
      --dataset_root             dataset root (string [=])
      --dataset_output_folder    dataset result output folder (string [=])
  -?, --help                     print this message
```
### bert 运行示例

在 build 目录里执行  

跑单个npz
```bash
./vastpipe-samples/bin/bert \
-m /opt/vastai/vastpipe/data/models/bert_base_en_qa-384-int8-max-1_384_1_384_1_384-vacc/mod \
--vdsp_params ../data/configs/bert_vdsp.json \
--device_id 0 \
--input_file  /opt/vastai/vastpipe/data/datasets/SQuAD_1.1/val_npz_6inputs/test_0.npz \
--output_file ./bert_result.npz
```
结果保存到 ./bert_result.npz 里


跑数据集
```bash
mkdir -p bert_output
./vastpipe-samples/bin/bert \
-m /opt/vastai/vastpipe/data/models/bert_base_en_qa-384-int8-max-1_384_1_384_1_384-vacc/mod \
--vdsp_params ../data/configs/bert_vdsp.json \
--device_id  0 \
--dataset_filelist  /opt/vastai/vastpipe/data/datasets/SQuAD_1.1_filelist.txt  \
--dataset_root /opt/vastai/vastpipe/data/datasets/ \
--dataset_output_folder ./bert_output
```

计算数据集精度

```bash
python3 ../evaluation/bert/squad_eval.py  \
--result_dir ./bert_output \
--eval_path /opt/vastai/vastpipe/data/datasets/SQuAD_1.1/dev-v1.1.json \
--vocab_path /opt/vastai/vastpipe/data/datasets/SQuAD_1.1/vocab.txt
```

精度输出结果
```bash
{"exact_match": 71.18259224219489, "f1": 81.61918268029476}
```

### bert_prof 命令行参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vastpipe/data/models/bert_base_en_qa-384-int8-max-1_384_1_384_1_384-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/bert_vdsp.json])
  -d, --device_ids      device id to run (string [=[0]])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        instance number or range for each device (unsigned int [=1])
      --iterations      iterations count for one profiling (int [=1024])
      --percentiles     percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=2])
  -?, --help            print this message
```
### bert_prof 运行示例

在 build 目录里执行  

```bash
# 测试最大吞吐
./vastpipe-samples/bin/bert_prof  \
-m /opt/vastai/vastpipe/data/models/bert_base_en_qa-384-int8-max-1_384_1_384_1_384-vacc/mod \
--vdsp_params ../data/configs/bert_vdsp.json \
--device_ids  [0] \
--batch_size  8 \
--instance 1 \
--iterations 200 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vastpipe-samples/bin/bert_prof  \
-m /opt/vastai/vastpipe/data/models/bert_base_en_qa-384-int8-max-1_384_1_384_1_384-vacc/mod \
--vdsp_params ../data/configs/bert_vdsp.json \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 500 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```
### bert_prof 运行结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  device: 0
  queue size: 1
  batch size: 8
  throughput (qps): 238.516
  latency (us):
    avg latency: 100088
    min latency: 50815
    max latency: 119114
    p50 latency: 100335
    p90 latency: 100452
    p95 latency: 100471
    p99 latency: 100517

# 测试最小时延
- number of instances: 1
  device: 0
  queue size: 0
  batch size: 1
  throughput (qps): 141.768
  latency (us):
    avg latency: 7052
    min latency: 6990
    max latency: 7473
    p50 latency: 7050
    p90 latency: 7060
    p95 latency: 7065
    p99 latency: 7101
```

## Python sample 功能测试

### bert.py 命令行参数说明
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
                        input_npz filename list
  --dataset_root DATASET_ROOT
                        dataset root
  --dataset_output_folder DATASET_OUTPUT_FOLDER
                        dataset output folder save result to
```

### 命令示例

在本目录下运行

跑单个实例
```bash
python3 bert.py \
-m /opt/vastai/vastpipe/data/models/bert_base_en_qa-384-int8-max-1_384_1_384_1_384-vacc/mod \
--vdsp_params ../../../data/configs/bert_vdsp.json \
--device_id 0 \
--input_file  /opt/vastai/vastpipe/data/datasets/SQuAD_1.1/val_npz_6inputs/test_0.npz \
--output_file ./bert_result.npz
```
结果保存于 bert_output.npz

跑数据集
```bash
mkdir -p bert_output
python3 bert.py \
-m /opt/vastai/vastpipe/data/models/bert_base_en_qa-384-int8-max-1_384_1_384_1_384-vacc/mod \
--vdsp_params ../../../data/configs/bert_vdsp.json \
--device_id 0 \
--dataset_filelist  /opt/vastai/vastpipe/data/datasets/SQuAD_1.1_filelist.txt  \
--dataset_root /opt/vastai/vastpipe/data/datasets/ \
--dataset_output_folder ./bert_output
```

计算数据集精度

```bash
python3 ../../../evaluation/bert/squad_eval.py  \
--result_dir ./bert_output \
--eval_path /opt/vastai/vastpipe/data/datasets/SQuAD_1.1/dev-v1.1.json \
--vocab_path /opt/vastai/vastpipe/data/datasets/SQuAD_1.1/vocab.txt
```

精度输出结果
```bash
{"exact_match": 71.18259224219489, "f1": 81.61918268029476}
```


### bert_prof.py 命令行说明

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
  --iterations ITERATIONS
                        iterations count for one profiling
  --queue_size QUEUE_SIZE
                        aync wait queue size
  --percentiles PERCENTILES
                        percentiles of latency
  --input_host INPUT_HOST
                        cache input data into host memory
```

### bert_prof.py 命令示例 

在本目录下运行

```bash
# 测试最大吞吐
python3 bert_prof.py \
-m /opt/vastai/vastpipe/data/models/bert_base_en_qa-384-int8-max-1_384_1_384_1_384-vacc/mod \
--vdsp_params ../../../data/configs/bert_vdsp.json \
--device_ids  [0] \
--batch_size  8 \
--instance 1 \
--iterations 200 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 bert_prof.py \
-m /opt/vastai/vastpipe/data/models/bert_base_en_qa-384-int8-max-1_384_1_384_1_384-vacc/mod \
--vdsp_params ../../../data/configs/bert_vdsp.json \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 500 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```


###  bert_prof.py 运行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 8
  throughput (qps): 231.25
  latency (us):
    avg latency: 103527
    min latency: 53478
    max latency: 117786
    p50 latency: 103636
    p90 latency: 103708
    p95 latency: 103733
    p99 latency: 103797

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 135.40
  latency (us):
    avg latency: 7384
    min latency: 7324
    max latency: 7920
    p50 latency: 7382
    p90 latency: 7391
    p95 latency: 7398
    p99 latency: 7431
```
