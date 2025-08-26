# CLIP Sample 

本 sample 提供 siglip 算法的基本用法。


## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://huggingface.co/google/siglip-so400m-patch14-384) 

## 数据准备
下载elf elf.tar.gz 并解压 到 /opt/vastai/vaststreamx/data/ 里  
tar xvf elf.tar.gz -C /opt/vastai/vaststreamx/data/
下载模型和数据集 siglip.tgz  并解压到 /opt/vastai/vaststreamx/data/ 里  


## Python Sample 
在当前文档所在目录执行

### siglip_sample.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        image model prefix of the model suite files
  --onnx_path ONNX_PATH
                        image model onnx file
  --hw_config HW_CONFIG
                        image model hw-config file of the model suite
  --norm_elf NORM_ELF   image model elf file
  --space2depth_elf SPACE2DEPTH_ELF
                        image model elf file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input file
  --dataset_root DATASET_ROOT
                        input dataset roo
```

### siglip_sample.py 命令行示例
```bash
# 测试 单张图片 将vsx 输出结果与 onnx输出结果计算余旋相似度
python3 siglip_sample.py \
--model_prefix /opt/vastai/vaststreamx/data/siglip/models/qnn/siglip-instruct-sim_vacc_runmodel \
--onnx_path /opt/vastai/vaststreamx/data/siglip/models/siglip-instruct-sim.onnx \
--norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
--space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_id 0 \
--input_file ../../data/images/CLIP.png 

#结果示例
score: [[0.5303]], score_onnx: [[0.53075683]], cos:1
#跑数据集, 计算vsx输出结果 与 onnx 输出结果的余旋相似度，统计其平均值
python3 siglip_sample.py \
--model_prefix /opt/vastai/vaststreamx/data/siglip/models/qnn/siglip-instruct-sim_vacc_runmodel \
--onnx_path /opt/vastai/vaststreamx/data/siglip/models/siglip-instruct-sim.onnx \
--norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
--space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_id 0 \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ 

# 余旋相似度统计结果
Average Cosine Similarity: 0.9999999865679674
Maximum Cosine Similarity: 1
Minimum Cosine Similarity: 0.9999998807907104
```

### siglip_image 模型性能分析

siglip_image_prof.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  --norm_elf NORM_ELF   normalize op elf file
  --space2depth_elf SPACE2DEPTH_ELF
                        space_to_depth op elf file
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
siglip_image_prof.py 命令行示例
```bash
# 测试最大吞吐
python3 siglip_image_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/siglip/models/qnn/siglip-instruct-sim_vacc_runmodel \
--norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
--space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 100 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 siglip_image_prof.py \
--model_prefix /opt/vastai/vaststreamx/data/siglip/models/qnn/siglip-instruct-sim_vacc_runmodel \
--norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
--space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 80 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```
siglip_image_prof.py 命令行结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 8.36
  latency (us):
    avg latency: 355070
    min latency: 237684
    max latency: 421579
    p50 latency: 356721
    p90 latency: 356814
    p95 latency: 356828
    p99 latency: 418224

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 7.51
  latency (us):
    avg latency: 133078
    min latency: 131472
    max latency: 134728
    p50 latency: 133063
    p90 latency: 134191
    p95 latency: 134504
    p99 latency: 134705
```
