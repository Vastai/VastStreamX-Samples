# NormaTensor3Ch Op

NormaTensor3Ch 是一个针对 3通道图像 的 归一化 tensor 化的操作

## norma_tensor_3ch 命令参数说明

```bash
options:
  -d, --device_id      device id to run (unsigned int [=0])
      --elf_file       elf file path (string [=/opt/vastai/vaststreamx/data/elf/norma_tensor_3ch])
      --input_shape    input_shape [c,h,w] (string [=[3,640,640]])
  -?, --help           print this message
```

## norma_tensor_3ch 命令示例
在build目录下执行
```bash
./vaststreamx-samples/bin/norma_tensor_3ch \
--device_id 0 \
--elf_file /opt/vastai/vaststreamx/data/elf/norma_tensor_3ch \
--input_shape "[3,640,640]"
```
最后将输出的tensor shape打印出来


## norma_tensor_3ch_prof 命令参数说明
```bash
options:
      --elf_file       elf_file path (string [=/opt/vastai/vaststreamx/data/elf/norma_tensor_3ch])
  -d, --device_ids     device id to run (string [=[0]])
      --shape          input shape [c,h,w] (string [=[3,640,640]])
  -i, --instance       instance number or range for each device (unsigned int [=1])
      --iterations     iterations count for one profiling (int [=10240])
      --percentiles    percentiles of latency (string [=[50,90,95,99]])
      --input_host     cache input data into host memory (bool [=0])
  -?, --help           print this message
```

## norma_tensor_3ch_prof 命令示例
```bash
./vaststreamx-samples/bin/norma_tensor_3ch_prof \
--elf_file /opt/vastai/vaststreamx/data/elf/norma_tensor_3ch \
--device_ids [0] \
--shape "[3,640,640]" \
--instance 4 \
--iterations 10000 \
--percentiles "[50,90,95,99]" \
--input_host 0 
```

## norma_tensor_3ch_prof 命令结果示例
```bash
- number of instances: 4
  device: 0
  queue size: 0
  batch size: 1
  throughput (qps): 2764.38
  latency (us):
    avg latency: 1442
    min latency: 1205
    max latency: 1751
    p50 latency: 1442
    p90 latency: 1464
    p95 latency: 1470
    p99 latency: 1479
```

## Python Sample 

### norma_tensor_3ch.py 命令参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --elf_file ELF_FILE   elf file
  -s SHAPE, --shape SHAPE
                        model input shape
```

### norma_tensor_3ch.py 命令示例
```bash
python3 norma_tensor_3ch.py \
--device_id 0 \
--elf_file /opt/vastai/vaststreamx/data/elf/norma_tensor_3ch \
--shape "[3,640,640]"
```

### norma_tensor_3ch.py 命令结果示例
```bash
output shape:  [3, 640, 640]
```

### norma_tensor_3ch_prof.py 命令参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  --elf_file ELF_FILE   elf file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  -i INSTANCE, --instance INSTANCE
                        instance number for each device
  -s SHAPE, --shape SHAPE
                        model input shape
  --iterations ITERATIONS
                        iterations count for one profiling
  --percentiles PERCENTILES
                        percentiles of latency
  --input_host INPUT_HOST
                        cache input data into host memory
```


### norma_tensor_3ch_prof.py 命令行示例

```bash
python3 norma_tensor_3ch_prof.py \
--elf_file /opt/vastai/vaststreamx/data/elf/norma_tensor_3ch \
--device_ids [0] \
--shape "[3,640,640]" \
--instance 4 \
--iterations 10000 \
--percentiles "[50,90,95,99]" \
--input_host 0 
```

### norma_tensor_3ch_prof.py 命令行结果示例

```bash
- number of instances: 4
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 2704.898668890017
  latency (us):
    avg latency: 1477
    min latency: 1218
    max latency: 2505
    p50 latency: 1476
    p90 latency: 1507
    p95 latency: 1524
    p99 latency: 1546
```