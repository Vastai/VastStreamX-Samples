# Argmax Op

Argmax Op主要用于计算 fp16 的 多通道 planar tensor 在每个plane上的最大值的索引。

## C++ Sample

### argmax 命令行参数说明
```bash
options:
  -d, --device_id      device id to run (unsigned int [=0])
      --elf_file       elf file path (string [=/opt/vastai/vaststreamx/data/elf/planar_argmax])
      --input_shape    input_shape [c,h,w] (string [=[19,512,512]])
  -?, --help           print this message
```

### argmax 命令行示例
在build 目录下执行
```bash
./vaststreamx-samples/bin/argmax \
--device_id 0 \
--elf_file /opt/vastai/vaststreamx/data/elf/planar_argmax \
--input_shape "[19,512,512]"
```
输出 
```bash
output tensor shape:[1,1,512,512]
```

### argmax_prof 命令行参数说明

```bash
options:
      --elf_file       elf_file path (string [=/opt/vastai/vaststreamx/data/elf/planar_argmax])
  -d, --device_ids     device id to run (string [=[0]])
      --shape          input shape [c,h,w] (string [=[19,512,512]])
  -i, --instance       instance number or range for each device (unsigned int [=1])
      --iterations     iterations count for one profiling (int [=10240])
      --percentiles    percentiles of latency (string [=[50,90,95,99]])
      --input_host     cache input data into host memory (bool [=0])
  -?, --help           print this message
```

### argmax_prof 命令行参数示例
在build目录里执行
```bash
./vaststreamx-samples/bin/argmax_prof \
--device_ids [0] \
--elf_file /opt/vastai/vaststreamx/data/elf/planar_argmax \
--shape "[19,512,512]" \
--instance 4 \
--iterations 15000 \
--percentiles "[50,90,95,99]" \
--input_host 0 
```

### argmax_prof 命令行输出示例
```bash
- number of instances: 4
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 3008.22
  latency (us):
    avg latency: 1328
    min latency: 1157
    max latency: 1784
    p50 latency: 1329
    p90 latency: 1335
    p95 latency: 1337
    p99 latency: 1340
```


## Python Sample


### argmax.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --elf_file ELF_FILE   elf file
  -s SHAPE, --shape SHAPE
                        model input shape
```
### argmax.py 命令行示例
在当前目录执行
```bash
python3 argmax.py \
--device_id 0 \
--elf_file /opt/vastai/vaststreamx/data/elf/planar_argmax \
--shape "[19,512,512]"
```
结果将打印出 output shape


### argmax_prof.py 命令行参数说明

```bash 
optional arguments:
  -h, --help            show this help message and exit
  --elf_file ELF_FILE   elf file
  -d DEVICE_IDS, --device_ids DEVICE_IDS
                        device ids to run
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

### argmax_prof.py 命令行示例
```bash
python3 argmax_prof.py \
--elf_file /opt/vastai/vaststreamx/data/elf/planar_argmax \
--device_ids [0] \
--shape "[19,512,512]" \
--instance 4 \
--iterations 15000 \
--percentiles "[50,90,95,99]" \
--input_host 0 
```

### argmax_prof.py 命令行结果示例
```bash
- number of instances: 4
  device: 0
  batch size: 1
  throughput (qps): 3022.362206780337
  latency (us):
    avg latency: 1322
    min latency: 779
    max latency: 4203
    p50 latency: 1328
    p90 latency: 1346
    p95 latency: 1352
    p99 latency: 1363
```


