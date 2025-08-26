# Brightness Op

本目录展示 自定义算子 img_brightness_adjust的用法：

1. 定义好自定义算子配置参数所需的结构体 
2. 通过 vsx::CustomOperator 加载 自定义算子
3. 设置好配置参数并通过 run_sync 调用 自定义算子


最后将图片转为bgr_interleave格式，并保存到文件  

自定义算子 img_brightness_adjust 的功能是将图片亮度进行调整，有一个 scale 参数，如果 scale > 1.0 则将对图片增加亮度，否则降低亮度。这个算子支持 yuv_nv12 格式图片输入。


## C++ Sample

### brightness 命令行参数说明
```bash
options:
  -d, --device_id      device id to run (unsigned int [=0])
      --input_file     input file (string [=../data/images/dog.jpg])
      --output_file    output file (string [=brightness_op_result.jpg])
      --elf_file       elf_file path (string [=/opt/vastai/vaststreamx/data/elf/brightness])
      --scale          brightness scale coefficient (float [=2.2])
  -?, --help           print this message
```
### brightness 命令示例     
在build目录里运行    
```bash
./vaststreamx-samples/bin/brightness  \
--device_id 0 \
--input_file ../data/images/dog.jpg \
--output_file brightness_result.jpg \
--elf_file /opt/vastai/vaststreamx/data/elf/brightness \
--scale 2.2
```
结果保存为 brightness_result.jpg


### brightness_prof 命令行参数说明
```bash
options:
      --elf_file       elf_file path (string [=/opt/vastai/vaststreamx/data/elf/brightness])
  -d, --device_ids     device id to run (string [=[0]])
      --shape          input shape [c,h,w] (string [=[3,640,640]])
  -i, --instance       instance number or range for each device (unsigned int [=1])
      --iterations     iterations count for one profiling (int [=10240])
      --scale          brightness scale coefficient (float [=2.2])
      --percentiles    percentiles of latency (string [=[50,90,95,99]])
      --input_host     cache input data into host memory (bool [=0])
  -?, --help           print this message
```
### brightness_prof 命令示例     
在build目录里运行    
```bash
./vaststreamx-samples/bin/brightness_prof \
--elf_file /opt/vastai/vaststreamx/data/elf/brightness \
--device_ids [0] \
--shape "[3,640,640]" \
--instance 8 \
--iterations 50000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```

### brightness_prof 命令结果示例     
```bash
- number of instances: 8
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 9899.89
  latency (us):
    avg latency: 807
    min latency: 560
    max latency: 1414
    p50 latency: 810
    p90 latency: 1027
    p95 latency: 1037
    p99 latency: 1055
```

## Python Sample


### brightness.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file
  --elf_file ELF_FILE   output file
  --scale SCALE         brightness scale coefficient
```
### brightness.py 命令行示例
在本文件目录下执行
```bash
python3 brightness.py \
--device_id 0 \
--input_file ../../../../data/images/dog.jpg \
--output_file brightness_result.jpg \
--elf_file /opt/vastai/vaststreamx/data/elf/brightness \
--scale 2.2
```

结果保存为 brightness_result.jpg


### brightness_prof.py 命令行参数说明

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
  --scale SCALE         brightness scale coefficient
  --percentiles PERCENTILES
                        percentiles of latency
  --input_host INPUT_HOST
                        cache input data into host memory
```
### brightness_prof.py 命令行示例

```bash
python3 brightness_prof.py \
--elf_file /opt/vastai/vaststreamx/data/elf/brightness \
--device_ids [0] \
--shape "[3,640,640]" \
--instance 8 \
--iterations 50000 \
--percentiles "[50,90,95,99]" \
--input_host  0
```

```bash
- number of instances: 2
  device: 0
  batch size: 1
  throughput (qps): 9918.775803750355
  latency (us):
    avg latency: 12240
    min latency: 494
    max latency: 32269
    p50 latency: 11688
    p90 latency: 20507
    p95 latency: 22382
    p99 latency: 26700
```


