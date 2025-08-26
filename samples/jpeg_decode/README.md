# image decode samples

本目录提供基于 vsx 开发的 jpeg decode sample。实现在产品卡上将jpeg格式解码为nv12数据。

## C++ Sample
### jpeg_decode 命令行格式
```bash
  -d, --device_id      device id to run (unsigned int [=0])
      --input_file     input jpeg file (string [=../data/images/cat.jpg])
      --output_file    output nv12 file (string [=./jpeg_decode_result.yuv])
  -?, --help           print this message
```
### jpeg_decode 命令示例
在 build 目录里执行  
```bash
./vaststreamx-samples/bin/jpeg_decode \
--device_id 0 \
--input_file ../data/images/cat.jpg \
--output_file ./jpeg_decode_result.yuv
# 解码 nv12 图像将被保存为 jpeg_decode_result.yuv，对应的解码RGB图像将被保存为jpeg_decode_result.bmp
```

### jpeg_decode 结果示例

```bash
Decoded image format is: YUV_NV12
```

### jpeg_decode_prof 命令行参数
```bash
options:
  -d, --device_ids     device id to run (string [=[0]])
      --input_file     input file (string [=../data/images/plate_1920_1080.jpg])
  -i, --instance       instance number for each device (unsigned int [=1])
      --iterations     iterations count for one profiling (int [=10240])
      --percentiles    percentiles of latency (string [=[50, 90, 95, 99]])
  -?, --help           print this message
```

### jpeg_decode_prof 运行示例

在 build 目录里   
```bash
# 测试最大吞吐
export VAME_JPEG_OUTBUFFER=0
./vaststreamx-samples/bin/jpeg_decode_prof \
--device_ids [0] \
--input_file ../data/images/plate_1920_1080.jpg \
--instance 100 \
--iterations 40480 

# H测试最小时延
export VAME_JPEG_OUTBUFFER=0
./vaststreamx-samples/bin/jpeg_decode_prof \
--device_ids [0] \
--input_file ../data/images/plate_1920_1080.jpg \
--instance 1 \
--iterations 3000
```

### jpeg_decode_prof 运行结果示例
```bash
# 最大吞吐, 本结果是在 DCLK=835 下的结果
- number of instances: 100
  devices: [ 0 ]
  throughput (qps): 2742.17
  latency (us):
    avg latency: 36467
    min latency: 21871
    max latency: 389832
    p50 latency: 35430
    p90 latency: 36507
    p95 latency: 36638
    p99 latency: 36899


# 测试最小时延, 本结果是在 DCLK=835 下的结果
- number of instances: 1
  devices: [ 0 ]
  throughput (qps): 482.801
  latency (us):
    avg latency: 2077
    min latency: 1924
    max latency: 13506
    p50 latency: 2086
    p90 latency: 2096
    p95 latency: 2107
    p99 latency: 2124

```

## Python Sample

### jpeg_decode.py 命令行格式
```bash
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file              INPUT_FILE
                        input jpeg file
  --output_file             OUTPUT_FILE
                        output nv12 file
```

### jpeg_decode.py 命令示例

```bash
python3 jpeg_decode.py \
--device_id 0 \
--input_file ../../data/images/cat.jpg \
--output_file jpeg_decode_result.yuv
# 解码 nv12 图像将被保存为 jpeg_decode_result.yuv，对应的解码RGB图像将被保存为jpeg_decode_result.bmp
```

### jpeg_decode.py 结果示例

```bash
Decoded image format is ImageFormat.YUV_NV12
```

## Python sample 性能测试

### jpeg_decode_prof.py 命令行参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE_IDS, --device_ids DEVICE_IDS
                        device ids to run
  --input_file INPUT_FILE
                        input file
  -i INSTANCE, --instance INSTANCE
                        instance number for each device
  --iterations ITERATIONS
                        iterations count for one profiling
  --percentiles PERCENTILES
                        percentiles of latency
```

### jpeg_decode_prof.py 运行示例

在本目录下运行
```bash
# 测试最大吞吐
export VAME_JPEG_OUTBUFFER=0
python3 jpeg_decode_prof.py \
--device_ids [0] \
--input_file ../../data/images/plate_1920_1080.jpg \
--instance 150 \
--iterations 80480 


# 测试最小时延
export VAME_JPEG_OUTBUFFER=0
python3 jpeg_decode_prof.py \
--device_ids [0] \
--input_file ../../data/images/plate_1920_1080.jpg \
--instance 1 \
--iterations 3000 
```

### jpeg_decode_prof 运行结果示例

```bash
# 最大吞吐结果, 本结果是在 DCLK=835 下的结果
- number of instances: 100
  devices: [0]
  throughput (qps): 2540.88
  latency (us):
    avg latency: 45421
    min latency: 3504
    max latency: 456286
    p50 latency: 44769
    p90 latency: 52534
    p95 latency: 56020
    p99 latency: 78804
    
# 测试最小时延, 本结果是在 DCLK=835 下的结果
- number of instances: 1
  devices: [0]
  throughput (qps): 471.66
  latency (us):
    avg latency: 2208
    min latency: 1952
    max latency: 17353
    p50 latency: 2190
    p90 latency: 2220
    p95 latency: 2240
    p99 latency: 2290
```
