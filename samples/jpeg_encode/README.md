# image encode samples

本目录提供基于vsx 开发的 jpeg encode sample。实现在产品卡上将nv12数据编码为jpeg格式。

## C++ Sample
### jpeg_encode 命令行格式
```bash
  -d, --device_id      device id to run (unsigned int [=0])
      --height         image height (unsigned int [=354])
      --width          image width (unsigned int [=474])
      --input_file     input nv12 file (string [=../data/images/cat_354x474_nv12.yuv])
      --output_file    output image file (string [=./jpeg_encode_result.jpg])
  -?, --help           print this message
```
### jpeg_encode 命令示例
在 build 目录里执行  
```bash
./vaststreamx-samples/bin/jpeg_encode \
--device_id 0 \
--height 354 \
--width 474 \
--input_file ../data/images/cat_354x474_nv12.yuv \
--output_file ./jpeg_encode_result.jpg
# 编码图像将被保存为 jpeg_encode_result.jpg
```

### jpeg_encode 结果示例

```bash
Encoded data bytes: 25433
```

### jpeg_encode_prof 命令行参数
```bash
options:
  -d, --device_ids     device id to run (string [=[0]])
      --input_file     input file (string [=../data/images/plate_1920_1080.yuv])
      --width          width (unsigned int [=1920])
      --height         height (unsigned int [=1080])
  -i, --instance       instance number for each device (unsigned int [=1])
      --iterations     iterations count for one profiling (int [=10240])
      --percentiles    percentiles of latency (string [=[50, 90, 95, 99]])
  -?, --help           print this message
```

### jpeg_encode_prof 运行示例

在 build 目录里   
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/jpeg_encode_prof \
--device_ids [0] \
--percentiles "[50,90,95,99]" \
--input_file ../data/images/plate_1920_1080.yuv \
--width 1920 \
--height 1080 \
--instance 20 \
--iterations 30000

# 测试最小时延
./vaststreamx-samples/bin/jpeg_encode_prof \
--device_ids [0] \
--percentiles "[50,90,95,99]" \
--input_file ../data/images/plate_1920_1080.yuv \
--width 1920 \
--height 1080 \
--instance 1 \
--iterations 500


```

### jpeg_encode_prof 运行结果示例

```bash
# 测试最大吞吐,本结果在ECLK=810MHz情况下测试得到
- number of instances: 20
  devices: [ 0 ]
  throughput (qps): 2651.95
  latency (us):
    avg latency: 7549
    min latency: 5349
    max latency: 86647
    p50 latency: 5468
    p90 latency: 10515
    p95 latency: 10544
    p99 latency: 10624


# 测试最小时延,本结果在ECLK=810MHz情况下测试得到
- number of instances: 1
  devices: [ 0 ]
  throughput (qps): 184.88
  latency (us):
    avg latency: 5413
    min latency: 5337
    max latency: 10468
    p50 latency: 5406
    p90 latency: 5431
    p95 latency: 5453
    p99 latency: 5581
```
## Python Sample

### jpeg_encode.py 命令行格式
```bash
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --height                  HEIGHT
                        image height
  --width                   WIDTH
                        image width
  --input_file              INPUT_FILE
                        input nv12 file
  --output_file             OUTPUT_FILE
                        output jpeg file
```

### jpeg_encode.py 命令示例

```bash
python3 jpeg_encode.py \
--device_id 0 \
--height 354 \
--width 474 \
--input_file ../../data/images/cat_354x474_nv12.yuv \
--output_file jpeg_encode_result.jpg
# 编码图像将被保存为 jpeg_encode_result.jpg
```

### jpeg_encode.py 结果示例

```bash
Encoded data bytes: 23662
```


## Python sample 性能测试

### jpeg_encode_prof.py 命令行参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE_IDS, --device_ids DEVICE_IDS
                        device ids to run
  --width WIDTH         frame width
  --height HEIGHT       frame height
  --input_file INPUT_FILE
                        input file path
  -i INSTANCE, --instance INSTANCE
                        instance number for each device
  --iterations ITERATIONS
                        iterations count for one profiling
  --percentiles PERCENTILES
                        percentiles of latency
```


### jpeg_encode_prof.py 运行示例

在本目录下运行  
```bash
# 测试最大吞吐
python3 jpeg_encode_prof.py \
--device_ids [0] \
--percentiles "[50,90,95,99]" \
--input_file ../../data/images/plate_1920_1080.yuv \
--width 1920 \
--height 1080 \
--instance 20 \
--iterations 30000

# H264测试最小时延
python3 jpeg_encode_prof.py \
--device_ids [0] \
--percentiles "[50,90,95,99]" \
--input_file ../../data/images/plate_1920_1080.yuv \
--width 1920 \
--height 1080 \
--instance 1 \
--iterations 1000
```

### jpeg_encode_prof.py 运行结果示例

```bash
# 测试最大吞吐,本结果在ECLK=810MHz情况下测试得到
- number of instances: 30
  devices: [0]
  throughput (qps): 2678.47
  latency (us):
    avg latency: 12092
    min latency: 5556
    max latency: 129807
    p50 latency: 11758
    p90 latency: 14401
    p95 latency: 16749
    p99 latency: 18010

# 测试最小时延,本结果在ECLK=810MHz情况下测试得到
- number of instances: 1
  devices: [0]
  throughput (qps): 179.23
  latency (us):
    avg latency: 5673
    min latency: 5463
    max latency: 17204
    p50 latency: 5555
    p90 latency: 5663
    p95 latency: 5686
    p99 latency: 10573
```
