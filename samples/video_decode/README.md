# Video Decode
本目录提供硬解码 Video Decode Sample

数据准备:
将 videos.tar.gz 解压到 /opt/vastai/vaststreamx/data/里得到 videos文件夹

## C++ sample

### video_decode 命令行参数
```bash
options:
      --codec_type       codec type (string [=H264])
  -d, --device_id        device id to run (unsigned int [=0])
      --input_file       input file (string [=])
      --output_folder    output folder (string [=])
  -?, --help             print this message
```

### video_decode 运行示例
在 build 目录里执行   
```bash
#decode h264 
mkdir -p output_h264
./vaststreamx-samples/bin/video_decode \
--device_id 0 \
--codec_type h264 \
--input_file /opt/vastai/vaststreamx/data/videos/h264/1920x1080.h264 \
--output_folder output_h264
#结果将保存于 output_h264

#decode h265
mkdir -p output_h265
./vaststreamx-samples/bin/video_decode \
--device_id 0 \
--codec_type h265 \
--input_file /opt/vastai/vaststreamx/data/videos/hevc/1920x1080.hevc \
--output_folder output_h265
#结果将保存于 output_h265

#decode av1
mkdir -p output_av1
./vaststreamx-samples/bin/video_decode \
--device_id 0 \
--codec_type av1 \
--input_file /opt/vastai/vaststreamx/data/videos/videodecode_1920x1080.ivf \
--output_folder output_av1
#结果将保存于 output_av1
```


### video_decode_prof 命令行参数
```bash
options:
      --codec_type     codec type (string [=H264])
  -d, --device_ids     device id to run (string [=[0]])
      --input_file     input file (string [=])
  -i, --instance       instance number for each device (unsigned int [=1])
      --iterations     iterations count for one profiling (int [=10240])
      --percentiles    percentiles of latency (string [=[50, 90, 95, 99]])
  -?, --help           print this message
```

### video_decode_prof 运行示例

在 build 目录里   
```bash
# 测试H264最大吞吐
./vaststreamx-samples/bin/video_decode_prof \
--device_ids [0] \
--input_file /opt/vastai/vaststreamx/data/videos/h264/1920x1080.h264 \
--codec_type H264  \
--instance 10 \
--iterations 10000 

# H264测试最小时延
./vaststreamx-samples/bin/video_decode_prof \
--device_ids [0] \
--input_file /opt/vastai/vaststreamx/data/videos/h264/1920x1080.h264 \
--codec_type H264   \
--instance 1 \
--iterations 3000

# H265测试最大吞吐
./vaststreamx-samples/bin/video_decode_prof \
--device_ids [0] \
--input_file /opt/vastai/vaststreamx/data/videos/hevc/1920x1080.hevc \
--codec_type H265 \
--instance 10 \
--iterations 10000 

# H265测试最小时延
./vaststreamx-samples/bin/video_decode_prof \
--device_ids [0] \
--input_file /opt/vastai/vaststreamx/data/videos/hevc/1920x1080.hevc \
--codec_type H265 \
--instance 1 \
--iterations 5000 

# AV1测试最大吞吐
./vaststreamx-samples/bin/video_decode_prof \
--device_ids [0] \
--input_file /opt/vastai/vaststreamx/data/videos/videodecode_1920x1080.ivf \
--codec_type AV1 \
--instance 10 \
--iterations 10000 

# AV1测试最小时延
./vaststreamx-samples/bin/video_decode_prof \
--device_ids [0] \
--input_file /opt/vastai/vaststreamx/data/videos/videodecode_1920x1080.ivf \
--codec_type AV1 \
--instance 1 \
--iterations 5000 
```

### video_decode_prof 运行结果示例

```bash
# H264测试最大吞吐, 本结果是在 DCLK=650 下的结果
- number of instances: 10
  devices: [ 0 ]
  throughput (qps): 1547.18
  latency (us):
    avg latency: 27190
    min latency: 9834
    max latency: 54289
    p50 latency: 25081
    p90 latency: 34567
    p95 latency: 39294
    p99 latency: 40446

# H264测试最小时延, 本结果是在 DCLK=650 下的结果
- number of instances: 1
  devices: [ 0 ]
  throughput (qps): 509.486
  latency (us):
    avg latency: 8636
    min latency: 3646
    max latency: 11967
    p50 latency: 8883
    p90 latency: 10340
    p95 latency: 10368
    p99 latency: 10463

# H265测试最大吞吐, 本结果是在 DCLK=650 下的结果
- number of instances: 10
  devices: [ 0 ]
  throughput (qps): 1861.39
  latency (us):
    avg latency: 17005
    min latency: 8311
    max latency: 36843
    p50 latency: 15373
    p90 latency: 20206
    p95 latency: 20555
    p99 latency: 26594

# H265测试最小时延, 本结果是在 DCLK=650 下的结果
- number of instances: 1
  devices: [ 0 ]
  throughput (qps): 600.305
  latency (us):
    avg latency: 5576
    min latency: 3337
    max latency: 17086
    p50 latency: 5450
    p90 latency: 5620
    p95 latency: 5767
    p99 latency: 11574

# AV1测试最大吞吐, 本结果是在 DCLK=650 下的结果
- number of instances: 10
  devices: [ 0 ]
  throughput (qps): 1285.22
  latency (us):
    avg latency: 7030
    min latency: 9
    max latency: 55037
    p50 latency: 6025
    p90 latency: 19851
    p95 latency: 28645
    p99 latency: 53235

# AV1测试最小时延, 本结果是在 DCLK=650 下的结果
- number of instances: 1
  devices: [ 0 ]
  throughput (qps): 645.761
  latency (us):
    avg latency: 1549
    min latency: 7
    max latency: 12155
    p50 latency: 1344
    p90 latency: 4403
    p95 latency: 6327
    p99 latency: 11667
```

## Python Sample 

### video_decode.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  --codec_type CODEC_TYPE
                        hw-config file of the model suite
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        video file
  --output_folder OUTPUT_FOLDER
                        output folder
```

### video_decode.py 运行示例

在本目录下运行  
```bash

#decode h264 
mkdir -p output_h264
python3 video_decode.py \
--device_id 0 \
--codec_type h264 \
--input_file /opt/vastai/vaststreamx/data/videos/h264/1920x1080.h264 \
--output_folder output_h264
#结果将保存于 output_h264

#decode h265
mkdir -p output_h265
python3 video_decode.py \
--device_id 0 \
--codec_type h265 \
--input_file /opt/vastai/vaststreamx/data/videos/hevc/1920x1080.hevc \
--output_folder output_h265
#结果将保存于 output_h265

#decode av1 
mkdir -p output_av1
python3 video_decode.py \
--device_id 0 \
--codec_type av1 \
--input_file /opt/vastai/vaststreamx/data/videos/videodecode_1920x1080.ivf \
--output_folder output_av1
#结果将保存于 output_av1
```

## Python sample 性能测试

### video_decode_prof.py 命令行参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  --codec_type CODEC_TYPE
                        codec type eg. H264/H265
  -d DEVICE_IDS, --device_ids DEVICE_IDS
                        device ids to run
  --input_file INPUT_FILE
                        input file path
  -i INSTANCE, --instance INSTANCE
                        instance number for each device
  --iterations ITERATIONS
                        iterations count for one profiling
  --percentiles PERCENTILES
                        percentiles of latency
```

### video_decode_prof.py 运行示例

在本目录下运行
```bash
# 测试H264最大吞吐
python3 video_decode_prof.py \
--device_ids [0] \
--input_file /opt/vastai/vaststreamx/data/videos/h264/1920x1080.h264 \
--codec_type H264  \
--instance 18 \
--iterations 10000 

# H265测试最大吞吐
python3 video_decode_prof.py \
--device_ids [0] \
--input_file /opt/vastai/vaststreamx/data/videos/hevc/1920x1080.hevc \
--codec_type H265 \
--instance 18 \
--iterations 10000 

# H264测试最小时延
python3 video_decode_prof.py \
--device_ids [0] \
--input_file /opt/vastai/vaststreamx/data/videos/h264/1920x1080.h264 \
--codec_type H264   \
--instance 1 \
--iterations 3000 

# H265测试最小时延
python3 video_decode_prof.py \
--device_ids [0] \
--input_file /opt/vastai/vaststreamx/data/videos/hevc/1920x1080.hevc \
--codec_type H265 \
--instance 1 \
--iterations 5000 

# AV1测试最大吞吐
python3 video_decode_prof.py \
--device_ids [0] \
--input_file /opt/vastai/vaststreamx/data/videos/videodecode_1920x1080.ivf \
--codec_type AV1 \
--instance 18 \
--iterations 10000 

# AV1测试最小时延
python3 video_decode_prof.py \
--device_ids [0] \
--input_file /opt/vastai/vaststreamx/data/videos/videodecode_1920x1080.ivf \
--codec_type AV1   \
--instance 1 \
--iterations 3000 
```

### video_decode_prof 运行结果示例

```bash
# h264最大吞吐结果, 本结果是在 DCLK=650 下的结果
- number of instances: 18
  devices: [0]
  throughput (qps): 1565.62
  latency (us):
    avg latency: 45966
    min latency: 9782
    max latency: 78561
    p50 latency: 45633
    p90 latency: 57135
    p95 latency: 57465
    p99 latency: 63645
    
# h265最大吞吐结果, 本结果是在 DCLK=650 下的结果
- number of instances: 18
  devices: [0]
  throughput (qps): 1885.57
  latency (us):
    avg latency: 28560
    min latency: 8966
    max latency: 49244
    p50 latency: 28304
    p90 latency: 28919
    p95 latency: 30414
    p99 latency: 38848


# h264测试最小时延, 本结果是在 DCLK=650 下的结果
- number of instances: 1
  devices: [0]
  throughput (qps): 500.86
  latency (us):
    avg latency: 7953
    min latency: 2087
    max latency: 22454
    p50 latency: 7840
    p90 latency: 9832
    p95 latency: 9911
    p99 latency: 12567

# h265测试最小时延, 本结果是在 DCLK=650 下的结果
- number of instances: 1
  devices: [0]
  throughput (qps): 591.46
  latency (us):
    avg latency: 5081
    min latency: 1759
    max latency: 16351
    p50 latency: 4953
    p90 latency: 5132
    p95 latency: 5278
    p99 latency: 10933

# AV1最大吞吐结果, 本结果是在 DCLK=650 下的结果
- number of instances: 18
  devices: [0]
  throughput (qps): 1422.52
  latency (us):
    avg latency: 13742
    min latency: 179
    max latency: 102355
    p50 latency: 10548
    p90 latency: 36071
    p95 latency: 53126
    p99 latency: 96513


# AV1测试最小时延, 本结果是在 DCLK=650 下的结果
- number of instances: 1
  devices: [0]
  throughput (qps): 631.19
  latency (us):
    avg latency: 1709
    min latency: 150
    max latency: 12205
    p50 latency: 1485
    p90 latency: 4535
    p95 latency: 6469
    p99 latency: 11832
```
