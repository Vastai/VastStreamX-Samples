# Video Encoder Sample
本目录提供 Video Encoder  sample, 视频编码支持 YUV_NV12 格式的图片输入

数据准备:
将 videos.tar.gz 解压到 /opt/vastai/vaststreamx/data/里得到 videos文件夹

## C++ sample

### video_encode 命令行参数
```bash
options:
      --codec_type     codec type (string [=H264])
  -d, --device_id      device id to run (unsigned int [=0])
      --width          width (unsigned int [=0])
      --height         height (unsigned int [=0])
      --frame_rate     frame rate (unsigned int [=30])
      --input_file     input file (string [=])
      --output_file    output file (string [=])
  -?, --help           print this message
```

### video_encode 运行示例
在 build 目录里执行   
```bash
#h264文件编码示例
./vaststreamx-samples/bin/video_encode \
--device_id 0 \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--codec_type h264 \
--output_file output.h264 
#结果保存为 output.h264，可以用 VLC 播放器查看 编码得到的视频

#h265文件编码示例
./vaststreamx-samples/bin/video_encode \
--device_id 0 \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--codec_type h265 \
--output_file output.h265 
#结果保存为 output.h265，可以用 VLC 播放器查看 编码得到的视频


#av1文件编码示例
./vaststreamx-samples/bin/video_encode \
--device_id 0 \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--codec_type av1 \
--output_file output.av1 
#结果保存为 output.h265，可以用 VLC 播放器查看 编码得到的视频
```

### video_encode_prof 命令行参数
```bash
options:
      --codec_type     codec type (string [=H264])
  -d, --device_ids     device id to run (string [=[0]])
      --input_file     input file (string [=])
      --width          width (unsigned int [=0])
      --height         height (unsigned int [=0])
      --frame_rate     frame rate (unsigned int [=30])
  -i, --instance       instance number for each device (unsigned int [=1])
      --iterations     iterations count for one profiling (int [=10240])
      --percentiles    percentiles of latency (string [=[50, 90, 95, 99]])
  -?, --help           print this message
```

### video_encode_prof 运行示例

在 build 目录里   
```bash
# H264测试最大吞吐
./vaststreamx-samples/bin/video_encode_prof \
--device_ids [0] \
--codec_type H264  \
--percentiles "[50,90,95,99]" \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--instance 4 \
--iterations 1000

# H264测试最小时延
./vaststreamx-samples/bin/video_encode_prof \
--device_ids [0] \
--codec_type H264  \
--percentiles "[50,90,95,99]" \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--instance 1 \
--iterations 500

# H265测试最大吞吐
./vaststreamx-samples/bin/video_encode_prof \
--device_ids [0] \
--codec_type H265  \
--percentiles "[50,90,95,99]" \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--instance 4 \
--iterations 1000

# H265测试最小时延
./vaststreamx-samples/bin/video_encode_prof \
--device_ids [0] \
--codec_type H265  \
--percentiles "[50,90,95,99]" \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--instance 1 \
--iterations 500


# AV1测试最大吞吐
./vaststreamx-samples/bin/video_encode_prof \
--device_ids [0] \
--codec_type AV1  \
--percentiles "[50,90,95,99]" \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--instance 9 \
--iterations 5000

# AV1测试最小时延
./vaststreamx-samples/bin/video_encode_prof \
--device_ids [0] \
--codec_type AV1  \
--percentiles "[50,90,95,99]" \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--instance 1 \
--iterations 500
```

### video_encode_prof 运行结果示例

```bash
# H264测试最大吞吐,本结果在ECLK=200MHz情况下测试得到
- number of instances: 4
  devices: [ 0 ]
  throughput (qps): 195.697
  latency (us):
    avg latency: 20021
    min latency: 15329
    max latency: 30269
    p50 latency: 19655
    p90 latency: 22162
    p95 latency: 22279
    p99 latency: 25145

# H264测试最小时延,本结果在ECLK=200MHz情况下测试得到
- number of instances: 1
  devices: [ 0 ]
  throughput (qps): 50.201
  latency (us):
    avg latency: 19481
    min latency: 15022
    max latency: 25019
    p50 latency: 19232
    p90 latency: 20195
    p95 latency: 21552
    p99 latency: 21977

# H265测试最大吞吐,本结果在ECLK=200MHz情况下测试得到
- number of instances: 4
  devices: [ 0 ]
  throughput (qps): 79.5337
  latency (us):
    avg latency: 49888
    min latency: 31585
    max latency: 59452
    p50 latency: 49019
    p90 latency: 54778
    p95 latency: 55623
    p99 latency: 56219

# H265测试最小时延,本结果在ECLK=200MHz情况下测试得到
- number of instances: 1
  devices: [ 0 ]
  throughput (qps): 19.7726
  latency (us):
    avg latency: 50127
    min latency: 31235
    max latency: 57870
    p50 latency: 49937
    p90 latency: 51477
    p95 latency: 54740
    p99 latency: 55876

    
# AV1测试最大吞吐,本结果在ECLK=200MHz情况下测试得到
- number of instances: 9
  devices: [ 0 ]
  throughput (qps): 411.893
  latency (us):
    avg latency: 21148
    min latency: 11443
    max latency: 42621
    p50 latency: 19220
    p90 latency: 28659
    p95 latency: 29167
    p99 latency: 29654

# AV1测试最小时延,本结果在ECLK=200MHz情况下测试得到
- number of instances: 1
  devices: [ 0 ]
  throughput (qps): 78.6211
  latency (us):
    avg latency: 11935
    min latency: 10590
    max latency: 20087
    p50 latency: 11981
    p90 latency: 12224
    p95 latency: 12293
    p99 latency: 13898
```

## Python Sample 

### video_encode.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  --codec_type CODEC_TYPE
                        hw-config file of the model suite
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --width WIDTH         frame width
  --height HEIGHT       frame height
  --frame_rate FRAME_RATE
                        frame rate
  --input_file INPUT_FILE
                        video file
  --output_file OUTPUT_FILE
                        output file
```

### video_encode.py 运行示例

在本目录下运行  
```bash
#h264文件编码示例
python3 video_encode.py  \
--device_id 0 \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--codec_type h264 \
--output_file output.h264 
#结果保存为 output.h264，可以用 VLC 播放器查看 编码得到的视频

#h265文件编码示例
python3 video_encode.py  \
--device_id 0 \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--codec_type h265 \
--output_file output.h265 
#结果保存为 output.h265，可以用 VLC 播放器查看 编码得到的视频

#av1文件编码示例
python3 video_encode.py  \
--device_id 0 \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--codec_type av1 \
--output_file output.av1 
#结果保存为 output.av1 VLC 播放器查看 编码得到的视频
```

## Python sample 性能测试

### video_encode_prof.py 命令行参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  --codec_type CODEC_TYPE
                        codec type eg. H264/H265
  -d DEVICE_IDS, --device_ids DEVICE_IDS
                        device ids to run
  --width WIDTH         frame width
  --height HEIGHT       frame height
  --frame_rate FRAME_RATE
                        frame rate
  --input_file INPUT_FILE
                        input file path
  -i INSTANCE, --instance INSTANCE
                        instance number for each device
  --iterations ITERATIONS
                        iterations count for one profiling
  --percentiles PERCENTILES
                        percentiles of latency
```


### video_encode_prof.py 运行示例

在本目录下运行  
```bash
# H264测试最大吞吐
python3 video_encode_prof.py \
--device_ids [0] \
--codec_type H264  \
--percentiles "[50,90,95,99]" \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--instance 4 \
--iterations 1000

# H264测试最小时延
python3 video_encode_prof.py \
--device_ids [0] \
--codec_type H264  \
--percentiles "[50,90,95,99]" \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--instance 1 \
--iterations 1000

# H265测试最大吞吐
python3 video_encode_prof.py \
--device_ids [0] \
--codec_type H265  \
--percentiles "[50,90,95,99]" \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--instance 4 \
--iterations 1000


# H265测试最小时延
python3 video_encode_prof.py \
--device_ids [0] \
--codec_type H265  \
--percentiles "[50,90,95,99]" \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--instance 1 \
--iterations 1000


# AV1测试最大吞吐
python3 video_encode_prof.py \
--device_ids [0] \
--codec_type AV1  \
--percentiles "[50,90,95,99]" \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--instance 4 \
--iterations 1000


# AV1测试最小时延
python3 video_encode_prof.py \
--device_ids [0] \
--codec_type AV1  \
--percentiles "[50,90,95,99]" \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--instance 1 \
--iterations 1000
```

### video_encode_prof.py 运行结果示例

```bash
# H264测试最大吞吐,本结果在ECLK=200MHz情况下测试得到
- number of instances: 4
  devices: [0]
  throughput (qps): 190.75
  latency (us):
    avg latency: 21034
    min latency: 19044
    max latency: 29012
    p50 latency: 20664
    p90 latency: 22975
    p95 latency: 23057
    p99 latency: 24312

# H264测试最小时延,本结果在ECLK=200MHz情况下测试得到
- number of instances: 1
  devices: [0]
  throughput (qps): 48.84
  latency (us):
    avg latency: 20523
    min latency: 18386
    max latency: 25987
    p50 latency: 20289
    p90 latency: 21120
    p95 latency: 22458
    p99 latency: 23425


# H265测试最大吞吐,本结果在ECLK=200MHz情况下测试得到
- number of instances: 4
  devices: [0]
  throughput (qps): 78.64
  latency (us):
    avg latency: 50925
    min latency: 34195
    max latency: 59887
    p50 latency: 50064
    p90 latency: 55609
    p95 latency: 56649
    p99 latency: 57161

# H265测试最小时延,本结果在ECLK=200MHz情况下测试得到
- number of instances: 1
  devices: [0]
  throughput (qps): 19.63
  latency (us):
    avg latency: 50978
    min latency: 34577
    max latency: 59039
    p50 latency: 50702
    p90 latency: 52361
    p95 latency: 55479
    p99 latency: 56691

# AV1测试最大吞吐,本结果在ECLK=200MHz情况下测试得到
- number of instances: 4
  devices: [0]
  throughput (qps): 316.38
  latency (us):
    avg latency: 12729
    min latency: 12124
    max latency: 20845
    p50 latency: 12493
    p90 latency: 12958
    p95 latency: 13131
    p99 latency: 17915

# AV1测试最小时延,本结果在ECLK=200MHz情况下测试得到
- number of instances: 1
  devices: [0]
  throughput (qps): 78.35
  latency (us):
    avg latency: 12829
    min latency: 11927
    max latency: 20962
    p50 latency: 12773
    p90 latency: 13145
    p95 latency: 13458
    p99 latency: 17276
```
