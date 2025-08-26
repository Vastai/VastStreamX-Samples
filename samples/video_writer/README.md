# Video Encoder Sample
本目录提供 Video Encoder  sample, 视频编码支持 YUV_NV12 格式的图片输入

数据准备:
将 videos.tar.gz 解压到 /opt/vastai/vaststreamx/data/里得到 videos文件夹

## C++ sample

### video_writer 命令行参数
```bash
options:
      --codec_type    codec type (string [=H264])
  -d, --device_id     device id to run (unsigned int [=0])
      --width         width (unsigned int [=0])
      --height        height (unsigned int [=0])
      --frame_rate    frame rate (unsigned int [=30])
      --input_file    input file (string [=])
      --output_uri    output uri (string [=./test.ts])
  -?, --help          print this message
```

### video_writer 运行示例
在 build 目录里执行   
```bash
#h264文件编码示例
./vaststreamx-samples/bin/video_writer \
--device_id 0 \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--codec_type h264 \
--output_uri ./test_h264.ts
#结果保存为 ./test_h264.ts，可以用 PotPlayer 播放器查看 编码得到的视频

#h265文件编码示例
./vaststreamx-samples/bin/video_writer \
--device_id 0 \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--codec_type h265 \
--output_uri ./test_h265.ts
#结果保存为 test_h265.ts PotPlayer 播放器查看 编码得到的视频
```

## Python Sample 

### video_writer.py 命令行参数说明
```bash
options:
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
  --output_uri OUTPUT_URI             output uri
```

### video_writer.py 运行示例

在本目录下运行  
```bash
#h264文件编码示例
python3 video_writer.py  \
--device_id 0 \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--codec_type h264 \
--output_uri test_h264.ts
#结果保存为 test_h264.ts，可以用 PotPlayer 播放器查看 编码得到的视频

#h264文件编码示例
python3 video_writer.py  \
--device_id 0 \
--input_file /opt/vastai/vaststreamx/data/videos/yuv/bbs_sunflower_nv12_1920x1080_30.yuv \
--width 1920 \
--height 1080 \
--frame_rate 30 \
--codec_type h265 \
--output_uri test_h265.ts
#结果保存为 test_h265.ts，可以用 PotPlayer 播放器查看 编码得到的视频
```
