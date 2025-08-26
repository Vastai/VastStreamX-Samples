# video capture samples

本目录提供基于 vsx 开发的 video capture sample。

## C++ Sample
### video_capture 命令行格式
```bash
options:
  -d, --device_id        device id to run (unsigned int [=0])
      --input_uri        input uri (string [=../data/videos/test.mp4])
      --frame_count      frame count to save (unsigned int [=0])
      --output_folder    output image file (string [=./output])
  -?, --help             print this message
```
### video_capture 命令示例
在 build 目录里执行  
```bash
mkdir video_capture_output
./vaststreamx-samples/bin/video_capture \
--device_id 0 \
--input_uri ../data/videos/test.mp4 \
--frame_count 10 \
--output_folder video_capture_output


# 解码图像将被保存在 video_capture_output 文件夹下
```

### video_capture 结果示例

```bash
Read total 10 frames.
```

## Python Sample

### video_capture.py 命令行格式
```bash
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file              INPUT_FILE
                        input file
  --output_folder           OUTPUT_FOLDER
                        output folder
```

### video_capture.py 命令示例

```bash
mkdir video_capture_output
python3 video_capture.py \
--device_id 0 \
--frame_count 10 \
--input_uri ../../data/videos/test.mp4 \
--output_folder video_capture_output
# 解码图像将被保存在 video_capture_output 文件夹下
```

### video_capture.py 结果示例

```bash
Read total 10 frames.
```