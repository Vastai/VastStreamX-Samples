# Decode  Detection Encode Sample

本目录展示如何实现视频 解码 + AI + 编码 的功能, 支持多路以及丢帧

注意：如果不希望丢帧，将 --drop 参数设置为 0 即可。

VideoEncoder 与 VideoWriter 的相同点是都可以对视频帧进行编码， 不同点是 VideoWriter 功能更多，支持推流，支持音频与视频同时写

## Cpp Sample

### decode_detection_encode_multi 命令行参数说明

```bash
options:
  -m, --model_prefix       model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod])
      --vdsp_params        vdsp preprocess parameter file (string [=./data/configs/yolo_div255_yuv_nv12.json])
  -d, --device_id          device id to run (unsigned int [=0])
      --threshold          threshold for detection (float [=0.5])
      --uri                input uri (string [=../data/videos/test.mp4])
      --output_path        output_path (string [=])
      --num_channels       number of channels to decode [ default: 1 ] (unsigned int [=1])
      --loop               loop count for each channel [ default: 1 ] (unsigned int [=1])
      --keep               keep num [default:1] (unsigned int [=1])
      --drop               drop num [default:0] (unsigned int [=0])
      --uri_list           input uri list file (string [=])
      --disable_encoder    enable encoder or not (unsigned int [=0])
      --save_output        save output or not (unsigned int [=0])
  -?, --help               print this message
```

### decode_detection_encode_multi 命令示例

```bash
./vaststreamx-samples/bin/decode_detection_encode_multi \
-m /opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../data/configs/yolo_div255_yuv_nv12.json \
--device_id 0 \
--threshold 0.5 \
--uri ../data/videos/test.mp4 \
--output_path output_encode_cpp \
--num_channels 6 \
--keep 1 \
--drop 0 \
--save_output 1

```

结果是在 output_encode_cpp 文件夹里，生成了 channel_0.ts 到 channel_5.ts 文件，以及 channel_0_detection.txt 到 channel_5_detection.txt 文件。 ts 文件可以用 potplayer 播放

### decode_detection_writer_multi 命令行参数说明

```bash
options:
  -m, --model_prefix       model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod])
      --vdsp_params        vdsp preprocess parameter file (string [=./data/configs/yolo_div255_yuv_nv12.json])
  -d, --device_id          device id to run (unsigned int [=0])
      --threshold          threshold for detection (float [=0.5])
      --uri                input uri (string [=../data/videos/test.mp4])
      --output_path        output_path (string [=])
      --num_channels       number of channels to decode [ default: 1 ] (unsigned int [=1])
      --loop               loop count for each channel [ default: 1 ] (unsigned int [=1])
      --keep               keep num [default:1] (unsigned int [=1])
      --drop               drop num [default:0] (unsigned int [=0])
      --uri_list           input uri list file (string [=])
      --disable_encoder    enable encoder or not (unsigned int [=0])
      --save_output        save output or not (unsigned int [=0])
```

### decode_detection_writer_multi 命令示例

```bash
./vaststreamx-samples/bin/decode_detection_writer_multi \
-m /opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../data/configs/yolo_div255_yuv_nv12.json \
--device_id 0 \
--threshold 0.5 \
--uri ../data/videos/test.mp4 \
--output_path output_writer_cpp \
--num_channels 6 \
--keep 1 \
--drop 0 \
--save_output 1
```

结果是在 output_writer_cpp 文件夹里，生成了 channel_0.ts 到 channel_5.ts 文件，以及 channel_0_detection.txt 到 channel_5_detection.txt 文件。 ts 文件可以用 potplayer 播放

## Python Sample

### decode_detection_encode_multi.py 脚本参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run [ default: 0 ]
  --threshold THRESHOLD
                        threshold for detection [default: 0.1]
  --uri URI             uri to decode
  --output_path OUTPUT_PATH
                        output path [ default: output_result ]
  --num_channels NUM_CHANNELS
                        number of channels to decode [ default: 1 ]
  --loop LOOP           loop count for each channel [ default: 1 ]
  --keep KEEP           keep num [default:1]
  --drop DROP           drop num [default:0]
  --uri_list URI_LIST   input uri list file
  --disable_encoder     enable encoder or not
  --save_output         save output or not
```

### decode_detection_encode_multi.py 脚本运行示例

```bash
python3 decode_detection_encode_multi.py \
-m /root/tools2/va1/build_model/deploy_weights/yolov5s_640/mod \
--vdsp_params ../../data/configs/yolo_div255_yuv_nv12.json \
--threshold 0.5 \
-d 0 \
--uri ../../data/videos/output_set.ts \
--output_path output_encoder_py \
--num_channels 6 \
--keep 1 \
--drop 0 \
--save_output
```

结果是在 output_encoder_py 文件夹里，生成了 channel_0.ts 到 channel_5.ts 文件，以及 channel_0_detection.txt 到 channel_5_detection.txt 文件。 ts 文件可以用 potplayer 播放

### decode_detection_writer_multi.py 脚本参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run [ default: 0 ]
  --threshold THRESHOLD
                        threshold for detection [default: 0.1]
  --uri URI             uri to decode
  --output_path OUTPUT_PATH
                        output path [ default: output_result ]
  --num_channels NUM_CHANNELS
                        number of channels to decode [ default: 1 ]
  --loop LOOP           loop count for each channel [ default: 1 ]
  --keep KEEP           keep num [default:1]
  --drop DROP           drop num [default:0]
  --uri_list URI_LIST   input uri list file
  --disable_encoder     enable encoder or not
  --save_output         save output or not
```

### decode_detection_writer_multi.py 脚本运行示例

```bash
python3 decode_detection_writer_multi.py \
-m /root/tools2/va1/build_model/deploy_weights/yolov5s_640/mod \
--vdsp_params ../../data/configs/yolo_div255_yuv_nv12.json \
--threshold 0.5 \
-d 0 \
--uri ../../data/videos/output_set.ts \
--output_path output_writer_py \
--num_channels 6 \
--keep 1 \
--drop 0 \
--save_output
```

结果是在 output_writer_py 文件夹里，生成了 channel_0.ts 到 channel_5.ts 文件，以及 channel_0_detection.txt 到 channel_5_detection.txt 文件。 ts 文件可以用 potplayer 播放
