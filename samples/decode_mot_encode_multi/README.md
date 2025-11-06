# Decode MOT Encode Multi Sample

本目录展示了 视频解码 + 行人跟踪 + 视频编码 sample

## C++ Sample

### decode_mot_encode_multi 命令参数说明

```bash
options:
  -m, --model_prefix       model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod])
      --vdsp_params        vdsp preprocess parameter file (string [=./data/configs/yolo_div255_yuv_nv12.json])
  -d, --device_id          device id to run (unsigned int [=0])
      --detect_thresh      threshold for detection (float [=0.5])
      --track_thresh       threshold for tracker (float [=0.6])
      --track_buffer       the frames for keep lost tracks (unsigned int [=30])
      --match_thresh       matching threshold for tracking (float [=0.9])
      --min_box_area       filter out tiny boxes (float [=100])
      --uri                input uri (string [=../data/videos/test.mp4])
      --output_path        output_path (string [=])
      --num_channels       number of channels to decode [ default: 1 ] (unsigned int [=1])
      --loop               loop count for each channel [ default: 1 ] (unsigned int [=1])
      --uri_list           input uri list file (string [=])
      --disable_encoder    enable encoder or not (unsigned int [=0])
  -?, --help               print this message
```

### decode_mot_encode_multi 命令示例

```bash
./vaststreamx-samples/bin/decode_mot_encode_multi \
-m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
--vdsp_params ../data/configs/bytetrack_yuv_nv12.json \
--device_id 0 \
--detect_thresh 0.01 \
--track_buffer 30 \
--track_thresh 0.6 \
--uri ../data/videos/test.mp4 \
--output_path output_mot_cpp \
--num_channels 6 
```

结果是在 output_mot_cpp 文件夹里，生成了 channel_0.ts 到 channel_5.ts 文件，以及 channel_0.txt 到 channel_5.txt 文件。 ts 文件可以用 potplayer 播放

## Python Sample

### decode_mot_encode_multi.py 脚本参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --detect_thresh DETECT_THRESH
                        detector threshold
  --track_thresh TRACK_THRESH
                        tracking confidence threshold
  --track_buffer TRACK_BUFFER
                        the frames for keep lost tracks
  --match_thresh MATCH_THRESH
                        matching threshold for tracking
  --min_box_area MIN_BOX_AREA
                        filter out tiny boxes
  --uri URI             uri to decode
  --output_path OUTPUT_PATH
                        output path [ default: output_result ]
  --num_channels NUM_CHANNELS
                        number of channels to decode [ default: 1 ]
  --loop LOOP           loop count for each channel [ default: 1 ]
  --uri_list URI_LIST   input uri list file
  --disable_encoder     enable encoder or not
  --save_output         save output or not
```

### decode_mot_encode_multi.py 脚本使用实例

```bash
python3 decode_mot_encode_multi.py \
-m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
--vdsp_params ../../data/configs/bytetrack_yuv_nv12.json \
--device_id 0 \
--detect_thresh 0.01 \
--track_buffer 30 \
--track_thresh 0.6 \
--uri ../../data/videos/test.mp4 \
--output_path output_mot_py \
--num_channels 6 \
--save_output 
```

结果是在 output_mot_py 文件夹里，生成了 channel_0.ts 到 channel_5.ts 文件，以及 channel_0.txt 到 channel_5.txt 文件。 ts 文件可以用 potplayer 播放
