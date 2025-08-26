# Decode  Detection Encode Sample

本目录展示如何实现  视频 解码 + AI + 编码 的功能

## C++ Sample 

### decode_detection_encode 命令参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=./data/configs/yolo_div255_yuv_nv12.json])
  -d, --device_id       device id to run (unsigned int [=0])
      --threshold       threshold for detection (float [=0.5])
      --label_file      label file (string [=../data/labels/coco2id.txt])
      --input_uri       input uri (string [=../data/videos/test.mp4])
      --output_file     output_file (string [=])
  -?, --help            print this message
```

### decode_detection_encode 命令示例

```bash
./vaststreamx-samples/bin/decode_detection_encode \
-m /opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../data/configs/yolo_div255_yuv_nv12.json \
--device_id  0 \
--threshold  0.5 \
--label_file ../data/labels/coco2id.txt \
--input_uri  ../data/videos/test.mp4 \
--output_file  out.mp4

#检测结果绘制与图片并保存于 dec_det_out
```


## Python Sample

### decode_detection_encode.py 脚本参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --threshold THRESHOLD
                        threshold for detection
  --uri URI             uri to decode
  --output_path OUTPUT_PATH
                        output path
  --num_channels NUM_CHANNELS
                        number of channels to decode
```
### decode_detection_encode.py 脚本运行示例

```bash
python3 decode_detection_encode.py \
-m /opt/vastai/vaststreamx/data/models/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline/mod \
--vdsp_params ../../data/configs/yolo_div255_yuv_nv12.json \
--device_id  0 \
--threshold  0.5 \
--label_file ../../data/labels/coco2id.txt \
--input_uri  ../../data/videos/test.mp4 \
--output_file  out.mp4
```
