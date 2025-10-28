# Decode  Detection Encode Sample

本目录展示如何实现视频 解码 + AI + 编码 的功能, 支持多路以及丢帧

注意：如果不希望丢帧，将 --drop 参数设置为 0 即可。

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
--output_path output_py \
--num_channels 10 \
--keep 1 \
--drop 0 \
--save_output
```
