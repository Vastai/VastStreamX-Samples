# Decode And Pose Sample

本目录展示如何实现  视频 解码 + AI 的功能

## C++ Sample 

### decode_and_pose 命令参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/yolov8_pose_int8/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/yolov8_pose_640_rgb888.json])
  -d, --device_id       device id to run (unsigned int [=0])
  -t, --threshold       threshold for pose (float [=0.1])
      --uri             uri to decode (string [=../data/videos/test.mp4])
      --output_path     output path (string [=])
      --num_channels    number of channles to decode (unsigned int [=1])
  -?, --help            print this message
```

### decode_and_pose 命令示例

```bash
mkdir -p dec_pose_out
./vaststreamx-samples/bin/decode_and_pose \
-m /work/data/deploy_weights/yolov8_pose_int8/mod \
--vdsp_params ../data/configs/yolo_div255_yuv_nv12.json \
--device_id 0 \
--threshold 0.5 \
--uri ../data/videos/test.mp4 \
--output_path dec_pose_out \
--num_channels 1

#检测结果绘制与图片并保存于 dec_pose_out
```
