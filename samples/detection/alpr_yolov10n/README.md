# ALPR

本目录提供基于 yolov10n 模型和 OCRv4 模型的车牌识别 sample

## 数据准备

下载模型 yolov10n_alpr 到 /opt/vastai/vaststreamx/data/models 里并解压
下载模型 PP-OCRv4_rec 到 /opt/vastai/vaststreamx/data/models 里并解压

## C++ sample

### detection 命令行参数说明

```bash
options:
      --yolov10_model_prefix    yolov10 model prefix of the model suite files (string [=/home/aico/Downloads/docker/yolov10n/deploy_weights/yolov10n_alpr/mod])
      --ocrv4_model_prefix      ocrv4 model prefix of the model suite files (string [=/home/aico/Downloads/docker/ocrv4/deploy_weights/PP-OCRv4_rec_infer/mod])
      --hw_config               hw-config file of the model suite (string [=])
      --yolov10_vdsp_params     vdsp preprocess parameter file (string [=/home/aico/Downloads/docker/yolov10n/official-yolov10n-vdsp_params.json])
      --ocrv4_vdsp_params       ocrv4 vdsp preprocess parameter file (string [=/home/aico/Downloads/docker/ocrv4/ppocr-v4-rec-vdsp_params.json])
  -d, --device_id               device id to run (unsigned int [=0])
  -t, --threshold               threshold for result (float [=0.25])
      --yolov10_label_file      label file (string [=../data/labels/alpr.txt])
      --ocrv4_label_file        label file (string [=/home/aico/Downloads/docker/ocrv4/ppocr_keys_v1.txt])
      --input_file              input file (string [=/home/aico/Downloads/images/double_yellow.jpg])
      --output_file             output file (string [=result.png])
  -?, --help                    print this message
```

### detection 运行示例

在 build 目录里执行

```bash
#跑单张图片
./vaststreamx-samples/bin/alpr \
--yolov10_model_prefix /opt/vastai/vaststreamx/data/models/yolov10n_alpr/mod \
--ocrv4_model_prefix /opt/vastai/vaststreamx/data/models/PP-OCRv4_rec_infer/mod \
--yolov10_vdsp_params ../data/configs/official-yolov10n-vdsp_params.json \
--ocrv4_vdsp_params ../data/configs/ppocr-v4-rec-vdsp_params.json \
--device_id 0 \
--threshold 0.25 \
--yolov10_label_file ../data/labels/alpr.txt \
--ocrv4_label_file ../data/labels/ppocr_keys_v1.txt \
--input_file ../data/images/double_yellow.jpg \
--output_file result.png
```
