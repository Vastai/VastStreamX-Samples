# PPOCR-V5 Samples

本文介绍 PPOCR-V5 用法以及 text_det & text_rec 模型精度性能测试方法

## 模型信息

### PPOCR-V5-DET

|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.5.0/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.md)  [modelzoo](https://github.com/Vastai/VastModelZOO/blob/main/cv/text_detection/dbnet/source_code/ppocr_v5_det.md)|
|  输入 shape |   (1,3,960,960)    |
| INT8量化方式 |   percentile         |
|  ONNX 精度(fixed shape) | {'precision': 0.7504, 'recall': 0.7922, 'hmean': 0.7707} |
|  VACC FP16  精度(mobile) | {'precision': 0.7545, 'recall': 0.7995, 'hmean': 0.7763} |
|  VACC INT8  精度(mobile)  |  - |

### PPOCR-V5-REC

|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.5.0/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.md)  [modelzoo](https://github.com/Vastai/VastModelZOO/blob/main/cv/text_recognition/ppocr_v5_rec/README.md)|
|  输入 shape |    (1,3,48,320)    |
| INT8量化方式 |   percentile          |
|  ONNX精度(fixed shape) | {'ExactMatch': 0.7548, 'CharMatch': 0.8759} |
|  VACC FP16  精度(mobile) | {'ExactMatch': 0.7871, 'CharMatch': 0.8979} |
|  VACC INT8  精度(mobile)  | - |

## 数据准备

### PPOCR-V5-DET

- 模型，请根据[modelzoo](https://github.com/Vastai/VastModelZOO/blob/main/cv/text_detection/dbnet/source_code/ppocr_v5_det.md)介绍来转模型
- 数据集，ppocr-v5.tar.gz

### PPOCR-V5-REC mobile

- 模型，请根据[modelzoo](https://github.com/Vastai/VastModelZOO/blob/main/cv/text_recognition/ppocr_v5_rec/README.md)介绍来转模型
- 数据集，ppocr-v5.tar.gz

## C++ Sample

### ocr_e2e 命令行参数说明

```bash
options:
      --det_model              text detection model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/det_model_vacc_fp16/mod])
      --det_config             text detection vdsp preprocess parameter file (string [=../data/configs/dbnet_rgbplanar.json])
      --cls_model              text classification model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/cls_model_vacc_fp16/mod])
      --cls_config             text classification vdsp preprocess parameter file (string [=../data/configs/crnn_rgbplanar.json])
      --rec_model              text recognition model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/rec_model_vacc_fp16/mod])
      --rec_config             text recognition vdsp preprocess parameter file (string [=../data/configs/crnn_rgbplanar.json])
      --det_box_type           text detection box type (string [=quad])
      --det_elf_file           text detection elf file (string [=/opt/vastai/vaststreamx/data/elf/find_contours_ext_op])
      --cls_labels             text classification label list (string [=[0, 180]])
      --cls_thresh             text classification thresh (float [=0.9])
      --rec_label_file         text recognition label file (string [=../data/labels/ppocr_keys_v1.txt])
      --rec_drop_score         text recogniztion drop score threshold (float [=0.5])
      --use_angle_cls          use text classification (bool [=1])
      --batch_size             batch size of the model (unsigned int [=1])
      --device_ids             device id to run (string [=[0]])
      --hw_config              hw-config file of the model suite (string [=])
      --input_file             input image (string [=../data/images/word_336.png])
      --output_file            output image file (string [=])
      --dataset_filelist       input dataset filelist (string [=])
      --dataset_root           input dataset root (string [=])
      --dataset_output_file    dataset output file (string [=dataset_output.txt])
  -?, --help                   print this message
```

### ocr_e2e 命令行示例

在build 目录里执行

单图片示例

```bash
./vaststreamx-samples/bin/ocr_e2e \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-det-fp16-none-1_3_960_960-vacc/mod \
--det_config ../data/configs/dbnet_rgbplanar.json \
--use_angle_cls 0 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-rec-fp16-none-1_3_48_320-vacc/mod \
--rec_config ../data/configs/crnn_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--rec_label_file ../data/labels/ppocrv5_dict.txt \
--rec_drop_score 0.5 \
--device_ids [0] \
--input_file ../data/images/detect.jpg \
--output_file ocr_e2e_result.jpg
```

输出

```bash
Thread 0 get ../data/images/detect.jpg result:
bbox:[ [653 78] [708 81] [704 100] [ 649 96] ], score: 0.995508, string: 20029
bbox:[ [633 132] [730 137] [726 159] [ 629 153] ], score: 0.998535, string: 97154197
bbox:[ [636 150] [668 153] [662 171] [ 630 168] ], score: 0.997721, string: 198
bbox:[ [665 156] [694 156] [694 169] [ 665 169] ], score: 0.992188, string: 727
bbox:[ [781 279] [910 282] [909 303] [ 780 300] ], score: 0.990039, string: Freeyourselfrom
bbox:[ [774 294] [922 291] [925 339] [ 777 342] ], score: 0.989648, string: JOINT
bbox:[ [777 330] [896 330] [896 371] [ 777 371] ], score: 0.984009, string: PAIN
bbox:[ [849 452] [908 452] [908 469] [ 849 469] ], score: 0.930969, string: JOINT-RY
Save file to: ./thread_0_ocr_e2e_result.jpg
```

并在图片上画出检测框，保存到  thread_0_ocr_e2e_result.jpg

测试 两个个模型同步推理 的性能与时延, 可以通过 --device_ids 指定多个 die

```bash
./vaststreamx-samples/bin/ocr_e2e \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-det-fp16-none-1_3_960_960-vacc/mod \
--det_config ../data/configs/dbnet_rgbplanar.json \
--use_angle_cls 0 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-rec-fp16-none-1_3_48_320-vacc/mod \
--rec_config ../data/configs/crnn_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--rec_label_file ../data/labels/ppocrv5_dict.txt \
--rec_drop_score 0.5 \
--device_ids [0] \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file ppocr_v4_dataset_output.txt

##结果示例  开启dpm 下
Image count: 500, total cost: 32992 ms, throughput: 15.1552 fps. Average latency: 65.984 ms.
```

### ocr_e2e_async 命令行参数说明

```bash
options:
      --det_model              text detection model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/det_model_vacc_fp16/mod])
      --det_config             text detection vdsp preprocess parameter file (string [=../data/configs/dbnet_rgbplanar.json])
      --cls_model              text classification model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/cls_model_vacc_fp16/mod])
      --cls_config             text classification vdsp preprocess parameter file (string [=../data/configs/crnn_rgbplanar.json])
      --rec_model              text recognition model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/rec_model_vacc_fp16/mod])
      --rec_config             text recognition vdsp preprocess parameter file (string [=../data/configs/crnn_rgbplanar.json])
      --det_box_type           text detection box type (string [=quad])
      --det_elf_file           text detection elf file (string [=/opt/vastai/vaststreamx/data/elf/find_contours_ext_op])
      --cls_labels             text classification label list (string [=[0, 180]])
      --cls_thresh             text classification thresh (float [=0.9])
      --rec_label_file         text recognition label file (string [=../data/labels/ppocr_keys_v1.txt])
      --rec_drop_score         text recogniztion drop score threshold (float [=0.5])
      --use_angle_cls          use text classification (int [=1])
      --batch_size             batch size of the model (unsigned int [=1])
      --device_ids             device id to run (string [=[0]])
      --hw_config              hw-config file of the model suite (string [=])
      --input_file             input image (string [=../data/images/word_336.png])
      --output_file            output image file (string [=])
      --dataset_filelist       input dataset filelist (string [=])
      --dataset_root           input dataset root (string [=])
      --dataset_output_file    dataset output file (string [=])
      --queue_size             set queue size (unsigned int [=1])
  -?, --help                   print this message
```

### ocr_e2e_async 命令行示例

在build 目录里执行

单图片示例

```bash
./vaststreamx-samples/bin/ocr_e2e_async \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-det-fp16-none-1_3_960_960-vacc/mod \
--det_config ../data/configs/dbnet_rgbplanar.json \
--use_angle_cls 0 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-rec-fp16-none-1_3_48_320-vacc/mod \
--rec_config ../data/configs/crnn_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--rec_label_file ../data/labels/ppocrv5_dict.txt \
--rec_drop_score 0.5 \
--device_ids [0] \
--queue_size 1 \
--input_file ../data/images/detect.jpg \
--output_file ocr_e2e_async_result.jpg
```

### ocr_e2e_async 结果示例

```bash
Thread 0 get ../data/images/detect.jpg result.
bbox:[ [653 78] [708 81] [704 100] [ 649 96] ], score: 0.995508, string: 20029
bbox:[ [633 132] [730 137] [726 159] [ 629 153] ], score: 0.998535, string: 97154197
bbox:[ [636 150] [668 153] [662 171] [ 630 168] ], score: 0.997721, string: 198
bbox:[ [665 156] [694 156] [694 169] [ 665 169] ], score: 0.992188, string: 727
bbox:[ [781 279] [910 282] [909 303] [ 780 300] ], score: 0.990039, string: Freeyourselfrom
bbox:[ [774 294] [922 291] [925 339] [ 777 342] ], score: 0.989648, string: JOINT
bbox:[ [777 330] [896 330] [896 371] [ 777 371] ], score: 0.984009, string: PAIN
bbox:[ [849 452] [908 452] [908 469] [ 849 469] ], score: 0.930969, string: JOINT-RY
Save file to: ./thread_0_ocr_e2e_async_result.jpg
```

测试 三个模型多线程异步推理 的性能与时延, 可以通过 --device_ids 指定多个 die

```bash
./vaststreamx-samples/bin/ocr_e2e_async \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-det-fp16-none-1_3_960_960-vacc/mod \
--det_config ../data/configs/dbnet_rgbplanar.json \
--use_angle_cls 0 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-rec-fp16-none-1_3_48_320-vacc/mod \
--rec_config ../data/configs/crnn_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--rec_label_file ../data/labels/ppocrv5_dict.txt \
--rec_drop_score 0.5 \
--device_ids [0] \
--queue_size 1 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/

##结果示例  开启dpm 下
Image count: 500, total cost: 12274 ms, throughput: 40.7365 fps. Average latency: 831.978 ms.
```

### text_det_prof 命令行参数说明

```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/dbnet_rgbplanar.json])
  -d, --device_ids      device id to run (string [=[0]])
      --elf_file        elf file path (string [=])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        model instance number (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (int [=1024])
      --percentiles     percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=1])
      --warmup_times    number of warmup iterations (unsigned int [=10])
  -?, --help            print this message
```

### text_det_prof 命令行示例

```bash
# 测试最大吞吐
./vaststreamx-samples/bin/text_det_prof \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-det-fp16-none-1_3_960_960-vacc/mod \
--vdsp_params ../data/configs/dbnet_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 600 \
--warmup_times 600 \
--shape "[3,900,900]" \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--input_host 1 \
--queue_size 1 

# 测试最小时延
./vaststreamx-samples/bin/text_det_prof \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-det-fp16-none-1_3_960_960-vacc/mod \
--vdsp_params ../data/configs/dbnet_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 300 \
--warmup_times 600 \
--shape "[3,900,900]" \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--input_host 1 \
--queue_size 0

```

### text_det_prof 命令行结果示例

```bash
# 本结果在 dpm下 OCLK 1025MHz 下测试所得
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 96.897
  latency (us):
    avg latency: 30751
    min latency: 24279
    max latency: 49591
    p50 latency: 30770
    p90 latency: 32501
    p95 latency: 33559
    p99 latency: 35731


# 本结果在 dpm下 OCLK 1025MHz 下测试所得
# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 41.5282
  latency (us):
    avg latency: 24078
    min latency: 20607
    max latency: 30254
    p50 latency: 24059
    p90 latency: 26024
    p95 latency: 26378
    p99 latency: 27389
```

### text_rec_prof 命令行参数说明

```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/crnn_rgbplanar.json])
  -d, --device_ids      device id to run (string [=[0]])
      --label_file      label file (string [=../data/labels/key_37.txt])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        model instance number (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (int [=10240])
      --percentiles     percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=1])
      --warmup_times    number of warmup iterations (unsigned int [=10])
  -?, --help            print this message
```

### text_rec_prof 命令行示例

```bash
# 测试最大吞吐
./vaststreamx-samples/bin/text_rec_prof \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-rec-fp16-none-1_3_48_320-vacc/mod \
--vdsp_params ../data/configs/crnn_rgbplanar.json \
--device_ids [0] \
--label_file ../data/labels/ppocrv5_dict.txt \
--batch_size 1 \
--instance 3 \
--shape "[3,48,320]" \
--iterations 2000 \
--warmup_times 600 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1


# 测试最小时延
./vaststreamx-samples/bin/text_rec_prof \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-rec-fp16-none-1_3_48_320-vacc/mod \
--vdsp_params ../data/configs/crnn_rgbplanar.json \
--device_ids [0] \
--label_file ../data/labels/ppocrv5_dict.txt \
--batch_size 1 \
--instance 1 \
--shape "[3,48,320]" \
--iterations 500 \
--warmup_times 600 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0

```

### text_rec_prof 命令行结果示例

```bash
# 本结果在 dpm下 OCLK 1025MHz 下测试所得
# 测试最大吞吐
- number of instances: 3
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 459.375
  latency (us):
    avg latency: 19210
    min latency: 8526
    max latency: 31255
    p50 latency: 19346
    p90 latency: 24696
    p95 latency: 25860
    p99 latency: 27736



# 本结果在 dpm下 OCLK 1025MHz 下测试所得
# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 117.172
  latency (us):
    avg latency: 8533
    min latency: 7739
    max latency: 14057
    p50 latency: 8432
    p90 latency: 8640
    p95 latency: 8709
    p99 latency: 12185
```

### text_det 命令行参数说明

```bash
options:
  -m, --model_prefix             model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod])
      --hw_config                hw-config file of the model suite (string [=])
      --vdsp_params              vdsp preprocess parameter file (string [=../data/configs/dbnet_rgbplanar.json])
  -d, --device_id                device id to run (unsigned int [=0])
      --threshold                threshold for detection (float [=0.3])
      --box_threshold            threshold for boxes (float [=0.6])
      --box_unclip_ratio         unclip ratio (float [=1.5])
      --use_polygon_score        use_polygon_score in postprocess (bool [=0])
      --elf_file                 elf file path (string [=/opt/vastai/vaststreamx/data/elf/find_contours_ext_op])
      --input_file               input image file (string [=../data/images/detect.jpg])
      --output_file              output image file (string [=])
      --dataset_filelist         input dataset filelist (string [=])
      --dataset_root             input dataset root (string [=])
      --dataset_output_folder    dataset output folder (string [=])
  -?, --help                     print this message
```

### text_det 命令行示例

在build 目录里执行
单图片示例

```bash
./vaststreamx-samples/bin/text_det \
--model_prefix /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-det-fp16-none-1_3_960_960-vacc/mod \
--vdsp_params ../data/configs/dbnet_rgbplanar.json \
--device_id 0 \
--threshold 0.3 \
--box_unclip_ratio 1.5 \
--use_polygon_score 0 \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--input_file ../data/images/detect.jpg \
--output_file text_det_result.jpg
```

输出

```bash
index:0, score:0.765429,bbox:[ [933 71] [945 71] [945 80] [933 80] ]
index:1, score:0.819702,bbox:[ [656 78] [705 82] [701 99] [652 96] ]
index:2, score:0.699494,bbox:[ [634 97] [725 103] [722 116] [632 110] ]
index:3, score:0.812828,bbox:[ [636 133] [728 138] [724 157] [632 152] ]
index:4, score:0.787996,bbox:[ [637 150] [666 153] [661 171] [632 168] ]
index:5, score:0.69045,bbox:[ [666 156] [693 156] [693 168] [666 168] ]
index:6, score:0.61548,bbox:[ [914 251] [966 252] [966 261] [914 259] ]
index:7, score:0.781122,bbox:[ [785 282] [906 284] [905 300] [784 298] ]
index:8, score:0.883249,bbox:[ [782 297] [916 296] [917 336] [784 338] ]
index:9, score:0.782801,bbox:[ [782 333] [890 333] [890 368] [782 368] ]
index:10, score:0.67182,bbox:[ [876 339] [932 341] [928 363] [872 361] ]
index:11, score:0.670831,bbox:[ [924 455] [944 455] [944 461] [924 461] ]
index:12, score:0.820915,bbox:[ [852 453] [905 453] [905 468] [852 468] ]
index:13, score:0.749038,bbox:[ [845 534] [884 531] [888 542] [848 545] ]
```

并在图片上画出检测框，保存到  text_det_result.jpg

测试数据集

```bash
mkdir -p text_det_output
./vaststreamx-samples/bin/text_det \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-det-fp16-none-1_3_960_960-vacc/mod \
--vdsp_params ../data/configs/dbnet_rgbplanar.json \
--device_id 0 \
--threshold 0.3 \
--box_unclip_ratio 1.5 \
--use_polygon_score 0 \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ppocr-v5/det_test_list.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ppocr-v5/ \
--dataset_output_folder text_det_output
```

结果保存在 text_det_output 文件夹里

统计精度

```bash
python3 ../evaluation/ppocr-v5/det_eval.py \
--test_image_path  /opt/vastai/vaststreamx/data/datasets/ppocr-v5/det_test \
--boxes_npz_dir ./text_det_output 
```

精度结果

```
metric:  {'precision': 0.7538, 'recall': 0.7987, 'hmean': 0.7756}
```

### text_rec 命令行参数说明

```bash
options:
  -m, --model_prefix             model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod])
      --hw_config                hw-config file of the model suite (string [=])
      --vdsp_params              vdsp preprocess parameter file (string [=../data/configs/dbnet_rgbplanar.json])
  -d, --device_id                device id to run (unsigned int [=0])
      --threshold                threshold for detection (float [=0.3])
      --box_threshold            threshold for boxes (float [=0.6])
      --box_unclip_ratio         unclip ratio (float [=1.5])
      --use_polygon_score        use_polygon_score in postprocess (bool [=0])
      --elf_file                 elf file path (string [=/opt/vastai/vaststreamx/data/elf/find_contours_ext_op])
      --input_file               input image file (string [=../data/images/detect.jpg])
      --output_file              output image file (string [=])
      --dataset_filelist         input dataset filelist (string [=])
      --dataset_root             input dataset root (string [=])
      --dataset_output_folder    dataset output folder (string [=])
  -?, --help                     print this message
```

### text_rec 命令行示例

在build 目录里执行
单图片示例

```bash
./vaststreamx-samples/bin/text_rec \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-rec-fp16-none-1_3_48_320-vacc/mod \
--vdsp_params ../data/configs/ppocr-v4-rec-vdsp_params.json \
--device_id 0 \
--label_file ../data/labels/ppocrv5_dict.txt \
--input_file ../data/images/word_336.png 
```

输出

```bash
score: 0.957715
text: SUPER
```

测试数据集

```bash
./vaststreamx-samples/bin/text_rec \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-rec-fp16-none-1_3_48_320-vacc/mod \
--vdsp_params ../data/configs/crnn_rgbplanar.json \
--device_id 0 \
--label_file ../data/labels/ppocrv5_dict.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ppocr-v5/rec_test_list.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ppocr-v5/ \
--dataset_output_file rec_pred.txt
```

统计精度

```bash
python3 ../evaluation/ppocr-v5/rec_eval.py \
--test_image_path /opt/vastai/vaststreamx/data/datasets/ppocr-v5/rec_test \
--pred_file rec_pred.txt
```

精度结果

```
metric:  {'ExactMatch': 0.7226, 'CharMatch': 0.861}
```

## Python sample

### ocr_e2e.py 命令行参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  --det_model DET_MODEL
                        text detection model prefix of the model suite files
  --det_vdsp_params DET_VDSP_PARAMS
                        text detection vdsp preprocess parameter file
  --det_box_type DET_BOX_TYPE
                        det box type, poly or quad
  --det_elf_file DET_ELF_FILE
                        input file
  --cls_model CLS_MODEL
                        text detection model prefix of the model suite files
  --cls_vdsp_params CLS_VDSP_PARAMS
                        text detection vdsp preprocess parameter file
  --cls_label_list CLS_LABEL_LIST
                        text classification label list
  --cls_thresh CLS_THRESH
                        text classification thresh
  --rec_model REC_MODEL
                        text detection model prefix of the model suite files
  --rec_vdsp_params REC_VDSP_PARAMS
                        text detection vdsp preprocess parameter file
  --rec_label_file REC_LABEL_FILE
                        text recognizition label file
  --rec_drop_score REC_DROP_SCORE
                        text recogniztion drop score threshold
  --use_angle_cls USE_ANGLE_CLS
                        whether use angle classifier
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  --device_ids DEVICE_IDS
                        device ids to run
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file
  --dataset_filelist DATASET_FILELIST
                        dataset filelist
  --dataset_root DATASET_ROOT
                        input dataset root
  --dataset_output_file DATASET_OUTPUT_FILE
                        dataset output file
```

### ocr_e2e.py 运行示例

在本目录下运行  

```bash
#单张图片示例
python3 ocr_e2e.py \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-det-fp16-none-1_3_960_960-vacc/mod \
--det_vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--use_angle_cls 0 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-rec-fp16-none-1_3_48_320-vacc/mod \
--rec_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_ids [0] \
--rec_label_file ../../data/labels/ppocrv5_dict.txt \
--input_file ../../data/images/detect.jpg \
--det_box_type quad \
--output_file ocr_res.jpg

```

### ocr_e2e.py 运行结果示例

终端显示 检测到的文字的 bbox 多边形的四个角的坐标，文本内容，识别分数，bbox也画在图片上并保存为 ocr_res.jpg

```bash
#单张图片结果示例
[[656,79], [705,82], [702,100], [652,96]],  [('20029', 0.9951171875)]
[[635,98], [725,103], [722,116], [633,111]],  [(' ', 0.55224609375)]
[[635,134], [728,138], [725,157], [632,153]],  [('97154197', 0.99853515625)]
[[637,151], [667,154], [662,171], [632,168]],  [('198', 0.99755859375)]
[[667,157], [693,157], [693,169], [667,169]],  [('727', 0.99072265625)]
[[785,282], [907,284], [906,301], [784,298]],  [('Free yourself from', 0.95947265625)]
[[782,298], [916,296], [918,336], [784,339]],  [('JOINT', 0.9951171875)]
[[783,334], [891,334], [891,368], [783,368]],  [('PAIN', 0.9794921875)]
[[852,454], [905,454], [905,468], [852,468]],  [('JOINT-RY', 0.89990234375)]
[[846,534], [885,531], [887,543], [849,546]],  [('TUFBAN', 0.56591796875)]
save file  thread_0_ocr_res.jpg
```

并在图片上画出检测框，保存到  thread_0_ocr_res.jpg

### ocr_e2e.py 测试 同步推理 性能与时延

```bash
python ocr_e2e.py \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-det-fp16-none-1_3_960_960-vacc/mod \
--det_vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--use_angle_cls 0 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-rec-fp16-none-1_3_48_320-vacc/mod \
--rec_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_ids [0] \
--det_box_type quad \
--rec_label_file ../../data/labels/ppocrv5_dict.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file ppocr_v4_dataset_output.txt
#测试结果  在开启dpm 下
Image count: 500, total cost: 47.68 s, throughput: 10.49 fps, average latency: 0.095 s
```

### ocr_e2e_async.py 命令行参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  --det_model DET_MODEL
                        text detection model prefix of the model suite files
  --det_vdsp_params DET_VDSP_PARAMS
                        text detection vdsp preprocess parameter file
  --det_box_type DET_BOX_TYPE
                        det box type, poly or quad
  --det_elf_file DET_ELF_FILE
                        input file
  --cls_model CLS_MODEL
                        text detection model prefix of the model suite files
  --cls_vdsp_params CLS_VDSP_PARAMS
                        text detection vdsp preprocess parameter file
  --cls_label_list CLS_LABEL_LIST
                        text classification label list
  --cls_thresh CLS_THRESH
                        text classification thresh
  --rec_model REC_MODEL
                        text detection model prefix of the model suite files
  --rec_vdsp_params REC_VDSP_PARAMS
                        text detection vdsp preprocess parameter file
  --rec_label_file REC_LABEL_FILE
                        text recognizition label file
  --rec_drop_score REC_DROP_SCORE
                        text recogniztion drop score threshold
  --use_angle_cls USE_ANGLE_CLS
                        whether use angle classifier
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  --device_ids DEVICE_IDS
                        device ids to run
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file
  --dataset_filelist DATASET_FILELIST
                        dataset filelist
  --dataset_root DATASET_ROOT
                        input dataset root
  --dataset_output_file DATASET_OUTPUT_FILE
                        dataset output file
```

### ocr_e2e_async.py 命令行示例

```bash
# 测试单张图片
python ocr_e2e_async.py \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-det-fp16-none-1_3_960_960-vacc/mod \
--det_vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--use_angle_cls 0 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-rec-fp16-none-1_3_48_320-vacc/mod \
--rec_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_ids [0] \
--det_box_type quad \
--rec_label_file ../../data/labels/ppocrv5_dict.txt \
--input_file ../../data/images/detect.jpg \
--output_file ocr_res.jpg

#结果示例

Thread:0,Get ../../data/images/detect.jpg result
[[656,79], [705,82], [702,100], [652,96]],  [('20029', 0.9951171875)]
[[635,98], [725,103], [722,116], [633,111]],  [(' ', 0.55224609375)]
[[635,134], [728,138], [725,157], [632,153]],  [('97154197', 0.99853515625)]
[[637,151], [667,154], [662,171], [632,168]],  [('198', 0.99755859375)]
[[667,157], [693,157], [693,169], [667,169]],  [('727', 0.99072265625)]
[[785,282], [907,284], [906,301], [784,298]],  [('Free yourself from', 0.95947265625)]
[[782,298], [916,296], [918,336], [784,339]],  [('JOINT', 0.9951171875)]
[[783,334], [891,334], [891,368], [783,368]],  [('PAIN', 0.9794921875)]
[[852,454], [905,454], [905,468], [852,468]],  [('JOINT-RY', 0.89990234375)]
[[846,534], [885,531], [887,543], [849,546]],  [('TUFBAN', 0.56591796875)]
save file to thread_0_ocr_res.jpg
```

### ocr_e2e_async.py 测试多线程异步推理 性能与时延

```bash
# 测试多线程异步
python ocr_e2e_async.py \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-det-fp16-none-1_3_960_960-vacc/mod \
--det_vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--use_angle_cls 0 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-rec-fp16-none-1_3_48_320-vacc/mod \
--rec_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_ids [0] \
--det_box_type quad \
--rec_label_file ../../data/labels/ppocrv5_dict.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/

#测试结果  880MHz 下
Image count: 500, total cost: 26.54 s, throughput: 18.84 fps, average latency: 5.275 s
```

### text_det_prof.py 命令行参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  --elf_file ELF_FILE   input file
  -d DEVICE_IDS, --device_ids DEVICE_IDS
                        device ids to run
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        profiling batch size of the model
  -i INSTANCE, --instance INSTANCE
                        model instance number
  -s SHAPE, --shape SHAPE
                        model input shape
  --iterations ITERATIONS
                        iterations count for one profiling
  --queue_size QUEUE_SIZE
                        aync wait queue size
  --percentiles PERCENTILES
                        percentiles of latency
  --input_host INPUT_HOST
                        cache input data into host memory
  --warmup_times WARMUP_TIMES
                        number of warmup iterations
```

### text_det_prof.py 运行示例

在本目录下运行  

```bash
# 测试最大吞吐
python3 text_det_prof.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-det-fp16-none-1_3_960_960-vacc/mod \
--vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_ids [0]  \
--batch_size 1 \
--instance 1 \
--shape "[3,960,960]" \
--iterations 500 \
--warmup_times 400 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1


# 测试最小时延
python3 text_det_prof.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-det-fp16-none-1_3_960_960-vacc/mod \
--vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_ids [0]  \
--batch_size 1 \
--instance 1 \
--shape "[3,960,960]" \
--iterations 300 \
--warmup_times 400 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### text_det_prof.py 运行结果示例

```bash
# 本结果在 dpm下 OCLK 1025MHz 下测试所得
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 97.47
  latency (us):
    avg latency: 30628
    min latency: 24158
    max latency: 42903
    p50 latency: 30641
    p90 latency: 32681
    p95 latency: 33281
    p99 latency: 34130

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 46.25
  latency (us):
    avg latency: 21617
    min latency: 17613
    max latency: 28243
    p50 latency: 21784
    p90 latency: 24327
    p95 latency: 25352
    p99 latency: 27096
```

### text_rec_prof.py 命令行参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  -d DEVICE_IDS, --device_ids DEVICE_IDS
                        device ids to run
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        profiling batch size of the model
  -i INSTANCE, --instance INSTANCE
                        model instance number
  --label_file LABEL_FILE
                        label file
  -s SHAPE, --shape SHAPE
                        model input shape
  --iterations ITERATIONS
                        iterations count for one profiling
  --queue_size QUEUE_SIZE
                        aync wait queue size
  --percentiles PERCENTILES
                        percentiles of latency
  --input_host INPUT_HOST
                        cache input data into host memory
  --warmup_times WARMUP_TIMES
                        number of warmup iterations
```

### text_rec_prof.py 运行示例

在本目录下运行  

```bash
# 测试最大吞吐
python3 text_rec_prof.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-rec-fp16-none-1_3_48_320-vacc/mod \
--vdsp_params ../../data/configs/crnn_rgbplanar.json \
--device_ids [0]  \
--batch_size 1 \
--instance 4 \
--label_file ../../data/labels/ppocrv5_dict.txt \
--shape "[3,48,320]" \
--iterations 3000 \
--warmup_times 400 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1


# 测试最小时延
python3 text_rec_prof.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-rec-fp16-none-1_3_48_320-vacc/mod \
--vdsp_params ../../data/configs/crnn_rgbplanar.json \
--device_ids [0]  \
--batch_size 1 \
--instance 3 \
--label_file ../../data/labels/ppocrv5_dict.txt \
--shape "[3,48,320]" \
--iterations 500 \
--warmup_times 400 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```

### text_rec_prof.py 运行结果示例

```bash
# 本结果在 dpm下 OCLK 1025MHz 下测试所得
# 测试最大吞吐
- number of instances: 4
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 455.65
  latency (us):
    avg latency: 26185
    min latency: 15351
    max latency: 47954
    p50 latency: 25983
    p90 latency: 31582
    p95 latency: 33551
    p99 latency: 36579

# 测试最小时延
- number of instances: 3
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 169.91
  latency (us):
    avg latency: 17654
    min latency: 15814
    max latency: 27934
    p50 latency: 17684
    p90 latency: 18277
    p95 latency: 19638
    p99 latency: 23618
```

### text_det.py 命令行参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --elf_file ELF_FILE   input file
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file
  --dataset_filelist DATASET_FILELIST
                        input dataset image list
  --dataset_root DATASET_ROOT
                        input dataset root
  --dataset_output_folder DATASET_OUTPUT_FOLDER
                        dataset output folder
```

### text_det.py 运行示例

在本目录下运行  

```bash
python3 text_det.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-det-fp16-none-1_3_960_960-vacc/mod \
--vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_id 0 \
--input_file ../../data/images/detect.jpg \
--output_file text_det_result.jpg

```

### text_det.py 运行结果示例

终端显示 检测到的文字的 bbox 多边形的四个角的坐标，bbox也画在图片上并保存为 text_det_result.jpg

```bash
index:0, score:0.7654287550184461,bbox:[[933  71],[945  71],[945  80],[933  80]]
index:1, score:0.8197022332085504,bbox:[[656  79],[705  82],[702 100],[652  96]]
index:2, score:0.699494037222355,bbox:[[635  98],[725 103],[722 116],[633 111]]
index:3, score:0.8128281873988077,bbox:[[635 134],[728 138],[725 157],[632 153]]
index:4, score:0.7879960683470998,bbox:[[637 151],[667 154],[662 171],[632 168]]
index:5, score:0.6904495597904564,bbox:[[667 157],[693 157],[693 169],[667 169]]
index:6, score:0.6154796412733735,bbox:[[915 251],[967 253],[966 261],[914 259]]
index:7, score:0.7811218425884657,bbox:[[785 282],[907 284],[906 301],[784 298]]
index:8, score:0.8832487357679263,bbox:[[782 298],[916 296],[918 336],[784 339]]
index:9, score:0.7828013907665613,bbox:[[783 334],[891 334],[891 368],[783 368]]
index:10, score:0.6718201371221584,bbox:[[875 339],[931 342],[929 363],[873 361]]
index:11, score:0.670831044514974,bbox:[[924 455],[944 455],[944 461],[924 461]]
index:12, score:0.8209148160872921,bbox:[[852 454],[905 454],[905 468],[852 468]]
index:13, score:0.749038332984561,bbox:[[846 534],[885 531],[887 543],[849 546]]
```

测试数据集

```bash
mkdir -p text_det_output
python3 text_det.py  \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-det-fp16-none-1_3_960_960-vacc/mod \
--vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ppocr-v5/det_test_list.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ppocr-v5/ \
--dataset_output_folder text_det_output
```

结果保存在 text_det_output 文件夹里

```bash
# 用刚才保存的npz文件测试精度
python3 ../../evaluation/ppocr-v5/det_eval.py \
--test_image_path  /opt/vastai/vaststreamx/data/datasets/ppocr-v5/det_test \
--boxes_npz_dir ./text_det_output 
```

精度结果

```
metric:  {'precision': 0.7545, 'recall': 0.7995, 'hmean': 0.7763}
```

### text_rec.py 命令行参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --label_file LABEL_FILE
                        label file
  --input_file INPUT_FILE
                        input file
  --dataset_filelist DATASET_FILELIST
                        dataset filelist
  --dataset_root DATASET_ROOT
                        input dataset root
  --dataset_output_file DATASET_OUTPUT_FILE
                        dataset output file
```

### text_rec.py 运行示例

在本目录下运行  

```bash
#单张图片示例
python3 text_rec.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-rec-fp16-none-1_3_48_320-vacc/mod \
--vdsp_params ../../data/configs/crnn_rgbplanar.json \
--device_id 0 \
--label_file ../../data/labels/ppocrv5_dict.txt \
--input_file ../../data/images/word_336.png 

#数据集示例
python3 text_rec.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5/mobile-rec-fp16-none-1_3_48_320-vacc/mod \
--vdsp_params ../../data/configs/crnn_rgbplanar.json \
--device_id 0 \
--label_file ../../data/labels/ppocrv5_dict.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ppocr-v5/rec_test_list.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ppocr-v5/ \
--dataset_output_file rec_pred.txt

# 统计精度
python3 ../../evaluation/ppocr-v5/rec_eval.py \
--test_image_path /opt/vastai/vaststreamx/data/datasets/ppocr-v5/rec_test \
--pred_file rec_pred.txt
```

### text_rec.py 运行结果示例

```bash
#单张图片结果示例
[('SUPER', 0.95751953125)]

#统计精度结果示例
metric:  {'ExactMatch': 0.7032, 'CharMatch': 0.8458}
```
