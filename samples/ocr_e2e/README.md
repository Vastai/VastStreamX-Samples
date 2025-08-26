# OCR_e2e SAMPLE

本目录提供端到端（文本检测，文本方向分类，文本识别）的 OCR sample

其中 ocr_e2e 是 三个模型同步推理,  ocr_e2e_async是三个模型多线程异步推理       
    ocr_e2e.py 是 三个模型同步推理,  ocr_e2e_async.py  是三个模型多线程异步推理          



## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/ppocr/blog/PP-OCRv4_introduction.md)  [modelzoo](-) |
|  输入 shape |   [ (1,3,736,1280) (1,3,48,320) ]     |
| INT8量化方式 |   -          |
|  官方精度 | "HMEAN": - , "ACC": - |
|  VACC FP16  精度 | "HMEAN": 47.3 , "ACC": 80.9 |
|  VACC INT8  精度 | "HMEAN": - , "ACC": - |


## 数据准备

下载模型 ppocr_v4.tar.gz 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 ch4_test_images 到 /opt/vastai/vaststreamx/data/datasets 里
下载 elf 压缩包到 /opt/vastai/vaststreamx/data/


## C++ sample

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
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
--det_config ../data/configs/dbnet_rgbplanar.json \
--cls_model  /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod  \
--cls_config ../data/configs/crnn_rgbplanar.json \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
--rec_config ../data/configs/crnn_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--rec_label_file ../data/labels/ocr_rec_dict.txt \
--rec_drop_score 0.5 \
--use_angle_cls 1 \
--device_ids [0] \
--input_file ../data/images/detect.jpg \
--output_file ocr_e2e_result.jpg
```
输出
```bash
bbox:[ [659 79] [702 81] [701 100] [ 658 98] ], score: 0.998438, string: 20029
bbox:[ [636 133] [726 138] [724 159] [ 635 154] ], score: 0.99823, string: 97154197
bbox:[ [636 151] [701 154] [700 173] [ 635 170] ], score: 0.998291, string: 198727
bbox:[ [784 279] [907 282] [907 304] [ 784 301] ], score: 0.986298, string: Freeyourselffrom
bbox:[ [788 298] [902 298] [902 338] [ 788 338] ], score: 0.994727, string: JOINT
bbox:[ [787 330] [869 332] [868 370] [ 786 368] ], score: 0.987061, string: PAIN
bbox:[ [852 452] [904 450] [905 469] [ 853 471] ], score: 0.923828, string: JOINT-RX
bbox:[ [846 531] [883 529] [884 544] [ 847 546] ], score: 0.746053, string: TUFBLN
Save file to: /thread_0_ocr_e2e_result.jpg
```
并在图片上画出检测框，保存到  thread_0_ocr_e2e_result.jpg


测试 三个模型同步推理 的性能与时延, 可以通过 --device_ids 指定多个 die
```bash
./vaststreamx-samples/bin/ocr_e2e \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
--det_config ../data/configs/dbnet_rgbplanar.json \
--cls_model  /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod  \
--cls_config ../data/configs/crnn_rgbplanar.json \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
--rec_config ../data/configs/crnn_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--rec_label_file ../data/labels/ocr_rec_dict.txt \
--rec_drop_score 0.5 \
--use_angle_cls 1 \
--device_ids [0] \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file ppocr_v4_dataset_output.txt

##结果示例  880MHz 下
Image count: 500, total cost: 27183 ms, throughput: 18.3938 fps. Average latency: 54.366 ms.
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
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
--det_config ../data/configs/dbnet_rgbplanar.json \
--cls_model  /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod  \
--cls_config ../data/configs/crnn_rgbplanar.json \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
--rec_config ../data/configs/crnn_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--rec_label_file ../data/labels/ocr_rec_dict.txt \
--rec_drop_score 0.5 \
--use_angle_cls 1 \
--device_ids [0] \
--queue_size 1 \
--input_file ../data/images/detect.jpg \
--output_file ocr_e2e_async_result.jpg
```

### ocr_e2e_async 结果示例
```bash
bbox:[ [659 79] [702 81] [701 100] [ 658 98] ], score: 0.998438, string: 20029
bbox:[ [636 133] [726 138] [724 159] [ 635 154] ], score: 0.99823, string: 97154197
bbox:[ [636 151] [701 154] [700 173] [ 635 170] ], score: 0.998291, string: 198727
bbox:[ [784 279] [907 282] [907 304] [ 784 301] ], score: 0.986298, string: Freeyourselffrom
bbox:[ [788 298] [902 298] [902 338] [ 788 338] ], score: 0.994727, string: JOINT
bbox:[ [787 330] [869 332] [868 370] [ 786 368] ], score: 0.987061, string: PAIN
bbox:[ [852 452] [904 450] [905 469] [ 853 471] ], score: 0.923828, string: JOINT-RX
bbox:[ [846 531] [883 529] [884 544] [ 847 546] ], score: 0.746053, string: TUFBLN
Save file to: /thread_0_ocr_e2e_async_result.jpg
```


测试 三个模型多线程异步推理 的性能与时延, 可以通过 --device_ids 指定多个 die

```bash
./vaststreamx-samples/bin/ocr_e2e_async \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
--det_config ../data/configs/dbnet_rgbplanar.json \
--cls_model  /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod  \
--cls_config ../data/configs/crnn_rgbplanar.json \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
--rec_config ../data/configs/crnn_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--rec_label_file ../data/labels/ocr_rec_dict.txt \
--rec_drop_score 0.5 \
--use_angle_cls 1 \
--device_ids [0] \
--queue_size 1 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/

##结果示例  880MHz 下
Image count: 500, total cost: 16952 ms, throughput: 29.495 fps. Average latency: 974.576 ms.
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
  -i, --instance        instance number for each device (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (int [=1024])
      --percentiles     percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=1])
  -?, --help            print this message
```
### text_det_prof 命令行示例

```bash
# 测试最大吞吐
./vaststreamx-samples/bin/text_det_prof \
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
--vdsp_params ../data/configs/dbnet_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 600 \
--shape "[3,736,1280]" \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/text_det_prof \
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
--vdsp_params ../data/configs/dbnet_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 300 \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--input_host 1 \
--queue_size 0

```

### text_det_prof 命令行结果示例

```bash
# 本结果在 OCLK 880MHz 下测试所得
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 91.4363
  latency (us):
    avg latency: 32636
    min latency: 19181
    max latency: 43072
    p50 latency: 32624
    p90 latency: 32957
    p95 latency: 33062
    p99 latency: 33267


# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 55.3135
  latency (us):
    avg latency: 18077
    min latency: 17934
    max latency: 21011
    p50 latency: 18059
    p90 latency: 18146
    p95 latency: 18157
    p99 latency: 18530
```

### text_cls_prof 命令行参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/cls_model_vacc_fp16/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/crnn_rgbplanar.json])
  -d, --device_ids      device id to run (string [=[0]])
      --elf_file        elf file path (string [=])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        instance number for each device (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (int [=1024])
      --percentiles     percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=1])
  -?, --help            print this message
```
### text_cls_prof 命令行示例

```bash
# 测试最大吞吐
./vaststreamx-samples/bin/text_cls_prof \
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod \
--vdsp_params ../data/configs/crnn_rgbplanar.json \
--device_ids [0] \
--batch_size 32 \
--instance 1 \
--iterations 600 \
--shape "[3,48,192]" \
--input_host 1 \
--queue_size 1


# 测试最小时延
./vaststreamx-samples/bin/text_cls_prof \
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod \
--vdsp_params ../data/configs/dbnet_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 6000 \
--input_host 1 \
--queue_size 0

```

### text_cls_prof 命令行结果示例

```bash
# 本结果在 OCLK 880MHz 下测试所得
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 32
  throughput (qps): 1858.37
  latency (us):
    avg latency: 51485
    min latency: 21194
    max latency: 55351
    p50 latency: 51543
    p90 latency: 51660
    p95 latency: 51676
    p99 latency: 51706


# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 981.215
  latency (us):
    avg latency: 1018
    min latency: 955
    max latency: 1239
    p50 latency: 1019
    p90 latency: 1022
    p95 latency: 1023
    p99 latency: 1027
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
  -i, --instance        instance number for each device (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (int [=10240])
      --percentiles     percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=1])
  -?, --help            print this message
```
### text_rec_prof 命令行示例

```bash
# 测试最大吞吐
./vaststreamx-samples/bin/text_rec_prof \
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
--vdsp_params ../data/configs/crnn_rgbplanar.json \
--device_ids [0] \
--label_file ../data/labels/ocr_rec_dict.txt \
--batch_size 1 \
--instance 4 \
--shape "[3,48,320]" \
--iterations 2000 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1


# 测试最小时延
./vaststreamx-samples/bin/text_rec_prof \
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
--vdsp_params ../data/configs/crnn_rgbplanar.json \
--device_ids [0] \
--label_file ../data/labels/ocr_rec_dict.txt \
--batch_size 1 \
--instance 1 \
--shape "[3,48,320]" \
--iterations 500 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0

```

### text_rec_prof 命令行结果示例

```bash
# 本结果在 OCLK 880MHz 下测试所得
# 测试最大吞吐
- number of instances: 4
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 252.275
  latency (us):
    avg latency: 47272
    min latency: 15116
    max latency: 57007
    p50 latency: 47450
    p90 latency: 47558
    p95 latency: 47585
    p99 latency: 47667


# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 160.751
  latency (us):
    avg latency: 6219
    min latency: 6201
    max latency: 6966
    p50 latency: 6214
    p90 latency: 6223
    p95 latency: 6229
    p99 latency: 6284
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
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
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
index:0, score:0.828569,bbox:[ [670 52] [688 52] [688 60] [670 60] ]
index:1, score:0.821961,bbox:[ [660 80] [701 82] [700 99] [659 97] ]
index:2, score:0.821632,bbox:[ [638 134] [723 139] [722 157] [637 152] ]
index:3, score:0.758833,bbox:[ [637 152] [700 155] [699 172] [636 169] ]
index:4, score:0.743378,bbox:[ [786 281] [905 284] [905 302] [786 299] ]
index:5, score:0.901436,bbox:[ [791 301] [899 301] [899 335] [791 335] ]
index:6, score:0.888345,bbox:[ [790 333] [866 335] [865 367] [789 365] ]
index:7, score:0.628097,bbox:[ [1 349] [26 349] [26 360] [1 360] ]
index:8, score:0.84285,bbox:[ [854 453] [903 452] [903 467] [854 468] ]
index:9, score:0.760088,bbox:[ [848 532] [881 530] [882 543] [849 545] ]
```
并在图片上画出检测框，保存到  text_det_result.jpg

测试数据集
```bash
mkdir -p text_det_output
./vaststreamx-samples/bin/text_det \
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
--vdsp_params ../data/configs/dbnet_rgbplanar.json \
--device_id 0 \
--threshold 0.3 \
--box_unclip_ratio 1.5 \
--use_polygon_score 0 \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder text_det_output
```
结果保存在 text_det_output 文件夹里

统计精度
```bash
python3 ../evaluation/text_detection/eval.py \
--test_image_path  /opt/vastai/vaststreamx/data/datasets/ch4_test_images \
--boxes_npz_dir ./text_det_output \
--label_file ../data/labels/test_icdar2015_label.txt 
```
精度结果
```
metric:  {'precision': 0.5459697732997482, 'recall': 0.41742898411169954, 'hmean': 0.4731241473396998}
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
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
--vdsp_params ../data/configs/crnn_rgbplanar.json \
--device_id 0 \
--label_file ../data/labels/ocr_rec_dict.txt \
--input_file ../data/images/word_336.png 

```
输出
```bash
score: 0.973047
text: SUPER
```

测试数据集
```bash
./vaststreamx-samples/bin/text_rec \
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
--vdsp_params ../data/configs/crnn_rgbplanar.json \
--device_id 0 \
--label_file ../data/labels/ocr_rec_dict.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/CUTE80/CUTE80_img_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/CUTE80 \
--dataset_output_file cute80_pred.txt
```

统计精度
```bash
python3 ../evaluation/crnn/crnn_eval.py \
--gt_file /opt/vastai/vaststreamx/data/datasets/CUTE80/CUTE80_gt.txt \
--output_file cute80_pred.txt
```
精度结果
```
right_num = 233 all_num=288, acc = 0.8090277777777778
```

##  Python sample 


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
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
--det_vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--cls_model /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod \
--cls_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
--rec_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_ids [0] \
--rec_label_file ../../data/labels/ocr_rec_dict.txt \
--input_file ../../data/images/detect.jpg \
--det_box_type quad \
--output_file ocr_res.jpg

```

### ocr_e2e.py 运行结果示例

终端显示 检测到的文字的 bbox 多边形的四个角的坐标，文本内容，识别分数，bbox也画在图片上并保存为 ocr_res.jpg

```bash
#单张图片结果示例
[[660,80], [701,83], [700,100], [659,97]],  [('20029', 0.998046875)]
[[638,135], [723,140], [722,158], [637,152]],  [('97154197', 0.9990234375)]
[[637,152], [700,156], [699,172], [636,169]],  [('198727', 0.99755859375)]
[[786,282], [905,285], [905,302], [786,299]],  [('Free yourself from', 0.96826171875)]
[[791,301], [899,301], [899,336], [791,336]],  [('JOINT', 0.99462890625)]
[[790,333], [866,336], [865,368], [789,366]],  [('PAIN', 0.99267578125)]
[[854,454], [903,453], [903,468], [854,469]],  [('JOINT-RX', 0.93212890625)]
[[848,532], [881,530], [882,544], [849,546]],  [('TUFBRAN', 0.78466796875)]
```


### ocr_e2e.py 测试 同步推理 性能与时延
```bash
python ocr_e2e.py \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
--det_vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--cls_model /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod \
--cls_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
--rec_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_ids [0] \
--det_box_type quad \
--rec_label_file ../../data/labels/ocr_rec_dict.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file ppocr_v4_dataset_output.txt
#测试结果  880MHz 下
Image count: 500, total cost: 31.75 s, throughput: 15.75 fps, average latency: 0.064 s
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
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
--det_vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--cls_model /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod \
--cls_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
--rec_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_ids [0] \
--det_box_type quad \
--rec_label_file ../../data/labels/ocr_rec_dict.txt \
--input_file ../../data/images/detect.jpg \
--output_file ocr_res.jpg

#结果示例
[[660,80], [701,83], [700,100], [659,97]],  [('20029', 0.998046875)]
[[638,135], [723,140], [722,158], [637,152]],  [('97154197', 0.9990234375)]
[[637,152], [700,156], [699,172], [636,169]],  [('198727', 0.99755859375)]
[[786,282], [905,285], [905,302], [786,299]],  [('Free yourself from', 0.96826171875)]
[[791,301], [899,301], [899,336], [791,336]],  [('JOINT', 0.99462890625)]
[[790,333], [866,336], [865,368], [789,366]],  [('PAIN', 0.99267578125)]
[[854,454], [903,453], [903,468], [854,469]],  [('JOINT-RX', 0.93212890625)]
[[848,532], [881,530], [882,544], [849,546]],  [('TUFBRAN', 0.78466796875)]
```
### ocr_e2e_async.py 测试多线程异步推理 性能与时延

```bash
# 测试多线程异步
python ocr_e2e_async.py \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
--det_vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--cls_model /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod \
--cls_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
--rec_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_ids [0] \
--det_box_type quad \
--rec_label_file ../../data/labels/ocr_rec_dict.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/

#测试结果  880MHz 下
Image count: 500, total cost: 15.59 s, throughput: 32.07 fps, average latency: 2.042 s
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
                        instance number for each device
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
```


### text_det_prof.py 运行示例

在本目录下运行  
```bash
# 测试最大吞吐
python3 text_det_prof.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
--vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_ids [0]  \
--batch_size 1 \
--instance 1 \
--shape "[3,736,1280]" \
--iterations 500 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1


# 测试最小时延
python3 text_det_prof.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
--vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_ids [0]  \
--batch_size 1 \
--instance 1 \
--shape "[3,736,1280]" \
--iterations 300 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```


### text_det_prof.py 运行结果示例

```bash
# 本结果在 OCLK 880MHz 下测试所得
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 91.49
  latency (us):
    avg latency: 32684
    min latency: 22904
    max latency: 40960
    p50 latency: 32699
    p90 latency: 32874
    p95 latency: 32904
    p99 latency: 33375

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 53.93
  latency (us):
    avg latency: 18540
    min latency: 18304
    max latency: 21528
    p50 latency: 18486
    p90 latency: 18614
    p95 latency: 18732
    p99 latency: 19456
```

### text_cls_prof.py 命令行参数说明

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
                        instance number for each device
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
```


### text_cls_prof.py 运行示例

在本目录下运行  
```bash
# 测试最大吞吐
python3 text_cls_prof.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod \
--vdsp_params ../../data/configs/crnn_rgbplanar.json \
--device_ids [0]  \
--batch_size 32 \
--instance 1 \
--shape "[3,48,192]" \
--iterations 500 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1


# 测试最小时延
python3 text_cls_prof.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/cls-fp16-none-1_3_48_192-vacc/mod \
--vdsp_params ../../data/configs/crnn_rgbplanar.json \
--device_ids [0]  \
--batch_size 1 \
--instance 1 \
--shape "[3,48,192]" \
--iterations 4000 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```


### text_cls_prof.py 运行结果示例

```bash
# 本结果在 OCLK 880MHz 下测试所得
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 32
  throughput (qps): 1850.60
  latency (us):
    avg latency: 51779
    min latency: 23549
    max latency: 53900
    p50 latency: 51799
    p90 latency: 51882
    p95 latency: 51930
    p99 latency: 52197

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 928.07
  latency (us):
    avg latency: 1076
    min latency: 1015
    max latency: 1929
    p50 latency: 1075
    p90 latency: 1080
    p95 latency: 1084
    p99 latency: 1090
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
                        instance number for each device
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
```


### text_rec_prof.py 运行示例

在本目录下运行  
```bash
# 测试最大吞吐
python3 text_rec_prof.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
--vdsp_params ../../data/configs/crnn_rgbplanar.json \
--device_ids [0]  \
--batch_size 6 \
--instance 1 \
--label_file ../../data/labels/ocr_rec_dict.txt \
--shape "[3,48,320]" \
--iterations 500 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1


# 测试最小时延
python3 text_rec_prof.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
--vdsp_params ../../data/configs/crnn_rgbplanar.json \
--device_ids [0]  \
--batch_size 1 \
--instance 1 \
--label_file ../../data/labels/ocr_rec_dict.txt \
--shape "[3,48,320]" \
--iterations 500 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```


### text_rec_prof.py 运行结果示例

```bash
# 本结果在 OCLK 880MHz 下测试所得
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 6
  throughput (qps): 275.15
  latency (us):
    avg latency: 65253
    min latency: 55501
    max latency: 97562
    p50 latency: 65184
    p90 latency: 65263
    p95 latency: 65348
    p99 latency: 66766

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 105.56
  latency (us):
    avg latency: 9471
    min latency: 9442
    max latency: 10652
    p50 latency: 9453
    p90 latency: 9492
    p95 latency: 9531
    p99 latency: 9784
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
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
--vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_id 0 \
--input_file ../../data/images/detect.jpg \
--output_file text_det_result.jpg

```

### text_det.py 运行结果示例

终端显示 检测到的文字的 bbox 多边形的四个角的坐标，bbox也画在图片上并保存为 text_det_result.jpg

```bash
index:0, score:0.8285687764485677,bbox:[[670  53],[688  53],[688  61],[670  61]]
index:1, score:0.8219611069251751,bbox:[[660  80],[701  83],[700 100],[659  97]]
index:2, score:0.8216322827082808,bbox:[[638 135],[723 140],[722 158],[637 152]]
index:3, score:0.7588333656047952,bbox:[[637 152],[700 156],[699 172],[636 169]]
index:4, score:0.7433777126839491,bbox:[[786 282],[905 285],[905 302],[786 299]]
index:5, score:0.9014363087964862,bbox:[[791 301],[899 301],[899 336],[791 336]]
index:6, score:0.8883451347142621,bbox:[[790 333],[866 336],[865 368],[789 366]]
index:7, score:0.6280966622488839,bbox:[[  1 349],[ 26 349],[ 26 361],[  1 361]]
index:8, score:0.8428502129119577,bbox:[[854 454],[903 453],[903 468],[854 469]]
index:9, score:0.7600875937420388,bbox:[[848 532],[881 530],[882 544],[849 546]]
```

测试数据集
```bash
mkdir -p text_det_output
python3 text_det.py  \
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/det-fp16-none-1_3_736_1280-vacc/mod \
--vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder text_det_output
```
结果保存在 text_det_output 文件夹里

```bash
# 用刚才保存的npz文件测试精度
python3 ../../evaluation/text_detection/eval.py \
--test_image_path  /opt/vastai/vaststreamx/data/datasets/ch4_test_images \
--boxes_npz_dir ./text_det_output \
--label_file ../../data/labels/test_icdar2015_label.txt 
```
精度结果
```
metric:  {'precision': 0.5449968533668974, 'recall': 0.4169475204622051, 'hmean': 0.47244953627932357}
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
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
--vdsp_params ../../data/configs/crnn_rgbplanar.json \
--device_id 0 \
--label_file ../../data/labels/ocr_rec_dict.txt \
--input_file ../../data/images/word_336.png 

#数据集示例
python3 text_rec.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v4/rec-fp16-none-1_3_48_320-vacc/mod \
--vdsp_params ../../data/configs/crnn_rgbplanar.json \
--device_id 0 \
--label_file ../../data/labels/ocr_rec_dict.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/CUTE80/CUTE80_img_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/CUTE80 \
--dataset_output_file cute80_pred.txt

# 统计精度
python3 ../../evaluation/crnn/crnn_eval.py \
--gt_file /opt/vastai/vaststreamx/data/datasets/CUTE80/CUTE80_gt.txt \
--output_file cute80_pred.txt
```

### text_rec.py 运行结果示例

```bash
#单张图片结果示例
[('SUPER', 0.97314453125)]

#统计精度结果示例
right_num = 233 all_num=288, acc = 0.8090277777777778

```