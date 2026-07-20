# PPOCR V5 Samples

本文介绍 PPOCR V5 用法以及各模型精度性能测试方法

## 模型信息

### Document Image Orientation Classification

|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.5.0/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.md)  [modelzoo](https://github.com/Vastai/VastModelZOO/blob/develop/cv/classification/pplcnet_doc_ori/source_code/pplcnet_x1_0_doc_ori.md)|
|  输入 shape |   (1,3,224,224)    |
| INT8量化方式 |   mse         |
|  ONNX 精度(fixed shape) | accuracy: 74.51 |
|  VACC FP16  精度(mobile) | accuracy: 74.59 |
|  VACC INT8  精度(mobile)  |  accuracy: 71.65 |

### Text Detection

|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.5.0/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.md)  [modelzoo](https://github.com/Vastai/VastModelZOO/blob/main/cv/text_detection/dbnet/source_code/ppocr_v5_det.md)|
|  输入 shape |   (1,3,960,960)    |
| INT8量化方式 |   percentile         |
|  ONNX 精度(fixed shape) | {'precision': 0.7504, 'recall': 0.7922, 'hmean': 0.7707} |
|  VACC FP16  精度(mobile) | {'precision': 0.7545, 'recall': 0.7995, 'hmean': 0.7763} |
|  VACC INT8  精度(mobile)  |  - |

### Text Line orientation Classification

|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.5.0/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.md)  [modelzoo](https://github.com/Vastai/VastModelZOO/blob/develop/cv/classification/pplcnet_textline_ori/source_code/pplcnet_textline_ori.md)|
|  输入 shape |   (1,3,80,160)    |
| INT8量化方式 |   kl_divergence         |
|  ONNX 精度(fixed shape) | 85.50 |
|  VACC FP16  精度(mobile) | 86.50 |
|  VACC INT8  精度(mobile)  |  84.00 |

### Text Recognition

|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/PaddlePaddle/PaddleOCR/blob/v3.5.0/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.md)  [modelzoo](https://github.com/Vastai/VastModelZOO/blob/main/cv/text_recognition/ppocr_v5_rec/README.md)|
|  输入 shape |    (1,3,48,320)    |
| INT8量化方式 |   percentile          |
|  ONNX精度(fixed shape) | {'ExactMatch': 0.7548, 'CharMatch': 0.8759} |
|  VACC FP16  精度(mobile) | {'ExactMatch': 0.7871, 'CharMatch': 0.8979} |
|  VACC INT8  精度(mobile)  | - |

## C++ Sample

### PPOCR-V5-E2E 测试

- ppocr.cpp：文档方向分类，文本检测，文字行方向分类，文字识别，四个模型串一起，同步推理sample。
- ppocr_async.cpp: 文档方向分类，文本检测，文字行方向分类，文字识别，四个模型串一起，异步推理sample。

#### ppocr 命令行参数说明

```bash
usage: vaststreamx-samples/bin/ppocr [options] ... 
options:
      --doc_ori_model           document image orientation classify model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/det_model_vacc_fp16/mod])
      --doc_ori_config          document image orientation classify vdsp preprocess parameter file (string [=../data/configs/dbnet_rgbplanar.json])
      --doc_ori_label_file      document image orientation classify vdsp preprocess parameter file (string [=../data/configs/dbnet_rgbplanar.json])
      --use_doc_ori_cls         use image orientation classification (bool [=0])
      --det_model               text detection model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/det_model_vacc_fp16/mod])
      --det_config              text detection vdsp preprocess parameter file (string [=../data/configs/dbnet_rgbplanar.json])
      --det_elf_file            text detection elf file (string [=/opt/vastai/vaststreamx/data/elf/find_contours_ext_op])
      --det_box_type            text detection box type (string [=quad])
      --det_box_thresh          text detection box thresh (float [=0.6])
      --text_ori_model          textline orientation classification model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/cls_model_vacc_fp16/mod])
      --text_ori_config         text classification vdsp preprocess parameter file (string [=../data/configs/crnn_rgbplanar.json])
      --text_ori_thresh         text classification thresh (float [=0.9])
      --use_text_ori_cls        use text classification (bool [=1])
      --rec_model               text recognition model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/rec_model_vacc_fp16/mod])
      --rec_config              text recognition vdsp preprocess parameter file (string [=../data/configs/crnn_rgbplanar.json])
      --rec_label_file          text recognition label file (string [=../data/labels/ppocr_keys_v1.txt])
      --rec_drop_score          text recogniztion drop score threshold (float [=0.5])
      --rotate_elf              rotate op elf file (string [=/opt/vastai/vaststreamx/data/elf/simple_rotate_ext_op])
      --warp_perspective_elf    warp perspective op elf file (string [=/opt/vastai/vaststreamx/data/elf/warp_perspective_ext_op])
      --batch_size              batch size of the model (unsigned int [=1])
      --device_ids              device id to run (string [=[0]])
      --hw_config               hw-config file of the model suite (string [=])
      --input_file              input image (string [=../data/images/ppocr.jpg])
      --output_file             output image file (string [=])
      --dataset_filelist        input dataset filelist (string [=])
      --dataset_root            input dataset root (string [=])
      --dataset_output_file     dataset output file (string [=dataset_output.txt])
  -?, --help                    print this message
```

#### ppocr 命令行示例

在build 目录里执行

```bash
#单图片, 使用文档图片方向分类，与文字行方向分类
./vaststreamx-samples/bin/ppocr \
--doc_ori_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/doc_ori_fp16_1-3-224-224/mod \
--doc_ori_config ../data/configs/doc_ori_rgbplanar.json \
--doc_ori_label_file ../data/labels/doc_ori_label.txt \
--use_doc_ori_cls 1 \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--det_config ../data/configs/dbnet_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--det_box_type "quad" \
--det_box_thresh 0.6 \
--text_ori_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--text_ori_config ../data/configs/textline_ori_rgbplanar.json \
--text_ori_thresh 0.9 \
--use_text_ori_cls 1 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--rec_config ../data/configs/crnn_rgbplanar.json \
--rec_label_file ../data/labels/ppocrv5_dict.txt \
--rec_drop_score 0.5 \
--rotate_elf /opt/vastai/vaststream/lib/op/ext_op/video/simple_rotate_ext_op \
--warp_perspective_elf /opt/vastai/vaststreamx/data/elf/warp_perspective_ext_op \
--device_ids [0] \
--input_file ../data/images/ppocr-rot.jpg \
--output_file ppocr_v5_result.jpg

## 输出

Thread 0 get ../data/images/ppocr-rot.jpg result:
bbox:[ [18 28] [311 28] [311 80] [ 18 80] ], score: 0.999581, string: 纯臻营养护发素
bbox:[ [21 77] [177 77] [177 107] [ 21 107] ], score: 0.999581, string: 产品信息/参数
bbox:[ [22 108] [334 108] [334 136] [ 22 136] ], score: 0.959386, string: （45元/每公斤，100公斤起订）
bbox:[ [21 137] [285 139] [285 169] [ 21 167] ], score: 0.98863, string: 每瓶22元，1000瓶起订）
bbox:[ [21 173] [303 172] [303 198] [ 21 198] ], score: 0.925034, string: 【品牌】：代加工方式/OEMODM
bbox:[ [21 204] [236 205] [236 231] [ 21 229] ], score: 0.989583, string: 【品名】：纯臻营养护发素
bbox:[ [411 231] [431 231] [431 304] [ 411 304] ], score: 0.985921, string: ODMOEM
bbox:[ [23 236] [244 236] [244 260] [ 23 260] ], score: 0.992676, string: 【产品编号】：YM-X-3011
bbox:[ [22 267] [182 266] [182 290] [ 22 292] ], score: 0.975852, string: 【净含量】：220ml
bbox:[ [22 298] [255 299] [255 323] [ 22 322] ], score: 0.993164, string: 【适用人群】：适合所有肤质
bbox:[ [21 329] [346 330] [346 356] [ 21 354] ], score: 0.981574, string: 【主要成分】：鲸蜡硬脂醇、
bbox:[ [23 361] [284 361] [284 385] [ 23 385] ], score: 0.960903, string: 糖、椰油酰胺丙基甜菜碱、泛
bbox:[ [365 364] [477 364] [477 390] [ 365 390] ], score: 0.979736, string: （成品包材）
bbox:[ [22 392] [364 392] [364 417] [ 22 417] ], score: 0.989592, string: 【主要功能】：可紧致头发磷
bbox:[ [24 423] [376 423] [376 448] [ 24 448] ], score: 0.99594, string: 即时持久改善头发光泽的效果
bbox:[ [23 455] [139 455] [139 480] [ 23 480] ], score: 0.998861, string: 发足够的滋养
Save file to: ./thread_0_ppocr_v5_result.jpg

# 并在图片上画出检测框，保存到  thread_0_ppocr_v5_result.jpg
```

```bash
#单图片, 不使用文档图片方向分类，使用文字行方向分类
./vaststreamx-samples/bin/ppocr \
--use_doc_ori_cls 0 \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--det_config ../data/configs/dbnet_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--det_box_type "quad" \
--det_box_thresh 0.6 \
--text_ori_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--text_ori_config ../data/configs/textline_ori_rgbplanar.json \
--text_ori_thresh 0.9 \
--use_text_ori_cls 1 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--rec_config ../data/configs/crnn_rgbplanar.json \
--rec_label_file ../data/labels/ppocrv5_dict.txt \
--rec_drop_score 0.5 \
--rotate_elf /opt/vastai/vaststream/lib/op/ext_op/video/simple_rotate_ext_op \
--warp_perspective_elf /opt/vastai/vaststreamx/data/elf/warp_perspective_ext_op \
--device_ids [0] \
--input_file ../data/images/ppocr.jpg \
--output_file ppocr_v5_result.jpg

## 输出
Thread 0 get ../data/images/ppocr.jpg result:
bbox:[ [18 28] [311 28] [311 80] [ 18 80] ], score: 0.999581, string: 纯臻营养护发素
bbox:[ [21 76] [176 76] [176 106] [ 21 106] ], score: 0.99986, string: 产品信息/参数
bbox:[ [23 108] [334 108] [334 136] [ 23 136] ], score: 0.938534, string: (45元/每公斤，100公斤起订）
bbox:[ [21 137] [285 138] [285 168] [ 21 167] ], score: 0.988107, string: 每瓶22元，1000瓶起订）
bbox:[ [21 173] [303 172] [303 198] [ 21 198] ], score: 0.906652, string: 【品牌】：代加工方式/OEMODM
bbox:[ [21 204] [237 205] [237 231] [ 21 229] ], score: 0.990682, string: 【品名】：纯臻营养护发素
bbox:[ [411 231] [431 231] [431 304] [ 411 304] ], score: 0.986572, string: ODMOEM
bbox:[ [23 236] [244 236] [244 260] [ 23 260] ], score: 0.990875, string: 【产品编号】：YM-X-3011
bbox:[ [22 267] [182 266] [182 291] [ 22 292] ], score: 0.967773, string: 【净含量】：220ml
bbox:[ [22 298] [255 299] [255 323] [ 22 322] ], score: 0.993164, string: 【适用人群】：适合所有肤质
bbox:[ [21 329] [346 330] [346 356] [ 21 354] ], score: 0.972194, string: 【主要成分】：鲸蜡硬脂醇、
bbox:[ [23 361] [284 361] [284 385] [ 23 385] ], score: 0.964948, string: 糖、椰油酰胺丙基甜菜碱、泛
bbox:[ [365 364] [477 364] [477 390] [ 365 390] ], score: 0.986084, string: （成品包材）
bbox:[ [23 392] [364 392] [364 417] [ 23 417] ], score: 0.990825, string: 【主要功能】：可紧致头发磷
bbox:[ [24 423] [376 423] [376 448] [ 24 448] ], score: 0.99594, string: 即时持久改善头发光泽的效果
bbox:[ [23 454] [139 454] [139 480] [ 23 480] ], score: 0.997721, string: 发足够的滋养
Save file to: ./thread_0_ppocr_v5_result.jpg
```

```bash
#单图片, 不使用文档图片方向分类，不使用文字行方向分类
./vaststreamx-samples/bin/ppocr \
--use_doc_ori_cls 0 \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--det_config ../data/configs/dbnet_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--det_box_type "quad" \
--det_box_thresh 0.6 \
--use_text_ori_cls 0 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--rec_config ../data/configs/crnn_rgbplanar.json \
--rec_label_file ../data/labels/ppocrv5_dict.txt \
--rec_drop_score 0.5 \
--rotate_elf /opt/vastai/vaststream/lib/op/ext_op/video/simple_rotate_ext_op \
--warp_perspective_elf /opt/vastai/vaststreamx/data/elf/warp_perspective_ext_op \
--device_ids [0] \
--input_file ../data/images/ppocr.jpg \
--output_file ppocr_v5_result.jpg

# 输出
Thread 0 get ../data/images/ppocr.jpg result:
bbox:[ [18 28] [311 28] [311 80] [ 18 80] ], score: 0.999581, string: 纯臻营养护发素
bbox:[ [21 76] [176 76] [176 106] [ 21 106] ], score: 0.99986, string: 产品信息/参数
bbox:[ [23 108] [334 108] [334 136] [ 23 136] ], score: 0.938534, string: (45元/每公斤，100公斤起订）
bbox:[ [21 137] [285 138] [285 168] [ 21 167] ], score: 0.988107, string: 每瓶22元，1000瓶起订）
bbox:[ [21 173] [303 172] [303 198] [ 21 198] ], score: 0.906652, string: 【品牌】：代加工方式/OEMODM
bbox:[ [21 204] [237 205] [237 231] [ 21 229] ], score: 0.990682, string: 【品名】：纯臻营养护发素
bbox:[ [23 236] [244 236] [244 260] [ 23 260] ], score: 0.990875, string: 【产品编号】：YM-X-3011
bbox:[ [22 267] [182 266] [182 291] [ 22 292] ], score: 0.967773, string: 【净含量】：220ml
bbox:[ [22 298] [255 299] [255 323] [ 22 322] ], score: 0.993164, string: 【适用人群】：适合所有肤质
bbox:[ [21 329] [346 330] [346 356] [ 21 354] ], score: 0.972194, string: 【主要成分】：鲸蜡硬脂醇、
bbox:[ [23 361] [284 361] [284 385] [ 23 385] ], score: 0.964948, string: 糖、椰油酰胺丙基甜菜碱、泛
bbox:[ [365 364] [477 364] [477 390] [ 365 390] ], score: 0.986084, string: （成品包材）
bbox:[ [23 392] [364 392] [364 417] [ 23 417] ], score: 0.990825, string: 【主要功能】：可紧致头发磷
bbox:[ [24 423] [376 423] [376 448] [ 24 448] ], score: 0.99594, string: 即时持久改善头发光泽的效果
bbox:[ [23 454] [139 454] [139 480] [ 23 480] ], score: 0.997721, string: 发足够的滋养
Save file to: ./thread_0_ppocr_v5_result.jpg

# 可以看到 ODMOEM字符无法正确识别
```

测试 后三个模型同步推理 的性能与时延, 可以通过 --device_ids 指定多个 die

```bash
./vaststreamx-samples/bin/ppocr \
--use_doc_ori_cls 0 \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--det_config ../data/configs/dbnet_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--det_box_type "quad" \
--det_box_thresh 0.6 \
--text_ori_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--text_ori_config ../data/configs/textline_ori_rgbplanar.json \
--text_ori_thresh 0.9 \
--use_text_ori_cls 1 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--rec_config ../data/configs/crnn_rgbplanar.json \
--rec_label_file ../data/labels/ppocrv5_dict.txt \
--rec_drop_score 0.5 \
--rotate_elf /opt/vastai/vaststream/lib/op/ext_op/video/simple_rotate_ext_op \
--warp_perspective_elf /opt/vastai/vaststreamx/data/elf/warp_perspective_ext_op \
--device_ids [0] \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file ppocr_v5_dataset_output.txt

##结果示例  开启dpm 下
Image count: 500, total cost: 30829 ms, throughput: 16.2185 fps. Average latency: 61.658 ms.
```

#### ppocr_async 命令行参数说明

```bash
usage: vaststreamx-samples/bin/ppocr_async [options] ... 
options:
      --doc_ori_model           document image orientation classify model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/det_model_vacc_fp16/mod])
      --doc_ori_config          document image orientation classify vdsp preprocess parameter file (string [=../data/configs/dbnet_rgbplanar.json])
      --doc_ori_label_file      document image orientation classify vdsp preprocess parameter file (string [=../data/configs/dbnet_rgbplanar.json])
      --use_doc_ori_cls         use image orientation classification (bool [=0])
      --det_model               text detection model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/det_model_vacc_fp16/mod])
      --det_config              text detection vdsp preprocess parameter file (string [=../data/configs/dbnet_rgbplanar.json])
      --det_elf_file            text detection elf file (string [=/opt/vastai/vaststreamx/data/elf/find_contours_ext_op])
      --det_box_type            text detection box type (string [=quad])
      --det_box_thresh          text detection box thresh (float [=0.6])
      --text_ori_model          textline orientation classification model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/cls_model_vacc_fp16/mod])
      --text_ori_config         text classification vdsp preprocess parameter file (string [=../data/configs/crnn_rgbplanar.json])
      --text_ori_thresh         text classification thresh (float [=0.9])
      --use_text_ori_cls        use text classification (bool [=1])
      --rec_model               text recognition model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/rec_model_vacc_fp16/mod])
      --rec_config              text recognition vdsp preprocess parameter file (string [=../data/configs/crnn_rgbplanar.json])
      --rec_label_file          text recognition label file (string [=../data/labels/ppocr_keys_v1.txt])
      --rec_drop_score          text recogniztion drop score threshold (float [=0.5])
      --rotate_elf              rotate op elf file (string [=/opt/vastai/vaststreamx/data/elf/simple_rotate_ext_op])
      --warp_perspective_elf    warp perspective op elf file (string [=/opt/vastai/vaststreamx/data/elf/warp_perspective_ext_op])
      --batch_size              batch size of the model (unsigned int [=1])
      --device_ids              device id to run (string [=[0]])
      --hw_config               hw-config file of the model suite (string [=])
      --input_file              input image (string [=../data/images/ppocr.jpg])
      --output_file             output image file (string [=])
      --dataset_filelist        input dataset filelist (string [=])
      --dataset_root            input dataset root (string [=])
      --dataset_output_file     dataset output file (string [=])
      --queue_size              set queue size (unsigned int [=1])
  -?, --help                    print this message
```

#### ppocr_async 命令行示例

在build 目录里执行

单图片示例

```bash
#单图片, 使用文档图片方向分类，与文字行方向分类
./vaststreamx-samples/bin/ppocr_async \
--doc_ori_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/doc_ori_fp16_1-3-224-224/mod \
--doc_ori_config ../data/configs/doc_ori_rgbplanar.json \
--doc_ori_label_file ../data/labels/doc_ori_label.txt \
--use_doc_ori_cls 1 \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--det_config ../data/configs/dbnet_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--det_box_type "quad" \
--det_box_thresh 0.6 \
--text_ori_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--text_ori_config ../data/configs/textline_ori_rgbplanar.json \
--text_ori_thresh 0.9 \
--use_text_ori_cls 1 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--rec_config ../data/configs/crnn_rgbplanar.json \
--rec_label_file ../data/labels/ppocrv5_dict.txt \
--rec_drop_score 0.5 \
--rotate_elf /opt/vastai/vaststream/lib/op/ext_op/video/simple_rotate_ext_op \
--warp_perspective_elf /opt/vastai/vaststreamx/data/elf/warp_perspective_ext_op \
--device_ids [0] \
--queue_size 1 \
--input_file ../data/images/ppocr-rot.jpg \
--output_file ppocr_v5_async_result.jpg

# 输出
bbox:[ [18 28] [311 28] [311 80] [ 18 80] ], score: 0.999581, string: 纯臻营养护发素
bbox:[ [21 77] [177 77] [177 107] [ 21 107] ], score: 0.999581, string: 产品信息/参数
bbox:[ [22 108] [334 108] [334 136] [ 22 136] ], score: 0.959386, string: （45元/每公斤，100公斤起订）
bbox:[ [21 137] [285 139] [285 169] [ 21 167] ], score: 0.98863, string: 每瓶22元，1000瓶起订）
bbox:[ [21 173] [303 172] [303 198] [ 21 198] ], score: 0.925034, string: 【品牌】：代加工方式/OEMODM
bbox:[ [21 204] [236 205] [236 231] [ 21 229] ], score: 0.989583, string: 【品名】：纯臻营养护发素
bbox:[ [411 231] [431 231] [431 304] [ 411 304] ], score: 0.985921, string: ODMOEM
bbox:[ [23 236] [244 236] [244 260] [ 23 260] ], score: 0.992676, string: 【产品编号】：YM-X-3011
bbox:[ [22 267] [182 266] [182 290] [ 22 292] ], score: 0.975852, string: 【净含量】：220ml
bbox:[ [22 298] [255 299] [255 323] [ 22 322] ], score: 0.993164, string: 【适用人群】：适合所有肤质
bbox:[ [21 329] [346 330] [346 356] [ 21 354] ], score: 0.981574, string: 【主要成分】：鲸蜡硬脂醇、
bbox:[ [23 361] [284 361] [284 385] [ 23 385] ], score: 0.960903, string: 糖、椰油酰胺丙基甜菜碱、泛
bbox:[ [365 364] [477 364] [477 390] [ 365 390] ], score: 0.979736, string: （成品包材）
bbox:[ [22 392] [364 392] [364 417] [ 22 417] ], score: 0.989592, string: 【主要功能】：可紧致头发磷
bbox:[ [24 423] [376 423] [376 448] [ 24 448] ], score: 0.99594, string: 即时持久改善头发光泽的效果
bbox:[ [23 455] [139 455] [139 480] [ 23 480] ], score: 0.998861, string: 发足够的滋养
Save file to: ./thread_0_ppocr_v5_async_result.jpg
```

```bash
#单图片, 不使用文档图片方向分类， 使用文字行方向分类
./vaststreamx-samples/bin/ppocr_async \
--use_doc_ori_cls 0 \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--det_config ../data/configs/dbnet_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--det_box_type "quad" \
--det_box_thresh 0.6 \
--text_ori_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--text_ori_config ../data/configs/textline_ori_rgbplanar.json \
--text_ori_thresh 0.9 \
--use_text_ori_cls 1 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--rec_config ../data/configs/crnn_rgbplanar.json \
--rec_label_file ../data/labels/ppocrv5_dict.txt \
--rec_drop_score 0.5 \
--rotate_elf /opt/vastai/vaststream/lib/op/ext_op/video/simple_rotate_ext_op \
--warp_perspective_elf /opt/vastai/vaststreamx/data/elf/warp_perspective_ext_op \
--device_ids [0] \
--queue_size 1 \
--input_file ../data/images/ppocr.jpg \
--output_file ppocr_v5_async_result.jpg

# 输出
bbox:[ [18 28] [311 28] [311 80] [ 18 80] ], score: 0.999581, string: 纯臻营养护发素
bbox:[ [21 76] [176 76] [176 106] [ 21 106] ], score: 0.99986, string: 产品信息/参数
bbox:[ [23 108] [334 108] [334 136] [ 23 136] ], score: 0.938534, string: (45元/每公斤，100公斤起订）
bbox:[ [21 137] [285 138] [285 168] [ 21 167] ], score: 0.988107, string: 每瓶22元，1000瓶起订）
bbox:[ [21 173] [303 172] [303 198] [ 21 198] ], score: 0.906652, string: 【品牌】：代加工方式/OEMODM
bbox:[ [21 204] [237 205] [237 231] [ 21 229] ], score: 0.990682, string: 【品名】：纯臻营养护发素
bbox:[ [411 231] [431 231] [431 304] [ 411 304] ], score: 0.986572, string: ODMOEM
bbox:[ [23 236] [244 236] [244 260] [ 23 260] ], score: 0.990875, string: 【产品编号】：YM-X-3011
bbox:[ [22 267] [182 266] [182 291] [ 22 292] ], score: 0.967773, string: 【净含量】：220ml
bbox:[ [22 298] [255 299] [255 323] [ 22 322] ], score: 0.993164, string: 【适用人群】：适合所有肤质
bbox:[ [21 329] [346 330] [346 356] [ 21 354] ], score: 0.972194, string: 【主要成分】：鲸蜡硬脂醇、
bbox:[ [23 361] [284 361] [284 385] [ 23 385] ], score: 0.964948, string: 糖、椰油酰胺丙基甜菜碱、泛
bbox:[ [365 364] [477 364] [477 390] [ 365 390] ], score: 0.986084, string: （成品包材）
bbox:[ [23 392] [364 392] [364 417] [ 23 417] ], score: 0.990825, string: 【主要功能】：可紧致头发磷
bbox:[ [24 423] [376 423] [376 448] [ 24 448] ], score: 0.99594, string: 即时持久改善头发光泽的效果
bbox:[ [23 454] [139 454] [139 480] [ 23 480] ], score: 0.997721, string: 发足够的滋养
Save file to: ./thread_0_ppocr_v5_async_result.jpg
```

```bash
#单图片, 不使用文档图片方向分类， 不使用文字行方向分类
./vaststreamx-samples/bin/ppocr_async \
--use_doc_ori_cls 0 \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--det_config ../data/configs/dbnet_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--det_box_type "quad" \
--det_box_thresh 0.6 \
--use_text_ori_cls 0 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--rec_config ../data/configs/crnn_rgbplanar.json \
--rec_label_file ../data/labels/ppocrv5_dict.txt \
--rec_drop_score 0.5 \
--rotate_elf /opt/vastai/vaststream/lib/op/ext_op/video/simple_rotate_ext_op \
--warp_perspective_elf /opt/vastai/vaststreamx/data/elf/warp_perspective_ext_op \
--device_ids [0] \
--queue_size 1 \
--input_file ../data/images/ppocr.jpg \
--output_file ppocr_v5_async_result.jpg

# 输出
bbox:[ [18 28] [311 28] [311 80] [ 18 80] ], score: 0.999581, string: 纯臻营养护发素
bbox:[ [21 76] [176 76] [176 106] [ 21 106] ], score: 0.99986, string: 产品信息/参数
bbox:[ [23 108] [334 108] [334 136] [ 23 136] ], score: 0.938534, string: (45元/每公斤，100公斤起订）
bbox:[ [21 137] [285 138] [285 168] [ 21 167] ], score: 0.988107, string: 每瓶22元，1000瓶起订）
bbox:[ [21 173] [303 172] [303 198] [ 21 198] ], score: 0.906652, string: 【品牌】：代加工方式/OEMODM
bbox:[ [21 204] [237 205] [237 231] [ 21 229] ], score: 0.990682, string: 【品名】：纯臻营养护发素
bbox:[ [23 236] [244 236] [244 260] [ 23 260] ], score: 0.990875, string: 【产品编号】：YM-X-3011
bbox:[ [22 267] [182 266] [182 291] [ 22 292] ], score: 0.967773, string: 【净含量】：220ml
bbox:[ [22 298] [255 299] [255 323] [ 22 322] ], score: 0.993164, string: 【适用人群】：适合所有肤质
bbox:[ [21 329] [346 330] [346 356] [ 21 354] ], score: 0.972194, string: 【主要成分】：鲸蜡硬脂醇、
bbox:[ [23 361] [284 361] [284 385] [ 23 385] ], score: 0.964948, string: 糖、椰油酰胺丙基甜菜碱、泛
bbox:[ [365 364] [477 364] [477 390] [ 365 390] ], score: 0.986084, string: （成品包材）
bbox:[ [23 392] [364 392] [364 417] [ 23 417] ], score: 0.990825, string: 【主要功能】：可紧致头发磷
bbox:[ [24 423] [376 423] [376 448] [ 24 448] ], score: 0.99594, string: 即时持久改善头发光泽的效果
bbox:[ [23 454] [139 454] [139 480] [ 23 480] ], score: 0.997721, string: 发足够的滋养
Save file to: ./thread_0_ppocr_v5_async_result.jpg

# 可以看到 ODMOEM字符无法正确识别
```

测试后三个模型多线程异步推理 的性能与时延, 可以通过 --device_ids 指定多个 die

```bash
./vaststreamx-samples/bin/ppocr_async \
--use_doc_ori_cls 0 \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--det_config ../data/configs/dbnet_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--det_box_type "quad" \
--det_box_thresh 0.6 \
--text_ori_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--text_ori_config ../data/configs/textline_ori_rgbplanar.json \
--text_ori_thresh 0.9 \
--use_text_ori_cls 1 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--rec_config ../data/configs/crnn_rgbplanar.json \
--rec_label_file ../data/labels/ppocrv5_dict.txt \
--rec_drop_score 0.5 \
--rotate_elf /opt/vastai/vaststream/lib/op/ext_op/video/simple_rotate_ext_op \
--warp_perspective_elf /opt/vastai/vaststreamx/data/elf/warp_perspective_ext_op \
--device_ids [0] \
--queue_size 1 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/

##结果示例  开启dpm 下
Image count: 500, total cost: 13543 ms, throughput: 36.9194 fps. Average latency: 341.604 ms. 
```

### 文档图片方向分类模型精度与性能测试

#### doc_img_orient_cls 命令参数说明

```bash
usage: ./vaststreamx-samples/bin/doc_img_orient_cls [options] ... 
options:
  -m, --model_prefix        model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/doc_ori_int8_mse_1-3-224-224/mod])
      --hw_config           hw-config file of the model suite (string [=])
      --vdsp_params         vdsp preprocess parameter file (string [=../data/configs/doc_ori_rgbplanar.json])
  -d, --device_id           device id to run (unsigned int [=0])
      --label_file          label file (string [=[0, 180]])
      --input_file          input image file (string [=../data/images/ppocr-rot.jpg])
      --dataset_val_file    dataset validation file (string [=])
      --dataset_root        input dataset root (string [=])
  -?, --help                print this message
```

#### doc_img_orient_cls 命令示例

```bash
# 单张图片测试
./vaststreamx-samples/bin/doc_img_orient_cls \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/doc_ori_fp16_1-3-224-224/mod \
--vdsp_params ../data/configs/doc_ori_rgbplanar.json \
--device_id 0 \
--label_file ../data/labels/doc_ori_label.txt \
--input_file ../data/images/ppocr-rot.jpg 

# 单张图片结果示例
Image angle: 90, confidence: 0.922363

# 数据集测试
./vaststreamx-samples/bin/doc_img_orient_cls \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/doc_ori_fp16_1-3-224-224/mod \
--vdsp_params ../data/configs/doc_ori_rgbplanar.json \
--device_id 0 \
--label_file ../data/labels/doc_ori_label.txt \
--dataset_val_file /opt/vastai/vaststreamx/data/datasets/text_image_orientation/val.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/text_image_orientation/

# 数据集结果示例
Accuracy: 0.775935
```

#### doc_img_orient_cls_prof 命令参数说明

```bash
usage: vaststreamx-samples/bin/doc_img_orient_cls_prof [options] ... 
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/doc_ori_int8_mse_1-3-224-224/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/doc_ori_rgbplanar.json])
  -d, --device_ids      device id to run (string [=[0]])
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

#### doc_img_orient_cls_prof 命令示例

```bash
# 最大吞吐
./vaststreamx-samples/bin/doc_img_orient_cls_prof \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/doc_ori_fp16_1-3-224-224/mod \
--vdsp_params ../data/configs/doc_ori_rgbplanar.json \
--device_ids [0] \
--batch_size 16 \
--instance 4 \
--shape "[3,224,224]" \
--percentiles "[50,90,95,99]" \
--iterations 1000 \
--warmup_times 500 \
--queue_size 1 \
--input_host 1

# 最小时延
./vaststreamx-samples/bin/doc_img_orient_cls_prof \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/doc_ori_fp16_1-3-224-224/mod \
--vdsp_params ../data/configs/doc_ori_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,224,224]" \
--percentiles "[50,90,95,99]" \
--iterations 10000 \
--warmup_times 5000 \
--queue_size 0 \
--input_host 1
```

### doc_img_orient_cls_prof 命令结果示例

```bash

# 最大吞吐 OCLK=1200MHz
- number of instances: 4
  devices: [ 0 ]
  queue size: 1
  batch size: 16
  throughput (qps): 3042.29
  latency (us):
    avg latency: 62084
    min latency: 24924
    max latency: 107570
    p50 latency: 61381
    p90 latency: 68645
    p95 latency: 74451
    p99 latency: 90704

# 最小时延 OCLK=630MHz
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 640.755
  latency (us):
    avg latency: 1559
    min latency: 1527
    max latency: 1769
    p50 latency: 1536
    p90 latency: 1598
    p95 latency: 1600
    p99 latency: 1610
```

### 文本检测模型精度与性能测试

#### text_det 命令行参数说明

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

#### text_det 命令行示例

在build 目录里执行
单图片示例

```bash
./vaststreamx-samples/bin/text_det \
--model_prefix /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--vdsp_params ../data/configs/dbnet_rgbplanar.json \
--device_id 0 \
--threshold 0.6 \
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
--model_prefix /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
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

```text
metric:  {'precision': 0.7538, 'recall': 0.7987, 'hmean': 0.7756}
```

#### text_det_prof 命令行参数说明

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

#### text_det_prof 命令行示例

```bash
# 测试最大吞吐
./vaststreamx-samples/bin/text_det_prof \
--model_prefix /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
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
--model_prefix /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
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

#### text_det_prof 命令行结果示例

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

### 文本行方向分类模型精度性能测试

#### text_cls 命令行参数说明

```bash
usage: vaststreamx-samples/bin/text_cls [options] ... 
options:
  -m, --model_prefix        model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/textline_ori_fp16_1-3-80-160/mod])
      --hw_config           hw-config file of the model suite (string [=])
      --vdsp_params         vdsp preprocess parameter file (string [=../data/configs/textline_ori_rgbplanar.json])
  -d, --device_id           device id to run (unsigned int [=0])
      --input_file          input image file (string [=../data/images/word_336.png])
      --dataset_val_file    dataset validation file (string [=])
      --dataset_root        input dataset root (string [=])
  -?, --help                print this message
```

#### text_cls 命令行示例

单张图片测试

```bash
vaststreamx-samples/bin/text_cls \
--model_prefix /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--vdsp_params ../data/configs/textline_ori_rgbplanar.json \
--device_id 0 \
--input_file ../data/images/word.jpg

#输出
Text Line angle: 180, confidence: 1
```

数据集测试

```bash
vaststreamx-samples/bin/text_cls \
--model_prefix /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--vdsp_params ../data/configs/textline_ori_rgbplanar.json \
--device_id 0 \
--dataset_val_file /opt/vastai/vaststreamx/data/datasets/textline_orientation_example_data/val.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/textline_orientation_example_data/

#输出
Accuracy: 0.91
```

#### text_cls_prof 命令行参数说明

```bash
usage: ./vaststreamx-samples/bin/text_cls_prof [options] ... 
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/resnet34_vd])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/crnn_rgbplanar.json])
  -d, --device_ids      device id to run (string [=[0]])
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

#### text_cls_prof 命令行示例

```bash
# 测试最大吞吐
./vaststreamx-samples/bin/text_cls_prof \
--model_prefix /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--vdsp_params ../data/configs/textline_ori_rgbplanar.json \
--device_ids [0] \
--batch_size 16 \
--instance 1 \
--iterations 1000 \
--warmup_times 1000 \
--shape "[3,80,160]" \
--input_host 1 \
--queue_size 1


# 测试最小时延
./vaststreamx-samples/bin/text_cls_prof \
--model_prefix /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--vdsp_params ../data/configs/textline_ori_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 8000 \
--warmup_times 8000 \
--input_host 1 \
--queue_size 0

```

#### text_cls_prof 命令行结果示例

```bash
# 本结果在 OCLK 1200MHz 下测试所得
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 16
  throughput (qps): 3103.16
  latency (us):
    avg latency: 14947
    min latency: 7759
    max latency: 18213
    p50 latency: 15085
    p90 latency: 15321
    p95 latency: 15374
    p99 latency: 15468


# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 1435
  latency (us):
    avg latency: 696
    min latency: 642
    max latency: 1710
    p50 latency: 695
    p90 latency: 699
    p95 latency: 701
    p99 latency: 711
```

### 文本识别模型精度性能测试

#### text_rec 命令行参数说明

```bash
usage: ./vaststreamx-samples/bin/text_rec [options] ... 
options:
  -m, --model_prefix           model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/resnet34_vd-int8-max-1_3_32_100-vacc/mod])
      --hw_config              hw-config file of the model suite (string [=])
      --vdsp_params            vdsp preprocess parameter file (string [=../data/configs/crnn_rgbplanar.json])
  -d, --device_id              device id to run (unsigned int [=0])
      --label_file             label file (string [=../data/labels/key_37.txt])
      --input_file             input image (string [=../data/images/word_336.png])
      --dataset_filelist       input dataset filelist (string [=])
      --dataset_root           input dataset root (string [=])
      --dataset_output_file    dataset output file (string [=])
  -?, --help                   print this message
```

### text_rec 命令行示例

在build 目录里执行
单图片示例

```bash
./vaststreamx-samples/bin/text_rec \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--vdsp_params ../data/configs/crnn_rgbplanar.json \
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
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
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
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
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
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
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

## Python sample

### PPOCR-V5 E2E Samples

#### ppocr.py 命令行参数说明

```bash
options:
  -h, --help            show this help message and exit
  --doc_ori_model DOC_ORI_MODEL
                        document image orientation classification model prefix of the model suite files
  --doc_ori_vdsp_params DOC_ORI_VDSP_PARAMS
                        document image orientation classification model vdsp preprocess parameter file
  --doc_ori_label_file DOC_ORI_LABEL_FILE
                        doc image orientation classification label file
  --use_doc_ori_cls USE_DOC_ORI_CLS
                        whether use document image orientation classifier
  --det_model DET_MODEL
                        text detection model prefix of the model suite files
  --det_vdsp_params DET_VDSP_PARAMS
                        text detection vdsp preprocess parameter file
  --det_box_type DET_BOX_TYPE
                        det box type, poly or quad
  --det_elf_file DET_ELF_FILE
                        input file
  --det_box_thresh DET_BOX_THRESH
                        text detection box thresh
  --text_ori_model TEXT_ORI_MODEL
                        text detection model prefix of the model suite files
  --text_ori_vdsp_params TEXT_ORI_VDSP_PARAMS
                        text detection vdsp preprocess parameter file
  --text_ori_label_list TEXT_ORI_LABEL_LIST
                        text line orientation classification label list
  --text_ori_thresh TEXT_ORI_THRESH
                        text line orientation classification thresh
  --use_text_ori_cls USE_TEXT_ORI_CLS
                        whether use text line orientation classifier
  --rec_model REC_MODEL
                        text recognition model prefix of the model suite files
  --rec_vdsp_params REC_VDSP_PARAMS
                        text recognition vdsp preprocess parameter file
  --rec_label_file REC_LABEL_FILE
                        text recognizition label file
  --rec_drop_score REC_DROP_SCORE
                        text recogniztion drop score threshold
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

#### ppocr.py 运行示例

在本目录下运行  

```bash
#单张图片示例, 判断文档图片方向, 判断文字行方向
python3 ppocr.py \
--doc_ori_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/doc_ori_fp16_1-3-224-224/mod \
--doc_ori_vdsp_params ../../data/configs/doc_ori_rgbplanar.json \
--doc_ori_label_file ../../data/labels/doc_ori_label.txt \
--use_doc_ori_cls 1 \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--det_vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--det_box_type quad \
--det_box_thresh 0.6 \
--text_ori_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--text_ori_vdsp_params ../../data/configs/textline_ori_rgbplanar.json \
--text_ori_thresh 0.9 \
--use_text_ori_cls 1 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--rec_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--rec_label_file ../../data/labels/ppocrv5_dict.txt \
--device_ids [0] \
--input_file ../../data/images/ppocr-rot.jpg \
--output_file ppocr_v5_result.jpg

# 输出
[[22,32], [308,32], [308,77], [22,77]],  [('纯臻营养护发素', 0.99951171875)]
[[24,80], [174,80], [174,105], [24,105]],  [('产品信息/参数', 0.99951171875)]
[[25,110], [333,110], [333,135], [25,135]],  [('（45元/每公斤，100公斤起订）', 0.95068359375)]
[[24,140], [283,142], [283,167], [24,165]],  [('每瓶22元，1000瓶起订）', 0.98486328125)]
[[24,176], [302,175], [302,196], [24,197]],  [('【品牌】：代加工方式/OEMODM', 0.98583984375)]
[[24,206], [235,208], [235,229], [24,228]],  [('【品名】：纯臻营养护发素', 0.98828125)]
[[413,233], [430,233], [430,303], [413,303]],  [('ODMOEM', 0.9912109375)]
[[26,239], [243,239], [243,259], [26,259]],  [('【产品编号】：YM-X-3011', 0.990234375)]
[[24,269], [181,268], [181,289], [25,291]],  [('【净含量】：220ml', 0.98095703125)]
[[25,300], [253,302], [253,321], [24,320]],  [('【适用人群】：适合所有肤质', 0.9921875)]
[[24,331], [345,333], [345,354], [24,353]],  [('【主要成分】：鲸蜡硬脂醇、燕麦β-葡聚', 0.9873046875)]
[[26,363], [283,363], [283,384], [26,384]],  [('糖、椰油酰胺丙基甜菜碱、泛醌', 0.96875)]
[[367,367], [476,367], [476,388], [367,388]],  [('（成品包材）', 0.9873046875)]
[[25,395], [362,395], [362,416], [25,416]],  [('【主要功能】：可紧致头发磷层，从而达到', 0.99267578125)]
[[27,426], [374,426], [374,447], [27,447]],  [('即时持久改善头发光泽的效果，给干燥的头', 0.9951171875)]
[[26,457], [137,457], [137,479], [26,479]],  [('发足够的滋养', 0.99853515625)]
save file  thread_0_ppocr_v5_result.jpg
```

```bash
#单张图片示例, 不判断文档图片方向, 判断文字行方向 
python3 ppocr.py \
--use_doc_ori_cls 0 \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--det_vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--det_box_type quad \
--det_box_thresh 0.6 \
--text_ori_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--text_ori_vdsp_params ../../data/configs/textline_ori_rgbplanar.json \
--text_ori_thresh 0.9 \
--use_text_ori_cls 1 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--rec_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--rec_label_file ../../data/labels/ppocrv5_dict.txt \
--device_ids [0] \
--input_file ../../data/images/ppocr.jpg \
--output_file ppocr_v5_result.jpg

# 输出
[[22,32], [308,32], [308,77], [22,77]],  [('纯臻营养护发素', 0.99951171875)]
[[24,79], [174,79], [174,104], [24,104]],  [('产品信息/参数', 0.9951171875)]
[[26,110], [333,110], [333,135], [26,135]],  [('（45元/每公斤，100公斤起订）', 0.94091796875)]
[[24,140], [283,141], [283,166], [24,165]],  [('每瓶22元，1000瓶起订）', 0.9833984375)]
[[24,176], [302,175], [302,196], [24,197]],  [('【品牌】：代加工方式/OEMODM', 0.98583984375)]
[[24,206], [235,208], [235,229], [24,228]],  [('【品名】：纯臻营养护发素', 0.9873046875)]
[[413,233], [430,233], [430,303], [413,303]],  [('ODMOEM', 0.9912109375)]
[[26,239], [243,239], [243,259], [26,259]],  [('【产品编号】：YM-X-3011', 0.98876953125)]
[[24,270], [181,268], [181,289], [25,291]],  [('【净含量】：220ml', 0.97412109375)]
[[25,300], [253,302], [253,321], [24,320]],  [('【适用人群】：适合所有肤质', 0.9921875)]
[[24,331], [345,333], [345,354], [24,353]],  [('【主要成分】：鲸蜡硬脂醇、燕麦β-葡聚', 0.98779296875)]
[[26,364], [283,364], [283,384], [26,384]],  [('糖、椰油酰胺丙基甜菜碱、泛醌', 0.9755859375)]
[[367,367], [476,367], [476,388], [367,388]],  [('（成品包材）', 0.994140625)]
[[26,395], [362,395], [362,416], [26,416]],  [('【主要功能】：可紧致头发磷层，从而达到', 0.990234375)]
[[27,426], [374,426], [374,447], [27,447]],  [('即时持久改善头发光泽的效果，给干燥的头', 0.9951171875)]
[[26,457], [137,457], [137,478], [26,478]],  [('发足够的滋养', 0.9970703125)]
save file  thread_0_ppocr_v5_result.jpg
```

```bash
#单张图片示例, 不判断文档图片方向, 不判断文字行方向 
python3 ppocr.py \
--use_doc_ori_cls 0 \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--det_vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--det_box_type quad \
--det_box_thresh 0.6 \
--use_text_ori_cls 0 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--rec_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--rec_label_file ../../data/labels/ppocrv5_dict.txt \
--device_ids [0] \
--input_file ../../data/images/ppocr.jpg \
--output_file ppocr_v5_result.jpg

# 输出 
[[22,32], [308,32], [308,77], [22,77]],  [('纯臻营养护发素', 0.99951171875)]
[[24,79], [174,79], [174,104], [24,104]],  [('产品信息/参数', 0.9951171875)]
[[26,110], [333,110], [333,135], [26,135]],  [('（45元/每公斤，100公斤起订）', 0.94091796875)]
[[24,140], [283,141], [283,166], [24,165]],  [('每瓶22元，1000瓶起订）', 0.9833984375)]
[[24,176], [302,175], [302,196], [24,197]],  [('【品牌】：代加工方式/OEMODM', 0.98583984375)]
[[24,206], [235,208], [235,229], [24,228]],  [('【品名】：纯臻营养护发素', 0.9873046875)]
[[413,233], [430,233], [430,303], [413,303]],  [('ODMOEM', 0.9912109375)]
[[26,239], [243,239], [243,259], [26,259]],  [('【产品编号】：YM-X-3011', 0.98876953125)]
[[24,270], [181,268], [181,289], [25,291]],  [('【净含量】：220ml', 0.97412109375)]
[[25,300], [253,302], [253,321], [24,320]],  [('【适用人群】：适合所有肤质', 0.9921875)]
[[24,331], [345,333], [345,354], [24,353]],  [('【主要成分】：鲸蜡硬脂醇、燕麦β-葡聚', 0.98779296875)]
[[26,364], [283,364], [283,384], [26,384]],  [('糖、椰油酰胺丙基甜菜碱、泛醌', 0.9755859375)]
[[367,367], [476,367], [476,388], [367,388]],  [('（成品包材）', 0.994140625)]
[[26,395], [362,395], [362,416], [26,416]],  [('【主要功能】：可紧致头发磷层，从而达到', 0.990234375)]
[[27,426], [374,426], [374,447], [27,447]],  [('即时持久改善头发光泽的效果，给干燥的头', 0.9951171875)]
[[26,457], [137,457], [137,478], [26,478]],  [('发足够的滋养', 0.9970703125)]
save file  thread_0_ppocr_v5_result.jpg
```

#### ppocr.py 测试 同步推理 性能与时延

```bash
python ppocr.py \
--use_doc_ori_cls 0 \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--det_vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--det_box_type quad \
--det_box_thresh 0.6 \
--text_ori_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--text_ori_vdsp_params ../../data/configs/textline_ori_rgbplanar.json \
--text_ori_thresh 0.9 \
--use_text_ori_cls 1 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--rec_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--rec_label_file ../../data/labels/ppocrv5_dict.txt \
--device_ids [0] \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file ppocr_v5_dataset_output.txt
#测试结果  在开启dpm 下
Image count: 500, total cost: 44.01 s, throughput: 11.36 fps, average latency: 0.088 s
```

#### ppocr_async.py 命令行参数说明

```bash
options:
  -h, --help            show this help message and exit
  --doc_ori_model DOC_ORI_MODEL
                        document image orientation classification model prefix of the model suite files
  --doc_ori_vdsp_params DOC_ORI_VDSP_PARAMS
                        document image orientation classification model vdsp preprocess parameter file
  --doc_ori_label_file DOC_ORI_LABEL_FILE
                        doc image orientation classification label file
  --use_doc_ori_cls USE_DOC_ORI_CLS
                        whether use document image orientation classifier
  --det_model DET_MODEL
                        text detection model prefix of the model suite files
  --det_vdsp_params DET_VDSP_PARAMS
                        text detection vdsp preprocess parameter file
  --det_box_type DET_BOX_TYPE
                        det box type, poly or quad
  --det_elf_file DET_ELF_FILE
                        input file
  --det_box_thresh DET_BOX_THRESH
                        text detection box threshold
  --text_ori_model TEXT_ORI_MODEL
                        text detection model prefix of the model suite files
  --text_ori_vdsp_params TEXT_ORI_VDSP_PARAMS
                        text detection vdsp preprocess parameter file
  --text_ori_label_list TEXT_ORI_LABEL_LIST
                        text line orientation classification label list
  --text_ori_thresh TEXT_ORI_THRESH
                        text line orientation classification thresh
  --use_text_ori_cls USE_TEXT_ORI_CLS
                        whether use text line orientation classifier
  --rec_model REC_MODEL
                        text recognition model prefix of the model suite files
  --rec_vdsp_params REC_VDSP_PARAMS
                        text recognition vdsp preprocess parameter file
  --rec_label_file REC_LABEL_FILE
                        text recognizition label file
  --rec_drop_score REC_DROP_SCORE
                        text recogniztion drop score threshold
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
  --queue_size QUEUE_SIZE
                        queue size of the pipeline
```

#### ppocr_async.py 命令行示例

```bash
#单张图片示例, 判断文档图片方向, 判断文字行方向
python ppocr_async.py \
--doc_ori_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/doc_ori_fp16_1-3-224-224/mod \
--doc_ori_vdsp_params ../../data/configs/doc_ori_rgbplanar.json \
--doc_ori_label_file ../../data/labels/doc_ori_label.txt \
--use_doc_ori_cls 1 \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--det_vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--det_box_type quad \
--det_box_thresh 0.6 \
--text_ori_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--text_ori_vdsp_params ../../data/configs/textline_ori_rgbplanar.json \
--text_ori_thresh 0.9 \
--use_text_ori_cls 1 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--rec_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--rec_label_file ../../data/labels/ppocrv5_dict.txt \
--device_ids [0] \
--input_file ../../data/images/ppocr-rot.jpg \
--output_file ppocr_v5_async_result.jpg

#输出
Thread:0,Get ../../data/images/ppocr-rot.jpg result
[[22,32], [308,32], [308,77], [22,77]],  [('纯臻营养护发素', 0.99951171875)]
[[24,80], [174,80], [174,105], [24,105]],  [('产品信息/参数', 0.99951171875)]
[[25,110], [333,110], [333,135], [25,135]],  [('（45元/每公斤，100公斤起订）', 0.95068359375)]
[[24,140], [283,142], [283,167], [24,165]],  [('每瓶22元，1000瓶起订）', 0.98486328125)]
[[24,176], [302,175], [302,196], [24,197]],  [('【品牌】：代加工方式/OEMODM', 0.98583984375)]
[[24,206], [235,208], [235,229], [24,228]],  [('【品名】：纯臻营养护发素', 0.98828125)]
[[413,233], [430,233], [430,303], [413,303]],  [('ODMOEM', 0.9912109375)]
[[26,239], [243,239], [243,259], [26,259]],  [('【产品编号】：YM-X-3011', 0.990234375)]
[[24,269], [181,268], [181,289], [25,291]],  [('【净含量】：220ml', 0.98095703125)]
[[25,300], [253,302], [253,321], [24,320]],  [('【适用人群】：适合所有肤质', 0.9921875)]
[[24,331], [345,333], [345,354], [24,353]],  [('【主要成分】：鲸蜡硬脂醇、燕麦β-葡聚', 0.9873046875)]
[[26,363], [283,363], [283,384], [26,384]],  [('糖、椰油酰胺丙基甜菜碱、泛醌', 0.96875)]
[[367,367], [476,367], [476,388], [367,388]],  [('（成品包材）', 0.9873046875)]
[[25,395], [362,395], [362,416], [25,416]],  [('【主要功能】：可紧致头发磷层，从而达到', 0.99267578125)]
[[27,426], [374,426], [374,447], [27,447]],  [('即时持久改善头发光泽的效果，给干燥的头', 0.9951171875)]
[[26,457], [137,457], [137,479], [26,479]],  [('发足够的滋养', 0.99853515625)]
save file to thread_0_ppocr_v5_async_result.jpg
```

```bash
#单张图片示例, 不判断文档图片方向, 判断文字行方向
python ppocr_async.py \
--use_doc_ori_cls 0 \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--det_vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--det_box_type quad \
--det_box_thresh 0.6 \
--text_ori_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--text_ori_vdsp_params ../../data/configs/textline_ori_rgbplanar.json \
--text_ori_thresh 0.9 \
--use_text_ori_cls 1 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--rec_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--rec_label_file ../../data/labels/ppocrv5_dict.txt \
--device_ids [0] \
--input_file ../../data/images/ppocr.jpg \
--output_file ppocr_v5_async_result.jpg

# 输出
Thread:0,Get ../../data/images/ppocr.jpg result
[[22,32], [308,32], [308,77], [22,77]],  [('纯臻营养护发素', 0.99951171875)]
[[24,79], [174,79], [174,104], [24,104]],  [('产品信息/参数', 0.9951171875)]
[[26,110], [333,110], [333,135], [26,135]],  [('（45元/每公斤，100公斤起订）', 0.94091796875)]
[[24,140], [283,141], [283,166], [24,165]],  [('每瓶22元，1000瓶起订）', 0.9833984375)]
[[24,176], [302,175], [302,196], [24,197]],  [('【品牌】：代加工方式/OEMODM', 0.98583984375)]
[[24,206], [235,208], [235,229], [24,228]],  [('【品名】：纯臻营养护发素', 0.9873046875)]
[[413,233], [430,233], [430,303], [413,303]],  [('ODMOEM', 0.9912109375)]
[[26,239], [243,239], [243,259], [26,259]],  [('【产品编号】：YM-X-3011', 0.98876953125)]
[[24,270], [181,268], [181,289], [25,291]],  [('【净含量】：220ml', 0.97412109375)]
[[25,300], [253,302], [253,321], [24,320]],  [('【适用人群】：适合所有肤质', 0.9921875)]
[[24,331], [345,333], [345,354], [24,353]],  [('【主要成分】：鲸蜡硬脂醇、燕麦β-葡聚', 0.98779296875)]
[[26,364], [283,364], [283,384], [26,384]],  [('糖、椰油酰胺丙基甜菜碱、泛醌', 0.9755859375)]
[[367,367], [476,367], [476,388], [367,388]],  [('（成品包材）', 0.994140625)]
[[26,395], [362,395], [362,416], [26,416]],  [('【主要功能】：可紧致头发磷层，从而达到', 0.990234375)]
[[27,426], [374,426], [374,447], [27,447]],  [('即时持久改善头发光泽的效果，给干燥的头', 0.9951171875)]
[[26,457], [137,457], [137,478], [26,478]],  [('发足够的滋养', 0.9970703125)]
save file to thread_0_ppocr_v5_async_result.jpg
```

```bash
#单张图片示例, 不判断文档图片方向, 不判断文字行方向
python ppocr_async.py \
--use_doc_ori_cls 0 \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--det_vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--det_box_type quad \
--det_box_thresh 0.6 \
--use_text_ori_cls 0 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--rec_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--rec_label_file ../../data/labels/ppocrv5_dict.txt \
--device_ids [0] \
--input_file ../../data/images/ppocr.jpg \
--output_file ppocr_v5_async_result.jpg

# 输出
Thread:0,Get ../../data/images/ppocr.jpg result
[[22,32], [308,32], [308,77], [22,77]],  [('纯臻营养护发素', 0.99951171875)]
[[24,79], [174,79], [174,104], [24,104]],  [('产品信息/参数', 0.9951171875)]
[[26,110], [333,110], [333,135], [26,135]],  [('（45元/每公斤，100公斤起订）', 0.94091796875)]
[[24,140], [283,141], [283,166], [24,165]],  [('每瓶22元，1000瓶起订）', 0.9833984375)]
[[24,176], [302,175], [302,196], [24,197]],  [('【品牌】：代加工方式/OEMODM', 0.98583984375)]
[[24,206], [235,208], [235,229], [24,228]],  [('【品名】：纯臻营养护发素', 0.9873046875)]
[[413,233], [430,233], [430,303], [413,303]],  [('ODMOEM', 0.9912109375)]
[[26,239], [243,239], [243,259], [26,259]],  [('【产品编号】：YM-X-3011', 0.98876953125)]
[[24,270], [181,268], [181,289], [25,291]],  [('【净含量】：220ml', 0.97412109375)]
[[25,300], [253,302], [253,321], [24,320]],  [('【适用人群】：适合所有肤质', 0.9921875)]
[[24,331], [345,333], [345,354], [24,353]],  [('【主要成分】：鲸蜡硬脂醇、燕麦β-葡聚', 0.98779296875)]
[[26,364], [283,364], [283,384], [26,384]],  [('糖、椰油酰胺丙基甜菜碱、泛醌', 0.9755859375)]
[[367,367], [476,367], [476,388], [367,388]],  [('（成品包材）', 0.994140625)]
[[26,395], [362,395], [362,416], [26,416]],  [('【主要功能】：可紧致头发磷层，从而达到', 0.990234375)]
[[27,426], [374,426], [374,447], [27,447]],  [('即时持久改善头发光泽的效果，给干燥的头', 0.9951171875)]
[[26,457], [137,457], [137,478], [26,478]],  [('发足够的滋养', 0.9970703125)]
save file to thread_0_ppocr_v5_async_result.jpg
```

#### ppocr_async.py 测试多线程异步推理 性能与时延

```bash
# 测试多线程异步
python ppocr_async.py \
--use_doc_ori_cls 0 \
--det_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--det_vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--det_elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--det_box_type quad \
--det_box_thresh 0.6 \
--text_ori_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--text_ori_vdsp_params ../../data/configs/textline_ori_rgbplanar.json \
--text_ori_thresh 0.9 \
--use_text_ori_cls 1 \
--rec_model /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--rec_vdsp_params ../../data/configs/crnn_rgbplanar.json \
--rec_label_file ../../data/labels/ppocrv5_dict.txt \
--device_ids [0] \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/

#测试结果  1200MHz 下
Image count: 500, total cost: 26.54 s, throughput: 18.84 fps, average latency: 5.275 s
```

### 文档图片方向分类模型精度性能测试

#### doc_img_orient_cls.py 命令行参数说明

```bash
usage: doc_img_orient_cls.py [-h] [-m MODEL_PREFIX] [--hw_config HW_CONFIG] [--vdsp_params VDSP_PARAMS] [-d DEVICE_ID]
                             [--label_file LABEL_FILE] [--input_file INPUT_FILE] [--dataset_val_file DATASET_VAL_FILE]
                             [--dataset_root DATASET_ROOT]

options:
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
  --dataset_val_file DATASET_VAL_FILE
                        validation file
  --dataset_root DATASET_ROOT
                        input dataset root
```

#### doc_img_orient_cls.py 命令行示例

```bash
# 单张图片
python3 doc_img_orient_cls.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/doc_ori_fp16_1-3-224-224/mod \
--vdsp_params ../../data/configs/doc_ori_rgbplanar.json \
--device_id 0 \
--label_file ../../data/labels/doc_ori_label.txt \
--input_file ../../data/images/ppocr-rot.jpg 

# 单张图片结果示例
Image angle: 90, confidence: 0.9224

# 数据集测试
python3 doc_img_orient_cls.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/doc_ori_fp16_1-3-224-224/mod \
--vdsp_params ../../data/configs/doc_ori_rgbplanar.json \
--device_id 0 \
--label_file ../../data/labels/doc_ori_label.txt \
--dataset_val_file /opt/vastai/vaststreamx/data/datasets/text_image_orientation/val.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/text_image_orientation/

# 数据集测试结果示例
Accuracy: 0.7759
```

#### doc_img_orient_cls_prof.py 命令行参数说明

```bash
usage: doc_img_orient_cls_prof.py [-h] [-m MODEL_PREFIX] [--hw_config HW_CONFIG] [--vdsp_params VDSP_PARAMS] [-d DEVICE_IDS]
                                  [-b BATCH_SIZE] [-i INSTANCE] [-s SHAPE] [--iterations ITERATIONS] [--queue_size QUEUE_SIZE]
                                  [--percentiles PERCENTILES] [--input_host INPUT_HOST] [--warmup_times WARMUP_TIMES]

options:
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

#### doc_img_orient_cls_prof.py 命令行示例

```bash
# 最大吞吐
python3 doc_img_orient_cls_prof.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/doc_ori_fp16_1-3-224-224/mod \
--vdsp_params ../../data/configs/doc_ori_rgbplanar.json \
--device_ids [0] \
--batch_size 16 \
--instance 4 \
--shape "[3,224,224]" \
--percentiles "[50,90,95,99]" \
--iterations 1000 \
--warmup_times 500 \
--queue_size 1 \
--input_host 1

#最小时延
python3 doc_img_orient_cls_prof.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/doc_ori_fp16_1-3-224-224/mod \
--vdsp_params ../../data/configs/doc_ori_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,224,224]" \
--percentiles "[50,90,95,99]" \
--iterations 5000 \
--warmup_times 5000 \
--queue_size 0 \
--input_host 0
```

#### doc_img_orient_cls_prof.py 命令行结果示例

```bash
# 最大吞吐 OCLK=1200MHz
- number of instances: 4
  devices: [0]
  queue size: 1
  batch size: 16
  throughput (qps): 3265.18
  latency (us):
    avg latency: 58461
    min latency: 26322
    max latency: 84619
    p50 latency: 57776
    p90 latency: 60934
    p95 latency: 65124
    p99 latency: 77871

# 最小时延 OCLK=630MHz
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 661.52
  latency (us):
    avg latency: 1510
    min latency: 1502
    max latency: 1692
    p50 latency: 1508
    p90 latency: 1515
    p95 latency: 1518
    p99 latency: 1572
```

### 文本检测模型精度性能测试

#### text_det.py 命令行参数说明

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

#### text_det.py 运行示例

在本目录下运行  

```bash
python3 text_det.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
--vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_id 0 \
--input_file ../../data/images/detect.jpg \
--output_file text_det_result.jpg
```

#### text_det.py 运行结果示例

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
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
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

```text
metric:  {'precision': 0.7545, 'recall': 0.7995, 'hmean': 0.7763}
```

#### text_det_prof.py 命令行参数说明

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

#### text_det_prof.py 运行示例

在本目录下运行  

```bash
# 测试最大吞吐
python3 text_det_prof.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
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
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/det_fp16_1-3-960-960/mod \
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

#### text_det_prof.py 运行结果示例

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

### 文本行方向分类模型精度性能测试

#### text_cls.py 命令行参数说明

```bash
options:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input file
  --dataset_val_file DATASET_VAL_FILE
                        dataset validation file
  --dataset_root DATASET_ROOT
                        input dataset root
```

#### text_cls.py 命令行示例

```bash
# 测试单张图片
python3 text_cls.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--vdsp_params ../../data/configs/textline_ori_rgbplanar.json \
--device_id 0 \
--input_file ../../data/images/word.jpg

# 输出
Image angle: 180, confidence: 1.0000

# 测试数据集
python3 text_cls.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--vdsp_params ../../data/configs/textline_ori_rgbplanar.json \
--device_id 0 \
--dataset_val_file /opt/vastai/vaststreamx/data/datasets/textline_orientation_example_data/val.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/textline_orientation_example_data/

# 输出
Accuracy: 0.9100
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

### text_cls_prof.py 运行示例

在本目录下运行  

```bash
# 测试最大吞吐
python3 text_cls_prof.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--vdsp_params ../../data/configs/textline_ori_rgbplanar.json \
--device_ids [0]  \
--batch_size 32 \
--instance 1 \
--shape "[3,48,192]" \
--iterations 500 \
--warmup_times 300 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1


# 测试最小时延
python3 text_cls_prof.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/textline_ori_fp16_1-3-80-160/mod \
--vdsp_params ../../data/configs/textline_ori_rgbplanar.json \
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
# 本结果在 OCLK 1200MHz 下测试所得
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 32
  throughput (qps): 2388.81
  latency (us):
    avg latency: 40065
    min latency: 20987
    max latency: 43615
    p50 latency: 40111
    p90 latency: 40164
    p95 latency: 40182
    p99 latency: 40228

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 988.41
  latency (us):
    avg latency: 1010
    min latency: 884
    max latency: 1945
    p50 latency: 953
    p90 latency: 1330
    p95 latency: 1333
    p99 latency: 1339
```

### 文本识别模型精度性能测试

#### text_rec.py 命令行参数说明

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

#### text_rec.py 运行示例

在本目录下运行  

```bash
#单张图片示例
python3 text_rec.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
--vdsp_params ../../data/configs/crnn_rgbplanar.json \
--device_id 0 \
--label_file ../../data/labels/ppocrv5_dict.txt \
--input_file ../../data/images/word_336.png 

#数据集示例
python3 text_rec.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
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

#### text_rec.py 运行结果示例

```bash
#单张图片结果示例
[('SUPER', 0.95751953125)]

#统计精度结果示例
metric:  {'ExactMatch': 0.7032, 'CharMatch': 0.8458}
```

#### text_rec_prof.py 命令行参数说明

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

#### text_rec_prof.py 运行示例

在本目录下运行  

```bash
# 测试最大吞吐
python3 text_rec_prof.py \
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
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
-m /opt/vastai/vaststreamx/data/models/ppocr-v5-mobile/rec_fp16_1-3-48-320/mod \
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

#### text_rec_prof.py 运行结果示例

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
