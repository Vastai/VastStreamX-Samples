# GroundingDino Sample

本目录提供基于 grounding_dino 模型的 目标检测 sample

## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/IDEA-Research/GroundingDINO)  [modelzoo](http://gitlabdev.vastai.com/VastML/algorithm_modelzoo/-/tree/develop/detection/grounding_dino) |
|  输入 shape |  "text": [(1,195),(1,195),(1,195),(1,195,195)], "image":[(1,3,800,1333)]  |
| INT8量化方式 |   -          |
|  官方精度 |  "mAP@.5":   - ;     "mAP@.5:.95":  48.4   |
|  VACC FP16  精度 | "mAP@.5":  60.4 ;  "mAP@.5:.95":  45.2  |
|  VACC INT8  精度 | - |

## 数据准备

下载模型 groundingdino.tar.gz 到 /opt/vastai/vaststreamx/data/models 里 
下载 tokenizer bert-base-uncased 到 /opt/vastai/vaststreamx/data/tokenizer 里  
下载数据集 det_coco_val 到 /opt/vastai/vaststreamx/data/datasets 里  

## C++ sample     
将标签字符串通过python脚本生成 tokens, 并保存到 input_tokens.npz

```bash
#在build目录执行
python3 ../samples/detection/grounding_dino/generate_tokens_for_cpp.py \
--tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/bert-base-uncased \
--label_file ../data/labels/coco2id.txt \
--output_file input_tokens.npz
```

### grounding_dino 命令行参数说明
```bash
options:
      --txtmod_prefix            prefix of the text model suite files (string [=/opt/vastai/vaststreamx/data/models/groundingdino_text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod])
      --txtmod_hw_config         hw-config file of the text model suite (string [=])
      --txtmod_vdsp_params       vdsp preprocess parameter file (string [=data/configs/clip_txt_vdsp.json])
      --imgmod_prefix            prefix of the image model suite files (string [=/opt/vastai/vaststreamx/data/models/groundingdino_img_encoder-fp16-none-1_3_800_1333-vacc/mod])
      --imgmod_hw_config         hw-config file of the image model suite (string [=])
      --imgmod_vdsp_params       image model dsp preprocess parameter file (string [=./data/configs/groundingdino_bgr888.json])
      --decmod_prefix            model prefix of the decoder model suite files (string [=/opt/vastai/vaststreamx/data/models/groundingdino_decoder-fp16-none-1_22223_256_1_195_256_1_195_1_195_1_195_195-vacc/mod])
      --decmod_hw_config         hw-config file of the decoder model suite (string [=])
      --npz_file                 npz file for text model (string [=])
  -d, --device_id                device id to run (unsigned int [=0])
      --threshold                threshold for detection (float [=0.2])
      --label_file               label file (string [=./data/labels/coco2id.txt])
      --input_file               input file (string [=./data/images/dog.jpg])
      --output_file              output file (string [=grounding_dino_result.jpg])
      --dataset_filelist         dataset filename list (string [=])
      --dataset_root             dataset root (string [=])
      --dataset_output_folder    dataset output folder path (string [=])
      --positive_map_file        positive map file (string [=../data/bin/positive_map.bin])
  -?, --help                     print this message
```

### grounding_dino 运行示例
在build目录执行
```bash
# 单张图片测试
./vaststreamx-samples/bin/grounding_dino \
--txtmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod \
--txtmod_vdsp_params  ../data/configs/clip_txt_vdsp.json \
--imgmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod \
--imgmod_vdsp_params ../data/configs/groundingdino_bgr888.json \
--decmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/decoder-fp16-none-1_22223_256_1_195_256_1_195_1_195_1_195_195-vacc/mod \
--npz_file input_tokens.npz \
--label_file ../data/labels/coco2id.txt \
--positive_map_file ../data/bin/positive_map.bin \
--device_id  0 \
--threshold 0.2 \
--input_file /opt/vastai/vaststreamx/data/datasets/det_coco_val/000000000139.jpg  \
--output_file grounding_dino_result.jpg


# 数据集测试
mkdir -p ./grounding_dino_out
./vaststreamx-samples/bin/grounding_dino \
--txtmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod \
--txtmod_vdsp_params  ../data/configs/clip_txt_vdsp.json \
--imgmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod \
--imgmod_vdsp_params ../data/configs/groundingdino_bgr888.json \
--decmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/decoder-fp16-none-1_22223_256_1_195_256_1_195_1_195_1_195_195-vacc/mod \
--npz_file input_tokens.npz \
--label_file ../data/labels/coco2id.txt \
--positive_map_file ../data/bin/positive_map.bin \
--device_id  0 \
--threshold 0.01 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder ./grounding_dino_out

# 精度统计
python3 ../evaluation/detection/eval_map.py \
--gt ../evaluation/detection/instances_val2017.json \
--txt ./grounding_dino_out
```


### grounding_dino 运行结果示例
```bash
# 单张图片检测结果
Detection objects:
Object class: tv, score: 0.823043, bbox: [6.21094, 165.886, 154.102, 272.178]
Object class: person, score: 0.808823, bbox: [409.453, 157.254, 465.547, 296.203]
Object class: clock, score: 0.627951, bbox: [447.051, 119.682, 461.699, 142.199]
Object class: person, score: 0.625666, bbox: [385.195, 172.698, 400.43, 206.708]
Object class: vase, score: 0.558928, bbox: [550.43, 299.453, 587.07, 401.117]
Object class: potted plant, score: 0.546747, bbox: [232.852, 175.767, 265.898, 212.792]
Object class: chair, score: 0.516412, bbox: [290.078, 217.992, 348.359, 318.252]
Object class: chair, score: 0.506061, bbox: [361.328, 219.162, 415.547, 317.914]
Object class: vase, score: 0.504005, bbox: [350.469, 206.344, 362.031, 231.305]
Object class: chair, score: 0.480234, bbox: [302.812, 217.29, 352.188, 305.641]
Object class: chair, score: 0.458136, bbox: [388.711, 219.838, 441.914, 304.341]
Object class: wine glass, score: 0.447178, bbox: [313.555, 191.549, 319.883, 214.066]
Object class: wine glass, score: 0.439137, bbox: [318.701, 192.173, 324.424, 213.858]
Object class: chair, score: 0.386589, bbox: [402.734, 220.202, 442.266, 305.225]
Object class: wine glass, score: 0.367332, bbox: [361.348, 214.56, 373.652, 232.241]
Object class: bowl, score: 0.364777, bbox: [465.889, 215.652, 475.361, 221.58]
Object class: cup, score: 0.361277, bbox: [166.719, 232.839, 186.406, 267.628]
Object class: microwave, score: 0.331011, bbox: [475.234, 136.687, 525.391, 175.117]
Object class: refrigerator, score: 0.330687, bbox: [446.133, 167.55, 513.242, 287.155]
Object class: bottle, score: 0.319227, bbox: [491.367, 153.77, 496.133, 171.97]
Object class: dining table, score: 0.313685, bbox: [310.977, 228.965, 448.398, 317.68]
Object class: refrigerator, score: 0.297572, bbox: [490.742, 173.219, 513.008, 285.231]
Object class: bottle, score: 0.296247, bbox: [496.201, 153.913, 502.549, 173.492]
Object class: hair drier, score: 0.296168, bbox: [425.84, 157.254, 445.41, 178.887]
Object class: refrigerator, score: 0.293809, bbox: [479.102, 171.502, 512.773, 286.115]
Object class: handbag, score: 0.29276, bbox: [212.266, 299.557, 257.109, 328.21]
Object class: oven, score: 0.287567, bbox: [557.695, 207.436, 639.805, 288.871]
Object class: vase, score: 0.272799, bbox: [242.1, 197.477, 253.213, 212.714]
Object class: bowl, score: 0.267025, bbox: [461.055, 213.221, 475.82, 221.931]
Object class: dining table, score: 0.261253, bbox: [461.641, 350.519, 639.609, 425.35]
Object class: wine glass, score: 0.256929, bbox: [361.777, 218.239, 373.848, 232.306]
Object class: toaster, score: 0.252904, bbox: [512.969, 206.11, 527.031, 221.554]
Object class: wine glass, score: 0.25165, bbox: [166.719, 232.839, 186.406, 267.628]
Object class: microwave, score: 0.251464, bbox: [512.969, 206.11, 527.031, 221.554]
Object class: microwave, score: 0.242385, bbox: [496.348, 140.925, 525.527, 174.831]
Object class: chair, score: 0.240596, bbox: [343.75, 216.172, 392.5, 305.095]
Object class: dining table, score: 0.23661, bbox: [484.023, 353.587, 639.727, 425.61]
Object class: tv, score: 0.236164, bbox: [557.695, 207.436, 639.805, 288.871]
Object class: dining table, score: 0.230219, bbox: [0.15625, 261.882, 216.406, 338.845]
Object class: chair, score: 0.229535, bbox: [372.383, 217.004, 417.617, 299.687]
Object class: book, score: 0.223886, bbox: [140.508, 278.821, 196.992, 292.368]
Object class: handbag, score: 0.221784, bbox: [554.648, 290.535, 575.977, 331.408]
Object class: sink, score: 0.219839, bbox: [512.969, 220.683, 538.281, 223.621]
Object class: remote, score: 0.219337, bbox: [90.293, 327.17, 109.395, 334.295]
Object class: microwave, score: 0.213705, bbox: [557.695, 207.436, 639.805, 288.871]
Object class: bottle, score: 0.213705, bbox: [491.377, 153.354, 502.998, 173.635]
Object class: potted plant, score: 0.210407, bbox: [4.29688, 38.7415, 173.047, 129.537]
Object class: book, score: 0.207375, bbox: [0.15625, 282.553, 19.5703, 290.301]
Object class: bottle, score: 0.206414, bbox: [397.08, 199.87, 407.92, 216.77]
Object class: oven, score: 0.206094, bbox: [547.695, 139.365, 639.805, 292.459]
Object class: potted plant, score: 0.203975, bbox: [342.93, 175.767, 380.195, 230.265]
Object class: tie, score: 0.203707, bbox: [6.21094, 165.886, 154.102, 272.178]
Object class: hair drier, score: 0.201487, bbox: [425.996, 157.176, 445.879, 195.189]



# 数据集精度统计结果
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.452
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.604
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.496
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.316
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.584
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.376
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.660
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.730
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.579
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.769
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.879
{'bbox_mAP': 0.452, 'bbox_mAP_50': 0.604, 'bbox_mAP_75': 0.496, 'bbox_mAP_s': 0.316, 'bbox_mAP_m': 0.485, 'bbox_mAP_l': 0.584, 'bbox_mAP_copypaste': '0.452 0.604 0.496 0.316 0.485 0.584'}
```

### grounding_dino_text_enc_prof 命令行参数说明
```bash
options:
  -m, --model_prefix     model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/groundingdino_text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod])
      --hw_config        hw-config file of the model suite (string [=])
      --vdsp_params      vdsp preprocess parameter file (string [=../data/configs/clip_txt_vdsp.json])
  -d, --device_ids       device id to run (string [=[0]])
  -b, --batch_size       profiling batch size of the model (unsigned int [=1])
  -i, --instance         instance number or range for each device (unsigned int [=1])
      --iterations       iterations count for one profiling (int [=1024])
      --percentiles      percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host       cache input data into host memory (bool [=0])
  -q, --queue_size       aync wait queue size (unsigned int [=2])
      --test_npz_file    npz_file for test (string [=])
  -?, --help             print this message
```
### grounding_dino_text_enc_prof 命令行示例
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/grounding_dino_text_enc_prof \
-m /opt/vastai/vaststreamx/data/models/groundingdino/text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod \
--vdsp_params ../data/configs/clip_txt_vdsp.json \
--test_npz_file input_tokens.npz \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 1000 \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 1 


# 测试最小时延
./vaststreamx-samples/bin/grounding_dino_text_enc_prof \
-m /opt/vastai/vaststreamx/data/models/groundingdino/text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod \
--vdsp_params ../data/configs/clip_txt_vdsp.json \
--test_npz_file input_tokens.npz \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 1000 \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 0 
```

### grounding_dino_text_enc_prof 命令行结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 139.599
  latency (us):
    avg latency: 21407
    min latency: 9325
    max latency: 23389
    p50 latency: 21422
    p90 latency: 21465
    p95 latency: 21475
    p99 latency: 21493

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 115.282
  latency (us):
    avg latency: 8673
    min latency: 8647
    max latency: 9124
    p50 latency: 8656
    p90 latency: 8718
    p95 latency: 8722
    p99 latency: 8729
```


### grounding_dino_image_enc_prof 命令行参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=./data/configs/groundingdino_bgr888.json])
  -d, --device_ids      device id to run (string [=[0]])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        instance number or range for each device (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (int [=10240])
      --percentiles     percentiles of latency (string [=[50,90,95,99]])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=2])
  -?, --help            print this message
```
### grounding_dino_image_enc_prof 命令行示例
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/grounding_dino_image_enc_prof \
-m /opt/vastai/vaststreamx/data/models/groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod \
--vdsp_params ../data/configs/groundingdino_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 10 \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 1 


# 测试最小时延
./vaststreamx-samples/bin/grounding_dino_image_enc_prof \
-m /opt/vastai/vaststreamx/data/models/groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod \
--vdsp_params ../data/configs/groundingdino_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 10 \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 0  
```

### grounding_dino_image_enc_prof 命令行结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 2.35091
  latency (us):
    avg latency: 895865
    min latency: 426437
    max latency: 1287392
    p50 latency: 865801
    p90 latency: 1272686
    p95 latency: 1287392
    p99 latency: 1287392


# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 2.3332
  latency (us):
    avg latency: 428594
    min latency: 425583
    max latency: 445934
    p50 latency: 426903
    p90 latency: 427407
    p95 latency: 445934
    p99 latency: 445934
```
## Python Sample

### 脚本 grounding_dino.py 参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  --imgmod_prefix IMGMOD_PREFIX
                        image model prefix of the model suite files
  --imgmod_hw_config IMGMOD_HW_CONFIG
                        image model hw-config file of the model suite
  --imgmod_vdsp_params IMGMOD_VDSP_PARAMS
                        vdsp preprocess parameter file
  --txtmod_prefix TXTMOD_PREFIX
                        text model prefix of the model suite files
  --txtmod_vdsp_params TXTMOD_VDSP_PARAMS
                        text model vdsp preprocess parameter file
  --txtmod_hw_config TXTMOD_HW_CONFIG
                        text model hw-config file of the model suite
  --decmod_prefix DECMOD_PREFIX
                        text model vdsp preprocess parameter file
  --decmod_hw_config DECMOD_HW_CONFIG
                        text model hw-config file of the model suite
  --tokenizer_path TOKENIZER_PATH
                        tokenizer path
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --threshold THRESHOLD
                        object confidence threshold
  --label_file LABEL_FILE
                        label file
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file
  --dataset_filelist DATASET_FILELIST
                        input dataset filelist
  --dataset_root DATASET_ROOT
                        input dataset root
  --dataset_output_folder DATASET_OUTPUT_FOLDER
                        dataset output folder path
```
### 脚本 grounding_dino.py 示例
```bash
# 单张图片测试
python3 grounding_dino.py \
--txtmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod \
--txtmod_vdsp_params  ../../../data/configs/clip_txt_vdsp.json \
--imgmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod \
--imgmod_vdsp_params ../../../data/configs/groundingdino_bgr888.json \
--decmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/decoder-fp16-none-1_22223_256_1_195_256_1_195_1_195_1_195_195-vacc/mod \
--tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/bert-base-uncased \
--label_file ../../../data/labels/coco2id.txt \
--device_id  0 \
--threshold 0.2 \
--input_file /opt/vastai/vaststreamx/data/datasets/det_coco_val/000000000139.jpg  \
--output_file grounding_dino_result.jpg

# 数据集测试
mkdir -p ./grounding_dino_out
python3 grounding_dino.py \
--txtmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod \
--txtmod_vdsp_params  ../../../data/configs/clip_txt_vdsp.json \
--imgmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod \
--imgmod_vdsp_params ../../../data/configs/groundingdino_bgr888.json \
--decmod_prefix /opt/vastai/vaststreamx/data/models/groundingdino/decoder-fp16-none-1_22223_256_1_195_256_1_195_1_195_1_195_195-vacc/mod \
--tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/bert-base-uncased \
--label_file ../../../data/labels/coco2id.txt \
--device_id  0 \
--threshold 0.01 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/det_coco_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder ./grounding_dino_out

# 精度统计
python3 ../../../evaluation/detection/eval_map.py \
--gt ../../../evaluation/detection/instances_val2017.json \
--txt ./grounding_dino_out
```
### 脚本 grounding_dino.py 结果示例
```bash
# 单张图片测试
Detection objects:
Object class: tv, score: 0.823043, bbox:[ 6.21,165.89,154.10,272.18 ]
Object class: person, score: 0.808823, bbox:[ 409.45,157.25,465.55,296.20 ]
Object class: clock, score: 0.627951, bbox:[ 447.05,119.68,461.70,142.20 ]
Object class: person, score: 0.625666, bbox:[ 385.20,172.70,400.43,206.71 ]
Object class: vase, score: 0.558928, bbox:[ 550.43,299.45,587.07,401.12 ]
Object class: potted plant, score: 0.546747, bbox:[ 232.85,175.77,265.90,212.79 ]
Object class: chair, score: 0.516412, bbox:[ 290.08,217.99,348.36,318.25 ]
Object class: chair, score: 0.506061, bbox:[ 361.33,219.16,415.55,317.91 ]
Object class: vase, score: 0.504005, bbox:[ 350.47,206.34,362.03,231.30 ]
Object class: chair, score: 0.480234, bbox:[ 302.81,217.29,352.19,305.64 ]
Object class: chair, score: 0.458136, bbox:[ 388.71,219.84,441.91,304.34 ]
Object class: wine glass, score: 0.447178, bbox:[ 313.55,191.55,319.88,214.07 ]
Object class: wine glass, score: 0.439137, bbox:[ 318.70,192.17,324.42,213.86 ]
Object class: chair, score: 0.386589, bbox:[ 402.73,220.20,442.27,305.23 ]
Object class: wine glass, score: 0.367332, bbox:[ 361.35,214.56,373.65,232.24 ]
Object class: bowl, score: 0.364777, bbox:[ 465.89,215.65,475.36,221.58 ]
Object class: cup, score: 0.361277, bbox:[ 166.72,232.84,186.41,267.63 ]
Object class: microwave, score: 0.331011, bbox:[ 475.23,136.69,525.39,175.12 ]
Object class: refrigerator, score: 0.330687, bbox:[ 446.13,167.55,513.24,287.15 ]
Object class: bottle, score: 0.319227, bbox:[ 491.37,153.77,496.13,171.97 ]
Object class: dining table, score: 0.313685, bbox:[ 310.98,228.96,448.40,317.68 ]
Object class: refrigerator, score: 0.297572, bbox:[ 490.74,173.22,513.01,285.23 ]
Object class: bottle, score: 0.296247, bbox:[ 496.20,153.91,502.55,173.49 ]
Object class: hair drier, score: 0.296168, bbox:[ 425.84,157.25,445.41,178.89 ]
Object class: refrigerator, score: 0.293809, bbox:[ 479.10,171.50,512.77,286.11 ]
Object class: handbag, score: 0.292760, bbox:[ 212.27,299.56,257.11,328.21 ]
Object class: oven, score: 0.287567, bbox:[ 557.70,207.44,639.80,288.87 ]
Object class: vase, score: 0.272799, bbox:[ 242.10,197.48,253.21,212.71 ]
Object class: bowl, score: 0.267025, bbox:[ 461.05,213.22,475.82,221.93 ]
Object class: dining table, score: 0.261253, bbox:[ 461.64,350.52,639.61,425.35 ]
Object class: wine glass, score: 0.256929, bbox:[ 361.78,218.24,373.85,232.31 ]
Object class: toaster, score: 0.252904, bbox:[ 512.97,206.11,527.03,221.55 ]
Object class: wine glass, score: 0.251650, bbox:[ 166.72,232.84,186.41,267.63 ]
Object class: microwave, score: 0.251464, bbox:[ 512.97,206.11,527.03,221.55 ]
Object class: microwave, score: 0.242385, bbox:[ 496.35,140.93,525.53,174.83 ]
Object class: chair, score: 0.240596, bbox:[ 343.75,216.17,392.50,305.10 ]
Object class: dining table, score: 0.236610, bbox:[ 484.02,353.59,639.73,425.61 ]
Object class: tv, score: 0.236164, bbox:[ 557.70,207.44,639.80,288.87 ]
Object class: dining table, score: 0.230219, bbox:[ 0.16,261.88,216.41,338.84 ]
Object class: chair, score: 0.229535, bbox:[ 372.38,217.00,417.62,299.69 ]
Object class: book, score: 0.223886, bbox:[ 140.51,278.82,196.99,292.37 ]
Object class: handbag, score: 0.221784, bbox:[ 554.65,290.53,575.98,331.41 ]
Object class: sink, score: 0.219839, bbox:[ 512.97,220.68,538.28,223.62 ]
Object class: remote, score: 0.219337, bbox:[ 90.29,327.17,109.39,334.29 ]
Object class: microwave, score: 0.213705, bbox:[ 557.70,207.44,639.80,288.87 ]
Object class: bottle, score: 0.213705, bbox:[ 491.38,153.35,503.00,173.63 ]
Object class: potted plant, score: 0.210407, bbox:[ 4.30,38.74,173.05,129.54 ]
Object class: book, score: 0.207375, bbox:[ 0.16,282.55,19.57,290.30 ]
Object class: bottle, score: 0.206414, bbox:[ 397.08,199.87,407.92,216.77 ]
Object class: oven, score: 0.206094, bbox:[ 547.70,139.37,639.80,292.46 ]
Object class: potted plant, score: 0.203975, bbox:[ 342.93,175.77,380.20,230.26 ]
Object class: tie, score: 0.203707, bbox:[ 6.21,165.89,154.10,272.18 ]
Object class: hair drier, score: 0.201487, bbox:[ 426.00,157.18,445.88,195.19 ]


# 数据集精度统计结果
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.452
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.603
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.495
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.316
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.483
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.584
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.376
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.660
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.730
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.578
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.769
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.880
{'bbox_mAP': 0.452, 'bbox_mAP_50': 0.603, 'bbox_mAP_75': 0.495, 'bbox_mAP_s': 0.316, 'bbox_mAP_m': 0.483, 'bbox_mAP_l': 0.584, 'bbox_mAP_copypaste': '0.452 0.603 0.495 0.316 0.483 0.584'}

```


### grounding_dino_text_enc_prof.py 参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  --vdsp_params VDSP_PARAMS
                        vdsp preprocess parameter file
  --tokenizer_path TOKENIZER_PATH
                        tokenizer path
  -d DEVICE_IDS, --device_ids DEVICE_IDS
                        device ids to run
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        profiling batch size of the model
  -i INSTANCE, --instance INSTANCE
                        instance number for each device
  --iterations ITERATIONS
                        iterations count for one profiling
  --queue_size QUEUE_SIZE
                        aync wait queue size
  --percentiles PERCENTILES
                        percentiles of latency
  --input_host INPUT_HOST
                        cache input data into host memory
```
### grounding_dino_text_enc_prof.py 命令行示例
```bash
# 测试最大吞吐
python3 grounding_dino_text_enc_prof.py \
-m /opt/vastai/vaststreamx/data/models/groundingdino/text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod \
--vdsp_params ../../../data/configs/clip_txt_vdsp.json \
--tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/bert-base-uncased \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 1000 \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 1 


# 测试最小时延
python3 grounding_dino_text_enc_prof.py \
-m /opt/vastai/vaststreamx/data/models/groundingdino/text_encoder-fp16-none-1_195_1_195_1_195_1_195_195-vacc/mod \
--vdsp_params ../../../data/configs/clip_txt_vdsp.json \
--tokenizer_path /opt/vastai/vaststreamx/data/tokenizer/bert-base-uncased \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 1000 \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 0 
```

### grounding_dino_text_enc_prof.py  命令行结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 139.58
  latency (us):
    avg latency: 21440
    min latency: 10407
    max latency: 23563
    p50 latency: 21462
    p90 latency: 21481
    p95 latency: 21490
    p99 latency: 21526

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 111.48
  latency (us):
    avg latency: 8969
    min latency: 8933
    max latency: 9635
    p50 latency: 8951
    p90 latency: 9011
    p95 latency: 9017
    p99 latency: 9075
```


### grounding_dino_image_enc_prof.py 参数说明
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
                        device id to run
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
### grounding_dino_image_enc_prof.py 命令行示例
```bash
# 测试最大吞吐
python3 grounding_dino_image_enc_prof.py \
-m /opt/vastai/vaststreamx/data/models/groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod \
--vdsp_params ../../../data/configs/groundingdino_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 10 \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 1 


# 测试最小时延
python3 grounding_dino_image_enc_prof.py \
-m /opt/vastai/vaststreamx/data/models/groundingdino/img_encoder-fp16-none-1_3_800_1333-vacc/mod \
--vdsp_params ../../../data/configs/groundingdino_bgr888.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 10 \
--percentiles [50,90,95,99] \
--input_host 1 \
--queue_size 0 
```

### grounding_dino_image_enc_prof 命令行结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 2.36
  latency (us):
    avg latency: 1144590
    min latency: 445181
    max latency: 1286246
    p50 latency: 1263684
    p90 latency: 1268980
    p95 latency: 1277613
    p99 latency: 1284519


# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 2.34
  latency (us):
    avg latency: 428061
    min latency: 425125
    max latency: 445614
    p50 latency: 426225
    p90 latency: 429177
    p95 latency: 437395
    p99 latency: 443970
```