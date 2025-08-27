# Text Detection sample 

本目录提供基于 dbnet 模型的 文字检测  sample


## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/algorithm_det_db.md)  [modelzoo](https://github.com/Vastai/VastModelZOO/tree/main/cv/text_detection/dbnet) |
|  输入 shape |   [ (1,3,736,1280) ]     |
| INT8量化方式 |   kl_divergence        |
|  官方精度 | "precision": 0.8641, "recall": 0.7872, "Hmean": 0.8238 |
|  VACC FP16  精度 | 'precision': 0.8096, 'recall': 0.8209, 'hmean': 0.8152  |
|  VACC INT8  精度 | "precision": 0.837, "recall":  0.8011, "Hmean": 0.8187  |


## 数据准备

下载模型 dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 ch4_test_images 到 /opt/vastai/vaststreamx/data/datasets 里
下载 elf 压缩包到 /opt/vastai/vaststreamx/data/


## C++ sample

### dbnet 命令行参数说明
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

### dbnet 命令行示例

在build 目录里执行
单图片示例
```bash
./vaststreamx-samples/bin/dbnet \
-m /opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod \
--vdsp_params ../data/configs/dbnet_rgbplanar.json \
--device_id 0 \
--threshold 0.3 \
--box_unclip_ratio 1.5 \
--use_polygon_score 0 \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--input_file ../data/images/detect.jpg \
--output_file dbnet_result.jpg
```
输出
```bash
index:0, score:0.855256,bbox:[ [660 81] [699 84] [698 97] [659 94] ]
index:1, score:0.74898,bbox:[ [673 103] [689 103] [689 110] [673 110] ]
index:2, score:0.753123,bbox:[ [637 134] [664 137] [662 151] [636 149] ]
index:3, score:0.7569,bbox:[ [661 134] [725 138] [724 156] [660 152] ]
index:4, score:0.820094,bbox:[ [632 151] [663 151] [663 170] [632 170] ]
index:5, score:0.742128,bbox:[ [665 153] [695 155] [694 171] [664 169] ]
index:6, score:0.675795,bbox:[ [914 251] [968 253] [968 263] [914 261] ]
index:7, score:0.834535,bbox:[ [787 279] [820 281] [819 299] [786 297] ]
index:8, score:0.887785,bbox:[ [822 284] [873 284] [873 298] [822 298] ]
index:9, score:0.86458,bbox:[ [873 284] [903 284] [903 299] [873 299] ]
index:10, score:0.889592,bbox:[ [788 300] [904 300] [904 334] [788 334] ]
index:11, score:0.916381,bbox:[ [790 334] [866 334] [866 362] [790 362] ]
index:12, score:0.787945,bbox:[ [854 453] [903 453] [903 467] [854 467] ]
index:13, score:0.712549,bbox:[ [868 536] [877 536] [877 541] [868 541] ]
```
并在图片上画出检测框，保存到  dbnet_result.jpg

测试数据集
```bash
mkdir -p dbnet_output
./vaststreamx-samples/bin/dbnet \
-m /opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod \
--vdsp_params ../data/configs/dbnet_rgbplanar.json \
--device_id 0 \
--threshold 0.3 \
--box_unclip_ratio 1.5 \
--use_polygon_score 0 \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder dbnet_output
```
结果保存在 dbnet_output 文件夹里

统计精度
```bash
python3 ../evaluation/text_detection/eval.py \
--test_image_path  /opt/vastai/vaststreamx/data/datasets/ch4_test_images \
--boxes_npz_dir ./dbnet_output \
--label_file ../data/labels/test_icdar2015_label.txt 
```
精度结果
```
metric:  {'precision': 0.8370221327967807, 'recall': 0.8011555127587867, 'hmean': 0.8186961869618696}
```

### dbnet_prof 命令行参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod])
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
### dbnet_prof 命令行示例

```bash
# 测试最大吞吐
./vaststreamx-samples/bin/dbnet_prof \
-m /opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod \
--vdsp_params ../data/configs/dbnet_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 600 \
--shape "[3,736,1280]" \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--queue_size 1


# 测试最小时延
./vaststreamx-samples/bin/dbnet_prof \
-m /opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod \
--vdsp_params ../data/configs/dbnet_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--iterations 300 \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--queue_size 0
```

### dbnet_prof 命令行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 100.027
  latency (us):
    avg latency: 29881
    min latency: 24550
    max latency: 42771
    p50 latency: 29885
    p90 latency: 30086
    p95 latency: 30152
    p99 latency: 30286


# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 57.2287
  latency (us):
    avg latency: 17472
    min latency: 17299
    max latency: 24807
    p50 latency: 17446
    p90 latency: 17568
    p95 latency: 17588
    p99 latency: 17682
```



## Python sample 功能测试

### dbnet.py 命令行参数说明
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

### dbnet.py 运行示例

在本目录下运行  
```bash
python3 dbnet.py \
-m /opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod \
--vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_id 0 \
--input_file ../../data/images/detect.jpg \
--output_file dbnet_result.jpg
```

### dbnet.py 运行结果示例

终端显示 检测到的文字的 bbox 多边形的四个角的坐标，bbox也画在图片上并保存为 dbnet_result.jpg

```bash
index:0, score:0.8552564713398266,bbox:[[660  81],[699  85],[698  98],[659  94]]
index:1, score:0.7489797152005709,bbox:[[673 104],[689 104],[689 111],[673 111]]
index:2, score:0.7531229985224737,bbox:[[637 135],[664 137],[662 152],[636 149]]
index:3, score:0.7568998821711136,bbox:[[661 135],[725 139],[724 157],[660 152]]
index:4, score:0.8200937444513494,bbox:[[632 152],[663 152],[663 170],[632 170]]
index:5, score:0.7421281688709549,bbox:[[665 153],[695 156],[694 172],[664 169]]
index:6, score:0.6757951846792678,bbox:[[914 251],[968 254],[968 263],[914 261]]
index:7, score:0.8345349513062644,bbox:[[787 279],[820 282],[819 300],[786 297]]
index:8, score:0.8877850321980266,bbox:[[822 285],[873 285],[873 298],[822 298]]
index:9, score:0.8645803202753481,bbox:[[873 285],[903 285],[903 299],[873 299]]
index:10, score:0.8895922985273538,bbox:[[788 300],[904 300],[904 335],[788 335]]
index:11, score:0.9163814249865065,bbox:[[790 335],[866 335],[866 363],[790 363]]
index:12, score:0.7879449234527796,bbox:[[854 454],[903 454],[903 468],[854 468]]
index:13, score:0.7125489371163504,bbox:[[868 536],[877 536],[877 542],[868 542]]
```


测试数据集
```bash
mkdir -p dbnet_output
python3 dbnet.py  \
-m /opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod \
--vdsp_params ../../data/configs/dbnet_rgbplanar.json \
--elf_file /opt/vastai/vaststreamx/data/elf/find_contours_ext_op \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ch4_test_images_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_folder dbnet_output
```
结果保存在 dbnet_output 文件夹里

```bash
# 用刚才保存的npz文件测试精度
python3 ../../evaluation/text_detection/eval.py \
--test_image_path  /opt/vastai/vaststreamx/data/datasets/ch4_test_images \
--boxes_npz_dir ./dbnet_output \
--label_file ../../data/labels/test_icdar2015_label.txt 
```
精度结果
```
metric:  {'precision': 0.8361809045226131, 'recall': 0.8011555127587867, 'hmean': 0.8182935824932384}
```

### dbnet_prof.py 命令行参数说明

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


### dbnet_prof.py 运行示例

在本目录下运行  
```bash
# 测试最大吞吐
python3 dbnet_prof.py \
-m /opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod \
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
python3 dbnet_prof.py \
-m /opt/vastai/vaststreamx/data/models/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc/mod \
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


### dbnet_prof.py 运行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 99.81
  latency (us):
    avg latency: 29995
    min latency: 20402
    max latency: 37946
    p50 latency: 30007
    p90 latency: 31402
    p95 latency: 31630
    p99 latency: 31813

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 59.38
  latency (us):
    avg latency: 16839
    min latency: 16553
    max latency: 19965
    p50 latency: 16830
    p90 latency: 16954
    p95 latency: 17015
    p99 latency: 17488
```
