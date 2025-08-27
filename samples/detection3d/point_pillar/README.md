# 3D Detection Sample 

本sample基于 point_pillar 算法实现 3d 目标检测


## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/nutonomy/second.pytorch)  [modelzoo](-) |
|  输入 shape |   [ (16000,32,10,3,16000,1,16000) ]     |
| INT8量化方式 |   percentile         |
|  官方精度 |  "Car@0.7":86.4516/77.2855/74.6538, "Pedestrian@0.5":57.7573/52.3014/47.9166, "Cyclist@0.5":79.9918/62.6580/59.6744 |
|  VACC FP16  精度 | - |
|  VACC INT8  精度 | "Car@0.7":85.2757/73.1001/67.9215, "Pedestrian@0.5":56.8780/51.0442/47.9645, "Cyclist@0.5":78.0941/60.7013/57.5004 |


## 数据准备
下载 pointpillar-int8-percentile-16000_32_10_3_16000_1_16000-vacc 模型到 /opt/vastai/vaststreamx/data/models/
下载数据集 fov_pointcloud_float16 到 /opt/vastai/vaststreamx/data/datasets


## C++ Sample 

### point_pillar 命令行参数说明
```bash
options:
  -m, --model_prefixs            model prefixs of the model suite files (string [=[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]])
      --hw_configs               hw-config file of the model suite (string [=[]])
      --elf_file                 elf file path (string [=/opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op])
  -d, --device_id                device id to run (unsigned int [=0])
      --max_points_num           max_points_num to run (unsigned int [=120000])
      --max_voxel_num            model max voxel number (string [=[16000]])
      --voxel_size               model max voxel number (string [=[0.16, 0.16, 4]])
      --coors_range              model max voxel number (string [=[0, -39.68, -3, 69.12, 39.68, 1]])
      --feat_size                set model feature sizes,[max_feature_width,max_feature_height,actual_feature_width,actual_feature_height] (string [=[864,496,480,480]])
      --input_file               input file (string [=/opt/vastai/vaststreamx/data/datasets/fov_pointcloud_float16/000001.bin])
      --shuffle_enabled          shuffle enabled (unsigned int [=0])
      --normalize_enabled        normalize enabled (unsigned int [=0])
      --dataset_filelist         dataset filename list (string [=])
      --dataset_root             dataset root (string [=])
      --dataset_output_folder    dataset output folder path (string [=])
  -?, --help                     print this message
```


### point_pillar 命令行示例

```bash
#测试单个文件
./vaststreamx-samples/bin/point_pillar \
-m "[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]" \
--elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
--max_voxel_num [16000] \
--voxel_size [0.16,0.16,4] \
--coors_range [0,-39.68,-3,69.12,39.68,1] \
--max_points_num 12000000 \
--shuffle_enabled 0 \
--normalize_enabled 0 \
--feat_size [864,496,480,480] \
--device_id 0 \
--input_file /opt/vastai/vaststreamx/data/datasets/kitti_val/fov_pointcloud_float16/000001.bin 


#测试数据集
mkdir -p pointpillar_out
./vaststreamx-samples/bin/point_pillar \
-m "[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]" \
--elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
--max_voxel_num [16000] \
--voxel_size [0.16,0.16,4] \
--coors_range [0,-39.68,-3,69.12,39.68,1] \
--max_points_num 12000000 \
--shuffle_enabled 0 \
--normalize_enabled 0 \
--feat_size [864,496,480,480] \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/kitti_val/fov_pointcloud_float16_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/kitti_val/ \
--dataset_output_folder pointpillar_out

# 统计精度
python3 ../evaluation/point_pillar/evaluation.py \
--out_dir pointpillar_out

```
### point_pillar 命令行结果示例

```bash
# 测试单个文件结果
label: 1, score: 0.488281, box:[ 29.875 -7.1875 -0.68457 3.84375 1.54785 1.46094 6.17578 ]
label: 3, score: 0.395996, box:[ 46.2812 -4.61328 -0.144043 1.69727 0.401367 1.74902 6.53125 ]
label: 1, score: 0.331055, box:[ 59.0938 16.5781 -1.06934 3.97656 1.61523 1.53516 6.38281 ]
label: 1, score: 0.324707, box:[ 46.7812 23.5312 -1.11523 3.81836 1.62109 1.48145 3.07812 ]
label: 3, score: 0.147949, box:[ 38.375 19.5156 -1.19531 1.52051 0.391846 1.66504 6.25391 ]
label: 2, score: 0.14502, box:[ 29.2969 24.5469 -1.6416 0.500488 0.557129 1.64453 1.11719 ]
label: 1, score: 0.142578, box:[ 11.4141 -5.4375 -0.828125 3.93945 1.58594 1.47266 6.41797 ]
label: 1, score: 0.137207, box:[ 39.7812 25.1875 -1.42676 3.59961 1.55859 1.47559 3.64062 ]
label: 2, score: 0.129395, box:[ 15.4844 -7.30469 -0.577148 0.730957 0.560547 1.78125 3.14648 ]
label: 2, score: 0.121094, box:[ 12.8438 -8.17188 -0.647461 0.647949 0.543945 1.72363 2.55859 ]
label: 1, score: 0.106445, box:[ 51.1562 -0.228271 -0.506836 4.27734 1.65918 1.50391 4.49219 ]
label: 3, score: 0.106445, box:[ 33.6875 24.5625 -1.04688 1.83203 0.589355 1.71973 3.52734 ]

# 精度统计结果
eval image 2D bounding boxes
car AP: 93.828209 88.676392 86.894783
car AP: 93.802193 88.457069 86.541924
pedestrian AP: 67.774147 63.358284 59.826515
pedestrian AP: 48.347351 45.521614 42.858315
cyclist AP: 84.695000 70.831528 67.571785
cyclist AP: 84.488907 69.354836 66.034897

eval bird's eye view bounding boxes
car AP: 89.124992 83.367134 82.288109
pedestrian AP: 63.745148 57.575775 53.721924
cyclist AP: 82.019577 66.072533 61.496628

eval 3D bounding boxes
car AP: 85.236931 71.718178 67.845100
pedestrian AP: 56.159348 50.777905 47.113194
cyclist AP: 77.871696 60.811256 57.609837
```

### point_pillar_prof 命令行参数说明

```bash
options:
  -m, --model_prefixs        model prefix of the model suite files (string [=[/opt/vastai/vaststreamx/data/models/pointpillar-int8-percentile-16000_32_10_3_16000_1_16000-vacc/mod]])
      --hw_configs           hw-config file of the model suite (string [=[]])
      --elf_file             elf file path (string [=/opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op])
      --max_voxel_num        model max voxel number (string [=[16000]])
      --max_points_num       max_points_num to run (unsigned int [=120000])
      --voxel_size           model max voxel number (string [=[0.16, 0.16, 4]])
      --coors_range          model max voxel number (string [=[0, -39.68, -3, 69.12, 39.68, 1]])
      --shuffle_enabled      shuffle enabled (unsigned int [=0])
      --normalize_enabled    normalize enabled (unsigned int [=0])
      --dataset_filelist     dataset filename list (string [=])
  -d, --device_ids           device id to run (string [=[0]])
  -b, --batch_size           profiling batch size of the model (unsigned int [=1])
  -i, --instance             instance number or range for each device (unsigned int [=1])
  -s, --shape                model input shape (string)
      --iterations           iterations count for one profiling (int [=10240])
      --percentiles          percentiles of latency (string [=[50,90,95,99]])
      --input_host           cache input data into host memory (bool [=0])
  -q, --queue_size           aync wait queue size (unsigned int [=1])
  -?, --help                 print this message
```
### point_pillar_prof 命令行示例
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/point_pillar_prof \
-m "[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]" \
--elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
--max_voxel_num [16000] \
--voxel_size [0.16,0.16,4] \
--coors_range [0,-39.68,-3,69.12,39.68,1] \
--shuffle_enabled 0 \
--normalize_enabled 0 \
--max_points_num 12000000 \
--feat_size [864,496,480,480] \
--device_ids [0] \
--shape [40000] \
--batch_size 1 \
--instance 1 \
--iterations 1500 \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/point_pillar_prof \
-m "[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]" \
--elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
--max_voxel_num [16000] \
--max_points_num 12000000 \
--voxel_size [0.16,0.16,4] \
--coors_range [0,-39.68,-3,69.12,39.68,1] \
--shuffle_enabled 0 \
--normalize_enabled 0 \
--feat_size [864,496,480,480] \
--device_ids [0] \
--shape [40000] \
--batch_size 1 \
--instance 1 \
--iterations 1000 \
--input_host 1 \
--queue_size 0
```

### point_pillar_prof 命令行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 249.715
  latency (us):
    avg latency: 11964
    min latency: 6087
    max latency: 14036
    p50 latency: 11973
    p90 latency: 11994
    p95 latency: 12000
    p99 latency: 12009

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 178.319
  latency (us):
    avg latency: 5607
    min latency: 5597
    max latency: 5796
    p50 latency: 5606
    p90 latency: 5613
    p95 latency: 5618
    p99 latency: 5636
```

## Python Sample

###  point_pillar.py 参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIXS, --model_prefixs MODEL_PREFIXS
                        model prefix of the model suite files
  --hw_configs HW_CONFIGS
                        hw-config file of the model suite
  --max_voxel_num MAX_VOXEL_NUM
                        model max voxel number
  --voxel_size VOXEL_SIZE
                        voxel size
  --coors_range COORS_RANGE
                        coors range
  --elf_file ELF_FILE   elf file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --max_points_num MAX_POINTS_NUM
                        max points number per input
  --shuffle_enabled SHUFFLE_ENABLED
                        shuffle enabled
  --normalize_enabled NORMALIZE_ENABLED
                        normalize enabled
  --input_file INPUT_FILE
                        input file
  --dataset_filelist DATASET_FILELIST
                        dataset filename list
  --dataset_root DATASET_ROOT
                        dataset root
  --dataset_output_folder DATASET_OUTPUT_FOLDER
                        dataset output folder path
```

###  point_pillar.py 命令示例
```bash
# 测试单个输入
python3  point_pillar.py \
-m "[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]" \
--elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
--max_voxel_num [16000] \
--voxel_size [0.16,0.16,4] \
--coors_range [0,-39.68,-3,69.12,39.68,1] \
--shuffle_enabled 0 \
--normalize_enabled 0 \
--max_points_num 12000000 \
--feat_size [864,496,480,480] \
--device_id 0 \
--input_file /opt/vastai/vaststreamx/data/datasets/kitti_val/fov_pointcloud_float16/000001.bin 

# 测试数据集
mkdir -p pointpillar_out
python3  point_pillar.py \
-m "[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]" \
--elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
--max_voxel_num [16000] \
--voxel_size [0.16,0.16,4] \
--coors_range [0,-39.68,-3,69.12,39.68,1] \
--shuffle_enabled 0 \
--normalize_enabled 0 \
--max_points_num 12000000 \
--feat_size [864,496,480,480] \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/kitti_val/fov_pointcloud_float16_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/kitti_val/ \
--dataset_output_folder pointpillar_out

# 统计精度
python3 ../../../evaluation/point_pillar/evaluation.py \
--out_dir pointpillar_out
```

```bash
# 测试单个文件结果
label: 1, score: 0.488281, box:[29.875  -7.1875 -0.6846  3.8438  1.5479  1.4609  6.1758]
label: 3, score: 0.395996, box:[46.2812 -4.6133 -0.144   1.6973  0.4014  1.749   6.5312]
label: 1, score: 0.331055, box:[59.0938 16.5781 -1.0693  3.9766  1.6152  1.5352  6.3828]
label: 1, score: 0.324707, box:[46.7812 23.5312 -1.1152  3.8184  1.6211  1.4814  3.0781]
label: 3, score: 0.147949, box:[38.375  19.5156 -1.1953  1.5205  0.3918  1.665   6.2539]
label: 2, score: 0.145020, box:[29.2969 24.5469 -1.6416  0.5005  0.5571  1.6445  1.1172]
label: 1, score: 0.142578, box:[11.4141 -5.4375 -0.8281  3.9395  1.5859  1.4727  6.418 ]
label: 1, score: 0.137207, box:[39.7812 25.1875 -1.4268  3.5996  1.5586  1.4756  3.6406]
label: 2, score: 0.129395, box:[15.4844 -7.3047 -0.5771  0.731   0.5605  1.7812  3.1465]
label: 2, score: 0.121094, box:[12.8438 -8.1719 -0.6475  0.6479  0.5439  1.7236  2.5586]
label: 1, score: 0.106445, box:[51.1562 -0.2283 -0.5068  4.2773  1.6592  1.5039  4.4922]
label: 3, score: 0.106445, box:[33.6875 24.5625 -1.0469  1.832   0.5894  1.7197  3.5273]

# 精度统计结果
eval image 2D bounding boxes
car AP: 93.828209 88.676392 86.894783
car AP: 93.802193 88.457069 86.541924
pedestrian AP: 67.774147 63.358284 59.826515
pedestrian AP: 48.347351 45.521614 42.858315
cyclist AP: 84.695000 70.831528 67.571785
cyclist AP: 84.488907 69.354836 66.034897

eval bird's eye view bounding boxes
car AP: 89.124992 83.367134 82.288109
pedestrian AP: 63.745148 57.575775 53.721924
cyclist AP: 82.019577 66.072533 61.496628

eval 3D bounding boxes
car AP: 85.236931 71.718178 67.845100
pedestrian AP: 56.159348 50.777905 47.113194
cyclist AP: 77.871696 60.811256 57.609837
```

###  point_pillar_prof.py 参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIXS, --model_prefixs MODEL_PREFIXS
                        model prefix of the model suite
                        files
  --hw_configs HW_CONFIGS
                        hw-config file of the model suite
  --elf_file ELF_FILE   elf file path
  --max_voxel_num MAX_VOXEL_NUM
                        model max voxel number
  --max_points_num MAX_POINTS_NUM
                        max_points_num to run
  --voxel_size VOXEL_SIZE
                        voxel size
  --coors_range COORS_RANGE
                        coors range
  -d DEVICE_IDS, --device_ids DEVICE_IDS
                        device ids to run
  --shuffle_enabled SHUFFLE_ENABLED
                        shuffle enabled
  --normalize_enabled NORMALIZE_ENABLED
                        normalize enabled
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

###  point_pillar_prof.py 命令示例
```bash
# 测试最大吞吐
python3 point_pillar_prof.py \
-m "[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]" \
--elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
--max_voxel_num [16000] \
--max_points_num 12000000 \
--voxel_size [0.16,0.16,4] \
--coors_range [0,-39.68,-3,69.12,39.68,1] \
--shuffle_enabled 0 \
--normalize_enabled 0 \
--feat_size [864,496,480,480] \
--device_ids [0] \
--shape [40000] \
--batch_size 1 \
--instance 1 \
--iterations 1500 \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 point_pillar_prof.py \
-m "[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]" \
--elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
--max_voxel_num [16000] \
--max_points_num 12000000 \
--voxel_size [0.16,0.16,4] \
--coors_range [0,-39.68,-3,69.12,39.68,1] \
--shuffle_enabled 0 \
--normalize_enabled 0 \
--feat_size [864,496,480,480] \
--device_ids [0] \
--shape [40000] \
--batch_size 1 \
--instance 1 \
--iterations 1000 \
--input_host 1 \
--queue_size 0
```

###  point_pillar_prof.py 命令结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 245.84
  latency (us):
    avg latency: 12144
    min latency: 6647
    max latency: 13912
    p50 latency: 12148
    p90 latency: 12186
    p95 latency: 12197
    p99 latency: 12244

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 169.09
  latency (us):
    avg latency: 5913
    min latency: 5887
    max latency: 6179
    p50 latency: 5900
    p90 latency: 5952
    p95 latency: 5956
    p99 latency: 5985
```