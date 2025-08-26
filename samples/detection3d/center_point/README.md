# 3D Detection Sample 

本sample基于 centerpoint 算法实现 3d 目标检测


## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github]()  [modelzoo]() |
|  输入 shape |   [ () ]     |
| INT8量化方式 |   percentile         |
|  官方精度 |  "Car@0.7":86.4516/77.2855/74.6538, "Pedestrian@0.5":57.7573/52.3014/47.9166, "Cyclist@0.5":79.9918/62.6580/59.6744 |
|  VACC FP16  精度 | - |
|  VACC INT8  精度 | "Car@0.7":85.2757/73.1001/67.9215, "Pedestrian@0.5":56.8780/51.0442/47.9645, "Cyclist@0.5":78.0941/60.7013/57.5004 |


## 数据准备
下载 onnx-model_centerpoint_pp_32000_v5-32000_32000-int8-percentile-N-N-2-none 模型到 /opt/vastai/vaststreamx/data/models/
下载数据集 centerpoint 精度验证数据集 到 /opt/vastai/vaststreamx/data/datasets


## C++ Sample 

### center_point 命令行参数说明
```bash
options:
  -m, --model_prefixs            model prefixs of the model suite files (string [=[/opt/vastai/vaststreamx/data/models/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc/mod]])
      --hw_configs               hw-config file of the model suite (string [=[]])
      --elf_file                 elf file path (string [=/opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op])
  -d, --device_id                device id to run (unsigned int [=0])
      --max_points_num           max_points_num to run (unsigned int [=2000000])
      --max_voxel_num            model max voxel number (string [=[32000]])
      --voxel_size               model max voxel number (string [=[0.32,0.32,4.2]])
      --coors_range              model max voxel number (string [=[-50,-103.6,-0.1,103.6,50,4.1]])
      --input_file               input file (string [=])
      --shuffle_enabled          shuffle enabled (unsigned int [=1])
      --normalize_enabled        normalize enabled (unsigned int [=1])
      --dataset_root             dataset root (string [=])
      --dataset_output_folder    dataset output folder path (string [=])
  -?, --help                     print this message
```


### center_point 命令行示例

```bash
#测试单个文件
./vaststreamx-samples/bin/center_point \
-m "[/opt/vastai/vaststreamx/data/models/onnx-model_centerpoint_pp_32000_v5-32000_32000-int8-percentile-N-N-2-none/mod]" \
--elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
--max_voxel_num [32000] \
--voxel_size [0.32,0.32,4.2] \
--coors_range [-50,-103.6,-0.1,103.6,50,4.1] \
--max_points_num 2000000 \
--shuffle_enabled 1 \
--normalize_enabled 1 \
--input_file /opt/vastai/vaststreamx/data/datasets/kitti_val/fov_pointcloud_float16/000001.bin 


#测试数据集 
#TODO.注意 centerpoint 测试精度数据集需要参考model_zoo 因为输出label index 与pointpillar 有所不同 所以下面的数据集需要修改
#如果输入的数据集 文件 为fp32 需要指定 --from_fp32 1
mkdir -p centerpoint_out
./vaststreamx-samples/bin/center_point \
-m "[/opt/vastai/vaststreamx/data/models/onnx-model_centerpoint_pp_32000_v5-32000_32000-int8-percentile-N-N-2-none/mod]" \
--elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
--max_voxel_num [32000] \
--voxel_size [0.32,0.32,4.2] \
--coors_range [-50,-103.6,-0.1,103.6,50,4.1] \
--max_points_num 2000000 \
--shuffle_enabled 1 \
--normalize_enabled 1 \
--dataset_root /opt/vastai/vaststreamx/data/datasets/kitti_val/ \
--dataset_output_folder centerpoint_out

# 统计精度
eval_runstream.py 可参考 http://192.168.20.70/VastML/algorithm_modelzoo/-/blob/develop/detection3d/center_point/source_code/eval.py
python eval_runstream.py --dataset_yaml /home/vastai/jgxue/work/object_detection3d/OpenPCDet/tools/center_point_zte_v2/data.yaml --result_npz  centerpoint_out

# 如果需要非cuda 环境下使用精度脚本，可按照一下步骤
1. conda create -n centerpoint_precision python=3.8
2. conda activate centerpoint_precision
3. pip install numpy torch torchvision pyyaml \ 
  easydict SharedArray scipy pillow scikit-image \ 
  tqdm pyquaternion opencv-python spconv numba \ 
  -i https://pypi.tuna.tsinghua.edu.cn/simple

4. tar xvf OpenPCDet.tar.gz /work/

5. cd OpenPCDet/pcdet/datasets/kitti/kitti_object_eval_python && pip install -e .
注意 OpenPCDet.tar.gz 可通过以下路径下载:
http://cee-release.vastai.com:32482/customers/centerpoint/OpenPCDet.tar.gz
```
### center_point 命令行结果示例

```bash
# 测试单个文件结果
label: 5, score: 0.782715, box:[ 25.7969 -7.10547 0.519531 0.53125 0.574707 1.61133 -0.518066 ]
label: 5, score: 0.647461, box:[ 30.1094 24.8281 0.308594 0.633789 0.754883 1.66992 -2.83203 ]
label: 5, score: 0.577637, box:[ 15.2031 -7.53906 0.261719 0.676758 0.711914 1.59863 -1.3584 ]
label: 4, score: 0.462891, box:[ 46.25 -4.56641 0.0717773 1.67578 0.659668 1.50781 -2.9668 ]
label: 4, score: 0.383301, box:[ 30.2812 -9.28906 0.832031 1.60547 0.876465 1.56152 0.198486 ]
label: 5, score: 0.317383, box:[ 33.4688 24.1094 0.609375 0.661133 0.657715 1.65137 3.07422 ]
label: 3, score: 0.304688, box:[ 68.1875 -0.273682 0.933594 9.33594 2.82812 3.00781 0.918457 ]
label: 5, score: 0.292969, box:[ 35.3125 24 0.535156 0.9375 0.743164 1.68945 2.80469 ]
label: 5, score: 0.256836, box:[ 31.7656 -9.58594 0.785156 1.06641 0.819336 1.60547 0.0724487 ]
label: 5, score: 0.255371, box:[ 21.2188 -9.42188 0.617188 0.737305 0.67334 1.59277 -0.577637 ]
label: 5, score: 0.248047, box:[ 31.6094 25.3125 0.515625 0.621582 0.705566 1.72949 3.11523 ]
label: 5, score: 0.231445, box:[ 32.875 24.7188 0.605469 0.682129 0.623047 1.63672 2.85156 ]
label: 4, score: 0.220215, box:[ 11.2031 -9.19531 0.322266 1.73633 0.691406 1.46094 -0.682617 ]
label: 5, score: 0.214844, box:[ 40.0625 25 0.675781 0.6875 0.606934 1.58008 -2.875 ]
label: 4, score: 0.174316, box:[ 37.25 24.0625 0.882812 2.71875 1.06152 1.9209 3.00391 ]
label: 4, score: 0.172363, box:[ 76.9375 20.3438 0.125 1.35059 0.678711 1.61133 -2.82812 ]
label: 1, score: 0.168945, box:[ 64.3125 -0.794922 0.742188 4.01953 1.77051 1.89844 -1.74609 ]
label: 5, score: 0.164551, box:[ 40.7188 23.5312 0.808594 0.597656 0.585938 1.59863 -2.86133 ]
label: 5, score: 0.157715, box:[ 30.9531 25.3438 0.488281 0.606934 0.703613 1.73633 -2.9082 ]
label: 5, score: 0.156738, box:[ 17.6094 -8.50781 0.464844 0.570312 0.552246 1.66992 -0.916504 ]
label: 4, score: 0.138672, box:[ 14.9375 -8.57031 0.248047 1.5498 0.633789 1.48438 0.42334 ]
label: 5, score: 0.129395, box:[ 34.1875 24.625 0.574219 0.69043 0.685547 1.64355 2.97656 ]
label: 1, score: 0.119629, box:[ 56.5625 27.3125 0.726562 4.3125 1.77637 1.5498 -2.29688 ]
label: 4, score: 0.117188, box:[ 47.625 23.3594 1.08594 1.67578 0.818359 1.57617 0.151611 ]
label: 5, score: 0.112793, box:[ 33.9688 25.4375 0.597656 0.592773 0.624512 1.7168 2.84766 ]

# 精度统计结果
#TODO. 需要从model_zoo 获取 cneterpoint 模型精度数据集 

```

### center_point_prof 命令行参数说明

```bash
options:
  -m, --model_prefixs        model prefix of the model suite files (string [=[/opt/vastai/vaststreamx/data/models/pointpillar-int8-percentile-16000_32_10_3_16000_1_16000-vacc/mod]])
      --hw_configs           hw-config file of the model suite (string [=[]])
      --elf_file             elf file path (string [=/opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op])
      --max_voxel_num        model max voxel number (string [=[32000]])
      --max_points_num       max_points_num to run (unsigned int [=2000000])
      --voxel_size           model max voxel number (string [=[0.32,0.32,4.2]])
      --coors_range          model max voxel number (string [=[-50,-103.6,-0.1,103.6,50,4.1]])
      --shuffle_enabled      shuffle enabled (unsigned int [=1])
      --normalize_enabled    normalize enabled (unsigned int [=1])
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
### center_point_prof 命令行示例
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/center_point_prof \
-m "[/opt/vastai/vaststreamx/data/models/onnx-model_centerpoint_pp_32000_v5-32000_32000-int8-percentile-N-N-2-none/mod]" \
--elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
--max_voxel_num [32000] \
--voxel_size [0.32,0.32,4.2] \
--coors_range [-50,-103.6,-0.1,103.6,50,4.1] \
--shuffle_enabled 1 \
--normalize_enabled 1 \
--max_points_num 2000000 \
--device_ids [0] \
--shape [40000] \
--batch_size 1 \
--instance 1 \
--iterations 1500 \
--input_host 1 \
--queue_size 1

# 测试最小时延
./vaststreamx-samples/bin/center_point_prof \
-m "[/opt/vastai/vaststreamx/data/models/onnx-model_centerpoint_pp_32000_v5-32000_32000-int8-percentile-N-N-2-none/mod]" \
--elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
--max_voxel_num [32000] \
--max_points_num 2000000 \
--voxel_size [0.32,0.32,4.2] \
--coors_range [-50,-103.6,-0.1,103.6,50,4.1] \
--shuffle_enabled 1 \
--normalize_enabled 1 \
--device_ids [0] \
--shape [40000] \
--batch_size 1 \
--instance 1 \
--iterations 1000 \
--input_host 1 \
--queue_size 0
```

### center_point_prof 命令行结果示例

```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 52.1483
  latency (us):
    avg latency: 57378
    min latency: 25929
    max latency: 68563
    p50 latency: 57425
    p90 latency: 57661
    p95 latency: 57764
    p99 latency: 58523

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 40.4586
  latency (us):
    avg latency: 24715
    min latency: 23610
    max latency: 38318
    p50 latency: 24656
    p90 latency: 24928
    p95 latency: 25049
    p99 latency: 30484
```

## Python Sample

###  center_point.py 参数说明
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
  --save_npz SAVE_NPZ   save npz file
  --dataset_root DATASET_ROOT
                        dataset root
  --dataset_output_folder DATASET_OUTPUT_FOLDER
                        dataset output folder path
```

###  center_point.py 命令示例
```bash
# 测试单个输入
python3  center_point.py \
-m "[/opt/vastai/vaststreamx/data/models/onnx-model_centerpoint_pp_32000_v5-32000_32000-int8-percentile-N-N-2-none/mod]" \
--elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
--max_voxel_num [32000] \
--voxel_size [0.32,0.32,4.2] \
--coors_range [-50,-103.6,-0.1,103.6,50,4.1] \
--shuffle_enabled 1 \
--normalize_enabled 1 \
--max_points_num 2000000 \
--input_file /opt/vastai/vaststreamx/data/datasets/kitti_val/fov_pointcloud_float16/000001.bin 

# 测试数据集
#TODO.注意 centerpoint 测试精度数据集需要参考model_zoo 因为输出label index 与pointpillar 有所不同 所以下面的数据集需要修改
#如果输入的数据集 文件 为fp32 需要指定 --from_fp32 1
mkdir -p centerpoint_out
python3  center_point.py \
-m "[/opt/vastai/vaststreamx/data/models/onnx-model_centerpoint_pp_32000_v5-32000_32000-int8-percentile-N-N-2-none/mod]" \
--elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
--max_voxel_num [32000] \
--voxel_size [0.32,0.32,4.2] \
--coors_range [-50,-103.6,-0.1,103.6,50,4.1] \
--shuffle_enabled 1 \
--normalize_enabled 1 \
--max_points_num 2000000 \
--dataset_root /opt/vastai/vaststreamx/data/datasets/kitti_val/ \
--dataset_output_folder centerpoint_out

# 统计精度
eval_runstream.py 可参考 http://192.168.20.70/VastML/algorithm_modelzoo/-/blob/develop/detection3d/center_point/source_code/eval.py
python eval_runstream.py --dataset_yaml /home/vastai/jgxue/work/object_detection3d/OpenPCDet/tools/center_point_zte_v2/data.yaml --result_npz  centerpoint_out
```

```bash
# 测试单个文件结果
label: 5, score: 0.785156, box:[25.7969 -7.1055  0.5273  0.5293  0.5747  1.6182 -0.52  ]
label: 5, score: 0.644043, box:[30.1094 24.8125  0.3086  0.6313  0.7568  1.6699 -2.8398]
label: 5, score: 0.575684, box:[15.2031 -7.5391  0.2598  0.6768  0.7129  1.5986 -1.2773]
label: 4, score: 0.476562, box:[46.25   -4.5586  0.0679  1.6758  0.6641  1.5078 -2.959 ]
label: 4, score: 0.390625, box:[30.2812 -9.2891  0.832   1.6113  0.8774  1.5615  0.2302]
label: 5, score: 0.312988, box:[33.4688 24.1094  0.6016  0.6611  0.6577  1.6514  3.0723]
label: 3, score: 0.311035, box:[68.1875 -0.2676  0.9219  9.3359  2.8281  3.0078  0.7642]
label: 5, score: 0.295898, box:[35.3125 24.      0.5312  0.9312  0.7393  1.6895  2.7969]
label: 5, score: 0.261230, box:[21.2188 -9.4219  0.6211  0.7373  0.6714  1.5967 -0.5776]
label: 5, score: 0.261230, box:[31.7656 -9.5859  0.7852  1.0615  0.8193  1.6055  0.1975]
label: 5, score: 0.249023, box:[31.6094 25.3125  0.5156  0.626   0.7065  1.7295  3.1309]
label: 5, score: 0.229492, box:[32.875  24.7188  0.5977  0.6875  0.626   1.6367  2.8555]
label: 4, score: 0.221191, box:[11.2109 -9.1953  0.3203  1.7432  0.6914  1.4609 -0.7637]
label: 5, score: 0.213379, box:[40.0625 25.      0.6719  0.6904  0.6045  1.5801 -2.877 ]
label: 4, score: 0.175293, box:[37.25   24.0625  0.8867  2.7402  1.0625  1.9287  2.9961]
label: 4, score: 0.172363, box:[76.9375 20.3438  0.125   1.3506  0.6787  1.6113 -2.8301]
label: 5, score: 0.165527, box:[40.7188 23.5312  0.8047  0.6001  0.5859  1.5986 -2.8594]
label: 5, score: 0.159180, box:[30.9531 25.3438  0.4883  0.6069  0.7036  1.7363 -2.8965]
label: 5, score: 0.148926, box:[17.6094 -8.5078  0.4609  0.5723  0.5547  1.6699 -0.9058]
label: 1, score: 0.148926, box:[64.3125 -0.27    0.7383  3.7754  1.5889  1.8545 -1.6807]
label: 4, score: 0.143066, box:[14.9375 -8.5781  0.2461  1.543   0.6338  1.4844  0.4351]
label: 5, score: 0.129395, box:[34.1875 24.625   0.5742  0.6982  0.6904  1.6436  2.9707]
label: 4, score: 0.121094, box:[47.625  23.3594  1.0859  1.6758  0.8208  1.5801  0.1641]
label: 1, score: 0.119629, box:[56.5625 27.3125  0.7266  4.3125  1.7764  1.5498 -2.3008]
label: 5, score: 0.111328, box:[33.9688 25.4219  0.5977  0.6045  0.6348  1.7227  2.8438]

# 精度统计结果
#TODO 需要从model_zoo 获取centerpoint 测试精度数据集
```

###  center_point_prof.py 参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIXS, --model_prefixs MODEL_PREFIXS
                        model prefix of the model suite files
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

###  center_point_prof.py 命令示例
```bash
# 测试最大吞吐
python3 center_point_prof.py \
-m "[/opt/vastai/vaststreamx/data/models/onnx-model_centerpoint_pp_32000_v5-32000_32000-int8-percentile-N-N-2-none/mod]" \
--elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
--max_voxel_num [32000] \
--max_points_num 2000000 \
--voxel_size [0.32,0.32,4.2] \
--coors_range [-50,-103.6,-0.1,103.6,50,4.1] \
--shuffle_enabled 1 \
--normalize_enabled 1 \
--device_ids [0] \
--shape [40000] \
--batch_size 1 \
--instance 1 \
--iterations 1500 \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 center_point_prof.py \
-m "[/opt/vastai/vaststreamx/data/models/onnx-model_centerpoint_pp_32000_v5-32000_32000-int8-percentile-N-N-2-none/mod]" \
--elf_file /opt/vastai/vaststream/lib/op/ext_op/pointpillar_ext_op \
--max_voxel_num [32000] \
--max_points_num 2000000 \
--voxel_size [0.32,0.32,4.2] \
--coors_range [-50,-103.6,-0.1,103.6,50,4.1] \
--shuffle_enabled 1 \
--normalize_enabled 1 \
--device_ids [0] \
--shape [40000] \
--batch_size 1 \
--instance 1 \
--iterations 1000 \
--input_host 1 \
--queue_size 0
```

###  center_point_prof.py 命令结果示例
```bash
# 测试最大吞吐 oclk:800MHz
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 52.29
  latency (us):
    avg latency: 57225
    min latency: 24291
    max latency: 67092
    p50 latency: 57247
    p90 latency: 57504
    p95 latency: 57609
    p99 latency: 58213

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 42.96
  latency (us):
    avg latency: 23271
    min latency: 22882
    max latency: 33076
    p50 latency: 23218
    p90 latency: 23433
    p95 latency: 23534
    p99 latency: 24043
```