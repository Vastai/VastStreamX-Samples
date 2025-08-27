# multi object tracking (MOT) sample

本目录提供基于vsx  API 开发的 MOT sample


## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/ifzhang/ByteTrack)  [modelzoo](https://github.com/Vastai/VastModelZOO/tree/main/cv/mot/bytetrack) |
|  输入 shape |   [ (1,3,800,1440) ]     |
| INT8量化方式 |   percentile          |
|  官方精度 | "MOTA": 87.0,"IDF1":80.1 |
|  VACC FP16  精度 | "MOTA": 83.7,"IDF1":78.0 |
|  VACC INT8  精度 | "MOTA": 83.8,"IDF1":77.4 |


## 数据准备

下载模型 bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline 到 /opt/vastai/vaststreamx/data/models 里
下载数据集 mot17 到 /opt/vastai/vaststreamx/data/datasets 里



## C++ sample

### bytetracker 命令参数说明
```bash
options:
  -m, --model_prefix           model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod])
      --hw_config              hw-config file of the model suite (string [=])
      --vdsp_params            vdsp preprocess parameter file (string [=../data/configs/bytetrack_rgbplanar.json])
  -d, --device_id              device id to run (unsigned int [=0])
      --threshold              threshold for detection (float [=0.001])
      --label_file             label file (string [=../data/labels/coco2id.txt])
      --input_file             input file (string [=])
      --output_file            output file (string [=mot_result.jpg])
      --dataset_filelist       dataset image file list  (string [=])
      --dataset_root           dataset root (string [=])
      --dataset_result_file    dataset result file (string [=])
  -?, --help                   print this message
```

### bytetracker 命令示例

在build目录里运行 

检测单张照片的示例
```bash
./vaststreamx-samples/bin/bytetracker  \
-m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
--vdsp_params ../data/configs/bytetrack_rgbplanar.json \
--device_id 0 \
--det_threshold 0.001 \
--label_file ../data/labels/coco2id.txt \
--input_file /opt/vastai/vaststreamx/data/datasets/mot17/test/MOT17-02-FRCNN/img1/000001.jpg \
--output_file ./mot_result.jpg 
```

检测单张照片的结果
```bash
detected object class: person, score: 0.953125, id: 1, bbox: [585.712, 442.463, 85.05, 267.638]
detected object class: person, score: 0.933594, id: 2, bbox: [633.975, 457.988, 62.775, 189.675]
detected object class: person, score: 0.933594, id: 3, bbox: [1340.03, 421.2, 158.625, 367.2]
detected object class: person, score: 0.92627, id: 4, bbox: [1057.2, 479.588, 33.75, 112.725]
detected object class: person, score: 0.918457, id: 5, bbox: [440.25, 446.513, 107.325, 275.737]
detected object class: person, score: 0.918457, id: 6, bbox: [1089.6, 483.638, 32.4, 114.075]
detected object class: person, score: 0.902832, id: 7, bbox: [549.262, 460.688, 32.4, 99.225]
detected object class: person, score: 0.898926, id: 8, bbox: [1100.4, 437.738, 37.7999, 109.35]
detected object class: person, score: 0.89502, id: 9, bbox: [1253.62, 447.863, 37.8, 100.575]
detected object class: person, score: 0.892578, id: 10, bbox: [1014, 436.05, 43.2, 116.1]
detected object class: person, score: 0.887207, id: 11, bbox: [421.687, 461.025, 37.8, 79.65]
detected object class: person, score: 0.887207, id: 12, bbox: [1424.4, 419.175, 175.5, 348.975]
detected object class: person, score: 0.872559, id: 13, bbox: [480.75, 459.675, 75.6, 239.625]
detected object class: person, score: 0.871582, id: 14, bbox: [581.662, 456.638, 35.775, 126.225]
detected object class: person, score: 0.844238, id: 15, bbox: [580.312, 429.975, 19.575, 43.875]
detected object class: person, score: 0.840332, id: 16, bbox: [596.175, 429.3, 16.875, 39.825]
detected object class: person, score: 0.834473, id: 17, bbox: [934.35, 435.038, 43.2, 110.025]
detected object class: person, score: 0.765625, id: 18, bbox: [970.8, 454.613, 36.45, 87.75]
detected object class: person, score: 0.765625, id: 19, bbox: [666.037, 450.562, 29.3625, 84.375]
detected object class: person, score: 0.751953, id: 20, bbox: [1028.85, 442.8, 32.4, 90.45]
detected object class: person, score: 0.739746, id: 21, bbox: [1362.3, 433.688, 43.2, 129.6]
```
图片保存为 ./mot_result.jpg 

运行mot单个数据集的示例
在build 目录下执行
```bash
mkdir -p ./mot_output
./vaststreamx-samples/bin/bytetracker  \
-m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
--vdsp_params ../data/configs/bytetrack_rgbplanar.json \
--device_id 0 \
--det_threshold 0.01 \
--track_buffer 30 \
--track_thresh 0.6 \
--label_file ../data/labels/coco2id.txt \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/mot17/MOT17-02-FRCNN-filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/mot17/ \
--dataset_result_file ./mot_output/MOT17-02-FRCNN.txt 
```
结果保存在 ./mot_output/MOT17-02-FRCNN.txt 

测数据集精度
在build目录下执行
```bash
python3 ../evaluation/mot/mot_eval.py \
-gt /opt/vastai/vaststreamx/data/datasets/mot17/test \
-r ./mot_output
```
精度输出为
```bash
                Rcll  Prcn GT    MT    PT   ML   FP    FN  IDs   FM  MOTA  MOTP num_objects
MOT17-02-FRCNN 79.8% 92.9% 62 58.1% 35.5% 6.5% 6.1% 20.2% 0.8% 1.4% 72.9% 0.175       18581
OVERALL        79.8% 92.9% 62 58.1% 35.5% 6.5% 6.1% 20.2% 0.8% 1.4% 72.9% 0.175       18581
                IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML   FP   FN IDs   FM  MOTA  MOTP IDt IDa IDm num_objects
MOT17-02-FRCNN 58.3% 63.1% 54.2% 79.8% 92.9% 62 36 22  4 1139 3753 148  267 72.9% 0.175 103  35  12       18581
OVERALL        58.3% 63.1% 54.2% 79.8% 92.9% 62 36 22  4 1139 3753 148  267 72.9% 0.175 103  35  12       18581
```

测试整个mot17数据集精度

```bash
#在build 目录执行，命令格式为 bash  [precision_cpp.sh] [model_prefix] [device_id=0]
bash ../samples/multi_object_tracking/precision_cpp.sh \
/opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod 0

#注意，各数据集的track_buffer参数值不一样，精度结果示例
                Rcll   Prcn  GT    MT    PT   ML   FP    FN  IDs   FM  MOTA  MOTP num_objects
MOT17-02-FRCNN 79.8%  92.9%  62 58.1% 35.5% 6.5% 6.1% 20.2% 0.8% 1.4% 72.9% 0.175       18581
MOT17-04-FRCNN 90.3% 100.0%  83 71.1% 21.7% 7.2% 0.0%  9.7% 0.0% 0.9% 90.2% 0.098       47557
MOT17-05-FRCNN 84.1%  94.6% 133 57.1% 33.8% 9.0% 4.8% 15.9% 0.9% 1.5% 78.4% 0.161        6917
MOT17-09-FRCNN 87.4%  97.9%  26 88.5% 11.5% 0.0% 1.9% 12.6% 0.5% 1.2% 85.0% 0.143        5325
MOT17-10-FRCNN 81.5%  94.5%  57 66.7% 33.3% 0.0% 4.8% 18.5% 1.1% 2.0% 75.6% 0.204       12839
MOT17-11-FRCNN 90.5%  96.0%  75 69.3% 25.3% 5.3% 3.8%  9.5% 0.4% 0.6% 86.3% 0.136        9436
MOT17-13-FRCNN 88.6%  95.7% 110 74.5% 17.3% 8.2% 4.0% 11.4% 0.7% 0.8% 83.9% 0.191       11642
OVERALL        86.9%  97.0% 546 67.0% 26.6% 6.4% 2.7% 13.1% 0.5% 1.1% 83.7% 0.140      112297
                IDF1   IDP   IDR  Rcll   Prcn  GT  MT  PT ML   FP    FN IDs    FM  MOTA  MOTP IDt IDa IDm num_objects
MOT17-02-FRCNN 58.3% 63.1% 54.2% 79.8%  92.9%  62  36  22  4 1139  3753 148   267 72.9% 0.175 103  35  12       18581
MOT17-04-FRCNN 92.0% 96.9% 87.5% 90.3% 100.0%  83  59  18  6    4  4628  17   410 90.2% 0.098   8   9   0       47557
MOT17-05-FRCNN 77.9% 82.8% 73.6% 84.1%  94.6% 133  76  45 12  334  1098  63   103 78.4% 0.161  59  28  27        6917
MOT17-09-FRCNN 69.4% 73.6% 65.7% 87.4%  97.9%  26  23   3  0   99   671  29    64 85.0% 0.143  24   8   6        5325
MOT17-10-FRCNN 62.9% 67.9% 58.6% 81.5%  94.5%  57  38  19  0  614  2375 143   256 75.6% 0.204 110  44  13       12839
MOT17-11-FRCNN 83.7% 86.2% 81.3% 90.5%  96.0%  75  52  19  4  355   900  37    56 86.3% 0.136  15  23   4        9436
MOT17-13-FRCNN 71.4% 74.2% 68.8% 88.6%  95.7% 110  82  19  9  469  1324  80    92 83.9% 0.191  72  19  29       11642
OVERALL        78.4% 83.0% 74.3% 86.9%  97.0% 546 366 145 35 3014 14749 517  1248 83.7% 0.140 391 166  91      112297

```


### bytetracker_prof 命令参数说明
```bash
options:
  -m, --model_prefix    model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod])
      --hw_config       hw-config file of the model suite (string [=])
      --vdsp_params     vdsp preprocess parameter file (string [=../data/configs/bytetrack_rgbplanar.json])
  -d, --device_ids      device id to run (string [=[0]])
  -b, --batch_size      profiling batch size of the model (unsigned int [=1])
  -i, --instance        instance number or range for each device (unsigned int [=1])
  -s, --shape           model input shape (string [=])
      --iterations      iterations count for one profiling (int [=10240])
      --percentiles     percentiles of latency (string [=[50, 90, 95, 99]])
      --threshold       threshold for detection (float [=0.001])
      --input_host      cache input data into host memory (bool [=0])
  -q, --queue_size      aync wait queue size (unsigned int [=2])
  -?, --help            print this message
```

### bytetracker_prof 命令示例
在build目录里运行
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/bytetracker_prof \
-m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
--vdsp_params ../data/configs/bytetrack_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 3 \
--shape "[3,800,1440]" \
--percentiles "[50,90,95,99]" \
--iterations 800 \
--input_host 1 \
--queue_size 1


# 测试最小时延
./vaststreamx-samples/bin/bytetracker_prof \
-m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
--vdsp_params ../data/configs/bytetrack_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,800,1440]" \
--percentiles "[50,90,95,99]" \
--iterations 800 \
--input_host 1 \
--queue_size 0
```
### bytetracker_prof 结果示例

```bash
# 测试最大吞吐
- number of instances: 3
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 140.943
  latency (us):
    avg latency: 63536
    min latency: 18245
    max latency: 74130
    p50 latency: 63774
    p90 latency: 63818
    p95 latency: 63830
    p99 latency: 63858

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 102.778
  latency (us):
    avg latency: 9728
    min latency: 9673
    max latency: 10944
    p50 latency: 9712
    p90 latency: 9782
    p95 latency: 9816
    p99 latency: 10005

```

## Python sample

### bytetracker.py 命令行参数说明
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
  --detect_threshold DETECT_THRESHOLD
                        detector threshold
  --label_file LABEL_FILE
                        label file
  --track_thresh TRACK_THRESH
                        tracking confidence threshold
  --track_buffer TRACK_BUFFER
                        the frames for keep lost tracks
  --match_thresh MATCH_THRESH
                        matching threshold for tracking
  --min_box_area MIN_BOX_AREA
                        filter out tiny boxes
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file
  --dataset_filelist DATASET_FILELIST
                        dataset image filelist
  --dataset_root DATASET_ROOT
                        dataset image root
  --dataset_result_file DATASET_RESULT_FILE
                        dataset result file
```

### bytetracker.py 运行示例

在本目录下运行
```bash
python3 bytetracker.py \
-m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
--vdsp_params ../../data/configs/bytetrack_rgbplanar.json \
--device_id 0 \
--label_file ../../data/labels/coco2id.txt \
--detect_threshold 0.001 \
--track_thresh 0.6 \
--track_buffer 30 \
--match_thresh 0.9 \
--min_box_area 100 \
--input_file /opt/vastai/vaststreamx/data/datasets/mot17/test/MOT17-02-FRCNN/img1/000001.jpg \
--output_file mot_result.jpg
```

### bytetracker.py 运行结果示例

检测单张照片的结果
```bash
Object class: person, score: 0.953125, id: 1, bbox: [585.713, 442.462, 85.050, 267.638]
Object class: person, score: 0.933594, id: 2, bbox: [633.975, 457.987, 62.775, 189.675]
Object class: person, score: 0.933594, id: 3, bbox: [1340.025, 421.200, 158.625, 367.200]
Object class: person, score: 0.926270, id: 4, bbox: [1057.200, 479.587, 33.750, 112.725]
Object class: person, score: 0.918457, id: 5, bbox: [440.250, 446.513, 107.325, 275.737]
Object class: person, score: 0.918457, id: 6, bbox: [1089.600, 483.638, 32.400, 114.075]
Object class: person, score: 0.902832, id: 7, bbox: [549.263, 460.688, 32.400, 99.225]
Object class: person, score: 0.898926, id: 8, bbox: [1100.400, 437.737, 37.800, 109.350]
Object class: person, score: 0.895020, id: 9, bbox: [1253.625, 447.862, 37.800, 100.575]
Object class: person, score: 0.892578, id: 10, bbox: [1014.000, 436.050, 43.200, 116.100]
Object class: person, score: 0.887207, id: 11, bbox: [421.688, 461.025, 37.800, 79.650]
Object class: person, score: 0.887207, id: 12, bbox: [1424.400, 419.175, 175.500, 348.975]
Object class: person, score: 0.872559, id: 13, bbox: [480.750, 459.675, 75.600, 239.625]
Object class: person, score: 0.871582, id: 14, bbox: [581.662, 456.638, 35.775, 126.225]
Object class: person, score: 0.844238, id: 15, bbox: [580.312, 429.975, 19.575, 43.875]
Object class: person, score: 0.840332, id: 16, bbox: [596.175, 429.300, 16.875, 39.825]
Object class: person, score: 0.834473, id: 17, bbox: [934.350, 435.038, 43.200, 110.025]
Object class: person, score: 0.765625, id: 18, bbox: [970.800, 454.612, 36.450, 87.750]
Object class: person, score: 0.765625, id: 19, bbox: [666.037, 450.562, 29.362, 84.375]
Object class: person, score: 0.751953, id: 20, bbox: [1028.850, 442.800, 32.400, 90.450]
Object class: person, score: 0.739746, id: 21, bbox: [1362.300, 433.688, 43.200, 129.600]
```
如果指定了输出文件，则可以在输出文件中看到检测框

运行数据集的示例
在当前目录下执行
```bash
mkdir -p mot_output
python3 bytetracker.py \
-m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
--vdsp_params ../../data/configs/bytetrack_rgbplanar.json \
--device_id 0 \
--label_file ../../data/labels/coco2id.txt \
--detect_threshold 0.001 \
--track_thresh 0.6 \
--track_buffer 30 \
--match_thresh 0.9 \
--min_box_area 100 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/mot17/MOT17-02-FRCNN-filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/mot17/ \
--dataset_result_file ./mot_output/MOT17-02-FRCNN.txt 
```



结果保存在 ./mot_output/MOT17-02-FRCNN.txt 

测数据集精度
在build目录下执行
```bash
python3 ../../evaluation/mot/mot_eval.py \
-gt /opt/vastai/vaststreamx/data/datasets/mot17/test \
-r ./mot_output
```
精度输出为
```bash
                Rcll  Prcn GT    MT    PT   ML   FP    FN  IDs   FM  MOTA  MOTP num_objects
MOT17-02-FRCNN 79.8% 92.9% 62 58.1% 35.5% 6.5% 6.1% 20.2% 0.8% 1.4% 72.9% 0.175       18581
OVERALL        79.8% 92.9% 62 58.1% 35.5% 6.5% 6.1% 20.2% 0.8% 1.4% 72.9% 0.175       18581
                IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML   FP   FN IDs   FM  MOTA  MOTP IDt IDa IDm num_objects
MOT17-02-FRCNN 58.3% 63.1% 54.2% 79.8% 92.9% 62 36 22  4 1139 3753 148  265 72.9% 0.175 103  35  12       18581
OVERALL        58.3% 63.1% 54.2% 79.8% 92.9% 62 36 22  4 1139 3753 148  265 72.9% 0.175 103  35  12       18581
```



测试整个mot17数据集精度

```bash
#在当前 目录执行，命令格式为 bash  [precision_py.sh] [model_prefix] [device_id=0]
bash precision_py.sh \
/opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod 0

#注意，各数据集的track_buffer参数值不一样，精度结果示例
                Rcll   Prcn  GT    MT    PT   ML   FP    FN  IDs   FM  MOTA  MOTP num_objects
MOT17-02-FRCNN 79.8%  92.9%  62 58.1% 35.5% 6.5% 6.1% 20.2% 0.8% 1.4% 72.9% 0.175       18581
MOT17-04-FRCNN 90.3% 100.0%  83 71.1% 21.7% 7.2% 0.0%  9.7% 0.0% 0.9% 90.2% 0.098       47557
MOT17-05-FRCNN 84.1%  94.6% 133 57.1% 33.8% 9.0% 4.8% 15.9% 0.9% 1.5% 78.4% 0.161        6917
MOT17-09-FRCNN 87.4%  97.9%  26 88.5% 11.5% 0.0% 1.9% 12.6% 0.5% 1.2% 85.0% 0.143        5325
MOT17-10-FRCNN 81.5%  94.5%  57 66.7% 33.3% 0.0% 4.8% 18.5% 1.1% 2.0% 75.6% 0.204       12839
MOT17-11-FRCNN 90.5%  96.0%  75 69.3% 25.3% 5.3% 3.8%  9.5% 0.4% 0.6% 86.3% 0.136        9436
MOT17-13-FRCNN 88.6%  95.6% 110 74.5% 17.3% 8.2% 4.0% 11.4% 0.7% 0.8% 83.9% 0.191       11642
OVERALL        86.9%  97.0% 546 67.0% 26.6% 6.4% 2.7% 13.1% 0.5% 1.1% 83.7% 0.140      112297
                IDF1   IDP   IDR  Rcll   Prcn  GT  MT  PT ML   FP    FN IDs    FM  MOTA  MOTP IDt IDa IDm num_objects
MOT17-02-FRCNN 58.3% 63.1% 54.2% 79.8%  92.9%  62  36  22  4 1139  3753 148   265 72.9% 0.175 103  35  12       18581
MOT17-04-FRCNN 92.0% 96.9% 87.5% 90.3% 100.0%  83  59  18  6    4  4628  17   410 90.2% 0.098   8   9   0       47557
MOT17-05-FRCNN 77.9% 82.8% 73.6% 84.1%  94.6% 133  76  45 12  334  1098  63   103 78.4% 0.161  59  28  27        6917
MOT17-09-FRCNN 69.4% 73.6% 65.7% 87.4%  97.9%  26  23   3  0   99   671  29    64 85.0% 0.143  24   8   6        5325
MOT17-10-FRCNN 62.9% 67.9% 58.6% 81.5%  94.5%  57  38  19  0  613  2374 143   255 75.6% 0.204 110  44  13       12839
MOT17-11-FRCNN 83.7% 86.2% 81.2% 90.5%  96.0%  75  52  19  4  356   901  37    56 86.3% 0.136  15  23   4        9436
MOT17-13-FRCNN 71.4% 74.2% 68.8% 88.6%  95.6% 110  82  19  9  471  1326  80    94 83.9% 0.191  72  19  29       11642
OVERALL        78.4% 83.0% 74.3% 86.9%  97.0% 546 366 145 35 3016 14751 517  1247 83.7% 0.140 391 166  91      112297
```


### bytetracker_prof.py 命令行参数说明

bytetracker_prof.py只测试检测模型，不带跟踪算法
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
### bytetracker_prof.py 命令行示例

```bash
# 测试最大吞吐
python3 bytetracker_prof.py \
-m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
--vdsp_params ../../data/configs/bytetrack_rgbplanar.json \
--device_ids [0] \
--batch_size 1 \
--instance 1 \
--shape "[3,800,1440]" \
--percentiles "[50,90,95,99]" \
--iterations 800 \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 bytetracker_prof.py \
-m /opt/vastai/vaststreamx/data/models/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline/mod \
--vdsp_params ../../data/configs/bytetrack_rgbplanar.json \
--device_ids [0]  \
--batch_size 1 \
--instance 1 \
--shape "[3,800,1440]" \
--percentiles "[50,90,95,99]" \
--iterations 800 \
--input_host 1 \
--queue_size 0
```

### bytetracker_prof.py 命令行结果示例
```bash

# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 140.65
  latency (us):
    avg latency: 21247
    min latency: 14636
    max latency: 28906
    p50 latency: 21248
    p90 latency: 21296
    p95 latency: 21315
    p99 latency: 21367

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 103.52
  latency (us):
    avg latency: 9658
    min latency: 9624
    max latency: 10810
    p50 latency: 9641
    p90 latency: 9704
    p95 latency: 9728
    p99 latency: 9886
```