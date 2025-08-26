# Buildin Op Profile

Buildin Op 即 SDK 自带的算子。 本 sample 用于展示如何测试各算子的性能   
可以更改json文件里的参数，实现测试不同输入场景下的性能测试 

注：本sample不支持 BERT_EMBEDDING_OP 的性能测试

## C++ Sample 


### buildin_op_prof 命令参数说明

```bash
options:
  -d, --device_ids     device id to run (string [=[0]])
      --op_config      build in op config json (string)
  -i, --instance       instance number or range for each device (unsigned int [=1])
      --iterations     iterations count for one profiling (int [=10240])
      --percentiles    percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host     cache input data into host memory (bool [=0])
  -?, --help           print this message
```


### SINGLE_OP_RESIZE

```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/resize_op.json \
--device_ids [0] \
--instance 6 \
--iterations 25000 \
--percentiles "[50,90,95,99]" \
--input_host 1 
```
结果示例
```bash
- number of instances: 6
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 6061.39
  latency (us):
    avg latency: 989
    min latency: 821
    max latency: 1807
    p50 latency: 1003
    p90 latency: 1093
    p95 latency: 1104
    p99 latency: 1126
```


### SINGLE_OP_CROP

```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/crop_op.json \
--device_ids [0] \
--instance 12 \
--iterations 150000 \
--percentiles "[50,90,95,99]" \
--input_host 0 
```
结果示例
```bash
- number of instances: 12
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 50529.1
  latency (us):
    avg latency: 236
    min latency: 103
    max latency: 896
    p50 latency: 222
    p90 latency: 334
    p95 latency: 361
    p99 latency: 409
```

### SINGLE_OP_CVT_COLOR

```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/cvtcolor_op.json \
--device_ids [0] \
--instance 8 \
--iterations 100000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 8
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 14838.1
  latency (us):
    avg latency: 538
    min latency: 392
    max latency: 1144
    p50 latency: 482
    p90 latency: 748
    p95 latency: 760
    p99 latency: 794
```



### SINGLE_OP_BATCH_CROP_RESIZE

```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/batch_crop_resize_op.json \
--device_ids [0] \
--instance 5 \
--iterations 10000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 5
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 2291.06
  latency (us):
    avg latency: 2181
    min latency: 1886
    max latency: 3971
    p50 latency: 2021
    p90 latency: 2693
    p95 latency: 2714
    p99 latency: 2752
```



### SINGLE_OP_WARP_AFFINE

```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/warpaffine_op.json \
--device_ids [0] \
--instance 5 \
--iterations 5000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 5
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 1048.27
  latency (us):
    avg latency: 4769
    min latency: 3935
    max latency: 7964
    p50 latency: 4109
    p90 latency: 6768
    p95 latency: 6969
    p99 latency: 7008
```


### SINGLE_OP_FLIP
```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/flip_op.json \
--device_ids [0] \
--instance 6 \
--iterations 20000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 6
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 4031.39
  latency (us):
    avg latency: 1487
    min latency: 1053
    max latency: 2660
    p50 latency: 1144
    p90 latency: 2570
    p95 latency: 2589
    p99 latency: 2615
```


### SINGLE_OP_SCALE
```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/scale_op.json \
--device_ids [0] \
--instance 5 \
--iterations 15000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 5
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 3873.49
  latency (us):
    avg latency: 1290
    min latency: 1143
    max latency: 2334
    p50 latency: 1243
    p90 latency: 1476
    p95 latency: 1494
    p99 latency: 1520
```
### SINGLE_OP_COPY_MAKE_BORDER
```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/copy_make_boarder_op.json \
--device_ids [0] \
--instance 6 \
--iterations 60000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 6
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 11733.5
  latency (us):
    avg latency: 510
    min latency: 403
    max latency: 842
    p50 latency: 515
    p90 latency: 564
    p95 latency: 578
    p99 latency: 603
```

### FUSION_OP_YUV_NV12_RESIZE_2RGB_NORM_TENSOR
```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/nv12_resize_2rgb.json \
--device_ids [0] \
--instance 5 \
--iterations 15000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 5
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 3062.17
  latency (us):
    avg latency: 1632
    min latency: 1352
    max latency: 2744
    p50 latency: 1416
    p90 latency: 2302
    p95 latency: 2335
    p99 latency: 2370
```

### FUSION_OP_YUV_NV12_CVTCOLOR_RESIZE_NORM_TENSOR
```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/nv12_cvtcolor_resize.json \
--device_ids [0] \
--instance 5 \
--iterations 35000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 5
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 7259.6
  latency (us):
    avg latency: 687
    min latency: 609
    max latency: 1417
    p50 latency: 670
    p90 latency: 771
    p95 latency: 791
    p99 latency: 812
```

### FUSION_OP_YUV_NV12_RESIZE_CVTCOLOR_CROP_NORM_TENSOR
```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/nv12_resize_cvtcolor_crop.json \
--device_ids [0] \
--instance 5 \
--iterations 10000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 5
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 1636.16
  latency (us):
    avg latency: 3055
    min latency: 2518
    max latency: 5193
    p50 latency: 2682
    p90 latency: 4261
    p95 latency: 4303
    p99 latency: 4364
```

### FUSION_OP_YUV_NV12_CROP_CVTCOLOR_RESIZE_NORM_TENSOR
```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/nv12_crop_cvtcolor_resize.json \
--device_ids [0] \
--instance 6 \
--iterations 30000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 6
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 5245.67
  latency (us):
    avg latency: 1143
    min latency: 904
    max latency: 1926
    p50 latency: 989
    p90 latency: 1636
    p95 latency: 1653
    p99 latency: 1681
```

### FUSION_OP_YUV_NV12_CVTCOLOR_RESIZE_CROP_NORM_TENSOR
```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/nv12_cvtcolor_resize_crop.json \
--device_ids [0] \
--instance 5 \
--iterations 30000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 6
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 4968.63
  latency (us):
    avg latency: 1206
    min latency: 955
    max latency: 2006
    p50 latency: 1042
    p90 latency: 1737
    p95 latency: 1760
    p99 latency: 1790
```

### FUSION_OP_YUV_NV12_CVTCOLOR_LETTERBOX_NORM_TENSOR
```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/nv12_cvtcolor_letterbox.json \
--device_ids [0] \
--instance 5 \
--iterations 30000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 5
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 7447.92
  latency (us):
    avg latency: 670
    min latency: 601
    max latency: 1222
    p50 latency: 654
    p90 latency: 745
    p95 latency: 760
    p99 latency: 788
```


### FUSION_OP_YUV_NV12_LETTERBOX_2RGB_NORM_TENSOR
```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/nv12_letterbox_2rgb.json \
--device_ids [0] \
--instance 5 \
--iterations 10000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 5
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 2263.86
  latency (us):
    avg latency: 2207
    min latency: 1823
    max latency: 3757
    p50 latency: 1923
    p90 latency: 3094
    p95 latency: 3118
    p99 latency: 3152
```

### FUSION_OP_RGB_CVTCOLOR_NORM_TENSOR
```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/rgb_cvtcolor.json \
--device_ids [0] \
--instance 8 \
--iterations 100000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 8
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 21095.4
  latency (us):
    avg latency: 378
    min latency: 293
    max latency: 985
    p50 latency: 377
    p90 latency: 413
    p95 latency: 427
    p99 latency: 459
```

### FUSION_OP_RGB_RESIZE_CVTCOLOR_NORM_TENSOR
```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/rgb_resize_cvtcolor.json \
--device_ids [0] \
--instance 6 \
--iterations 30000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 6
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 7396.65
  latency (us):
    avg latency: 810
    min latency: 617
    max latency: 1297
    p50 latency: 693
    p90 latency: 1201
    p95 latency: 1214
    p99 latency: 1236
```

### FUSION_OP_RGB_RESIZE_CVTCOLOR_CROP_NORM_TENSOR
```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/rgb_resize_cvtcolor_crop.json \
--device_ids [0] \
--instance 6 \
--iterations 30000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 6
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 5122.42
  latency (us):
    avg latency: 1170
    min latency: 1027
    max latency: 2357
    p50 latency: 1168
    p90 latency: 1214
    p95 latency: 1228
    p99 latency: 1255
```


### FUSION_OP_RGB_CROP_RESIZE_CVTCOLOR_NORM_TENSOR
```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/rgb_crop_resize_cvtcolor.json \
--device_ids [0] \
--instance 5 \
--iterations 30000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 6
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 5155.88
  latency (us):
    avg latency: 1162
    min latency: 1041
    max latency: 2245
    p50 latency: 1162
    p90 latency: 1197
    p95 latency: 1209
    p99 latency: 1234
```

### FUSION_OP_RGB_LETTERBOX_CVTCOLOR_NORM_TENSOR
```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/rgb_letterbox_cvtcolor.json \
--device_ids [0] \
--instance 6 \
--iterations 12000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 6
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 2228.93
  latency (us):
    avg latency: 2691
    min latency: 2216
    max latency: 4821
    p50 latency: 2724
    p90 latency: 2995
    p95 latency: 3031
    p99 latency: 3099
```

### FUSION_OP_RGB_LETTERBOX_CVTCOLOR_NORM_TENSOR_EXT
```bash
./vaststreamx-samples/bin/buildin_op_prof \
--op_config ../samples/vdsp_op/buildin_op_prof/rgb_letterbox_cvtcolor_ext.json \
--device_ids [0] \
--instance 5 \
--iterations 8000 \
--percentiles "[50,90,95,99]" \
--input_host 1
```
结果示例
```bash
- number of instances: 5
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 1532.75
  latency (us):
    avg latency: 3261
    min latency: 2945
    max latency: 6201
    p50 latency: 3239
    p90 latency: 3446
    p95 latency: 3534
    p99 latency: 3675
```










## Python Sample

### buildin_op_prof.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  --op_config OP_CONFIG
                        op config file
  -d DEVICE_IDS, --device_ids DEVICE_IDS
                        device ids to run
  -i INSTANCE, --instance INSTANCE
                        instance number for each device
  --iterations ITERATIONS
                        iterations count for one profiling
  --percentiles PERCENTILES
                        percentiles of latency
  --input_host INPUT_HOST
                        cache input data into host memory
```

### SINGLE_OP_RESIZE
```bash
python3 buildin_op_prof.py \
--op_config resize_op.json \
--device_ids [0] \
--instance 6 \
--iterations 50000 \
--percentiles "[50,90,95,99]" \
--input_host 0 
```
结果示例
```bash
- number of instances: 6
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 6535.005363103857
  latency (us):
    avg latency: 917
    min latency: 520
    max latency: 3065
    p50 latency: 829
    p90 latency: 1132
    p95 latency: 1143
    p99 latency: 1163
```

### SINGLE_OP_CROP
```bash
python3 buildin_op_prof.py \
--op_config crop_op.json \
--device_ids [0] \
--instance 12 \
--iterations 150000 \
--percentiles "[50,90,95,99]" \
--input_host 0 
```
结果示例
```bash
- number of instances: 12
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 51661.45001275277
  latency (us):
    avg latency: 231
    min latency: 88
    max latency: 8139
    p50 latency: 189
    p90 latency: 407
    p95 latency: 443
    p99 latency: 488
```

### SINGLE_OP_CVT_COLOR

```bash
python3 buildin_op_prof.py \
--op_config cvtcolor_op.json \
--device_ids [0] \
--instance 8 \
--iterations 100000 \
--percentiles "[50,90,95,99]" \
--input_host 0
```
结果示例
```bash
- number of instances: 8
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 15303.986405609608
  latency (us):
    avg latency: 521
    min latency: 230
    max latency: 6850
    p50 latency: 362
    p90 latency: 843
    p95 latency: 856
    p99 latency: 872
```
### SINGLE_OP_BATCH_CROP_RESIZE
```bash
python3 buildin_op_prof.py \
--op_config batch_crop_resize_op.json \
--device_ids [0] \
--instance 5 \
--iterations 10000 \
--percentiles "[50,90,95,99]" \
--input_host 1 
```
结果示例
```bash
- number of instances: 5
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 2282.6354484680005
  latency (us):
    avg latency: 2189
    min latency: 1878
    max latency: 2778
    p50 latency: 2059
    p90 latency: 2631
    p95 latency: 2663
    p99 latency: 2705
```
### SINGLE_OP_WARP_AFFINE
```bash
python3 buildin_op_prof.py \
--op_config warpaffine_op.json \
--device_ids [0] \
--instance 5 \
--iterations 5000 \
--percentiles "[50,90,95,99]" \
--input_host 1 
```
结果示例
```bash
- number of instances: 5
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 1051.4850792976165
  latency (us):
    avg latency: 4754
    min latency: 3941
    max latency: 5594
    p50 latency: 4816
    p90 latency: 5506
    p95 latency: 5562
    p99 latency: 5580
```

### SINGLE_OP_FLIP
```bash
python3 buildin_op_prof.py \
--op_config flip_op.json \
--device_ids [0] \
--instance 6 \
--iterations 20000 \
--percentiles "[50,90,95,99]" \
--input_host 1 
```
结果示例
```bash
- number of instances: 6
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 4038.0185257460507
  latency (us):
    avg latency: 1484
    min latency: 967
    max latency: 2522
    p50 latency: 1178
    p90 latency: 2449
    p95 latency: 2464
    p99 latency: 2485
```
### SINGLE_OP_SCALE
```bash
python3 buildin_op_prof.py \
--op_config scale_op.json \
--device_ids [0] \
--instance 5 \
--iterations 15000 \
--percentiles "[50,90,95,99]" \
--input_host 1 
```
结果示例
```bash
- number of instances: 5
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 3861.2648859534556
  latency (us):
    avg latency: 1293
    min latency: 1096
    max latency: 1628
    p50 latency: 1249
    p90 latency: 1455
    p95 latency: 1466
    p99 latency: 1486
```


### SINGLE_OP_COPY_MAKE_BORDER
```bash
python3 buildin_op_prof.py \
--op_config copy_make_boarder_op.json \
--device_ids [0] \
--instance 5 \
--iterations 60000 \
--percentiles "[50,90,95,99]" \
--input_host 0 
```
结果示例
```bash
- number of instances: 5
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 12097.874208821248
  latency (us):
    avg latency: 412
    min latency: 331
    max latency: 2978
    p50 latency: 395
    p90 latency: 473
    p95 latency: 492
    p99 latency: 507
```

### FUSION_OP_YUV_NV12_RESIZE_2RGB_NORM_TENSOR
```bash
python3 buildin_op_prof.py \
--op_config nv12_resize_2rgb.json \
--device_ids [0] \
--instance 5 \
--iterations 15000 \
--percentiles "[50,90,95,99]" \
--input_host 1 
```
结果示例
```bash
- number of instances: 5
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 3062.869470765104
  latency (us):
    avg latency: 1631
    min latency: 1222
    max latency: 2307
    p50 latency: 1442
    p90 latency: 2221
    p95 latency: 2238
    p99 latency: 2265
```


### FUSION_OP_YUV_NV12_CVTCOLOR_RESIZE_NORM_TENSOR
```bash
python3 buildin_op_prof.py \
--op_config nv12_cvtcolor_resize.json \
--device_ids [0] \
--instance 5 \
--iterations 35000 \
--percentiles "[50,90,95,99]" \
--input_host 1 
```
结果示例
```bash
- number of instances: 5
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 7184.065961151978
  latency (us):
    avg latency: 694
    min latency: 573
    max latency: 1702
    p50 latency: 691
    p90 latency: 725
    p95 latency: 738
    p99 latency: 760
```

### FUSION_OP_YUV_NV12_RESIZE_CVTCOLOR_CROP_NORM_TENSOR
```bash
python3 buildin_op_prof.py \
--op_config nv12_resize_cvtcolor_crop.json \
--device_ids [0] \
--instance 5 \
--iterations 10000 \
--percentiles "[50,90,95,99]" \
--input_host 1 
```
结果示例
```bash
- number of instances: 5
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 1640.2346316691578
  latency (us):
    avg latency: 3047
    min latency: 2139
    max latency: 4223
    p50 latency: 2728
    p90 latency: 4086
    p95 latency: 4117
    p99 latency: 4161
```

### FUSION_OP_YUV_NV12_CROP_CVTCOLOR_RESIZE_NORM_TENSOR
```bash
python3 buildin_op_prof.py \
--op_config nv12_crop_cvtcolor_resize.json \
--device_ids [0] \
--instance 6 \
--iterations 30000 \
--percentiles "[50,90,95,99]" \
--input_host 0 
```
结果示例
```bash
- number of instances: 6
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 5338.250377215747
  latency (us):
    avg latency: 1122
    min latency: 717
    max latency: 2399
    p50 latency: 874
    p90 latency: 1508
    p95 latency: 1527
    p99 latency: 1569
```

### FUSION_OP_YUV_NV12_CVTCOLOR_RESIZE_CROP_NORM_TENSOR
```bash
python3 buildin_op_prof.py \
--op_config nv12_cvtcolor_resize_crop.json \
--device_ids [0] \
--instance 5 \
--iterations 30000 \
--percentiles "[50,90,95,99]" \
--input_host 0
```
结果示例
```bash
- number of instances: 5
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 4965.967704317237
  latency (us):
    avg latency: 1005
    min latency: 770
    max latency: 1916
    p50 latency: 865
    p90 latency: 1459
    p95 latency: 1474
    p99 latency: 1495
```

### FUSION_OP_YUV_NV12_CVTCOLOR_LETTERBOX_NORM_TENSOR
```bash
python3 buildin_op_prof.py \
--op_config  nv12_cvtcolor_letterbox.json \
--device_ids [0] \
--instance 5 \
--iterations 30000 \
--percentiles "[50,90,95,99]" \
--input_host 1 
```
结果示例
```bash
- number of instances: 5
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 7368.262845336983
  latency (us):
    avg latency: 677
    min latency: 575
    max latency: 1552
    p50 latency: 675
    p90 latency: 704
    p95 latency: 716
    p99 latency: 739
```

### FUSION_OP_YUV_NV12_LETTERBOX_2RGB_NORM_TENSOR
```bash
python3 buildin_op_prof.py \
--op_config nv12_letterbox_2rgb.json \
--device_ids [0] \
--instance 5 \
--iterations 10000 \
--percentiles "[50,90,95,99]" \
--input_host 1 
```
结果示例
```bash
- number of instances: 5
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 2264.827890426464
  latency (us):
    avg latency: 2206
    min latency: 1595
    max latency: 3207
    p50 latency: 1916
    p90 latency: 3115
    p95 latency: 3136
    p99 latency: 3163
```

### FUSION_OP_RGB_CVTCOLOR_NORM_TENSOR
```bash
python3 buildin_op_prof.py \
--op_config rgb_cvtcolor.json \
--device_ids [0] \
--instance 8 \
--iterations 100000 \
--percentiles "[50,90,95,99]" \
--input_host 0 
```
结果示例
```bash
- number of instances: 8
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 21443.826304877006
  latency (us):
    avg latency: 371
    min latency: 189
    max latency: 8196
    p50 latency: 264
    p90 latency: 720
    p95 latency: 732
    p99 latency: 755
```


### FUSION_OP_RGB_RESIZE_CVTCOLOR_NORM_TENSOR

```bash
python3 buildin_op_prof.py \
--op_config rgb_resize_cvtcolor.json \
--device_ids [0] \
--instance 6 \
--iterations 30000 \
--percentiles "[50,90,95,99]" \
--input_host 1 
```
结果示例
```bash
- number of instances: 6
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 7525.072314710935
  latency (us):
    avg latency: 796
    min latency: 561
    max latency: 2017
    p50 latency: 780
    p90 latency: 949
    p95 latency: 994
    p99 latency: 1089
```
### FUSION_OP_RGB_RESIZE_CVTCOLOR_CROP_NORM_TENSOR
```bash
python3 buildin_op_prof.py \
--op_config rgb_resize_cvtcolor_crop.json \
--device_ids [0] \
--instance 6 \
--iterations 30000 \
--percentiles "[50,90,95,99]" \
--input_host 0 
```
结果示例
```bash
- number of instances: 6
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 5262.4180383612
  latency (us):
    avg latency: 1138
    min latency: 703
    max latency: 2522
    p50 latency: 1209
    p90 latency: 1347
    p95 latency: 1362
    p99 latency: 1417
```

### FUSION_OP_RGB_CROP_RESIZE_CVTCOLOR_NORM_TENSOR
```bash
python3 buildin_op_prof.py \
--op_config rgb_crop_resize_cvtcolor.json \
--device_ids [0] \
--instance 5 \
--iterations 30000 \
--percentiles "[50,90,95,99]" \
--input_host 0 
```
结果示例
```bash
- number of instances: 5
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 5390.043493414126
  latency (us):
    avg latency: 926
    min latency: 702
    max latency: 1434
    p50 latency: 801
    p90 latency: 1346
    p95 latency: 1357
    p99 latency: 1374
```

### FUSION_OP_RGB_LETTERBOX_CVTCOLOR_NORM_TENSOR
```bash
python3 buildin_op_prof.py \
--op_config rgb_letterbox_cvtcolor.json \
--device_ids [0] \
--instance 6 \
--iterations 12000 \
--percentiles "[50,90,95,99]" \
--input_host 0 
```
结果示例
```bash
- number of instances: 6
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 2423.1987977548465
  latency (us):
    avg latency: 2473
    min latency: 1232
    max latency: 3159
    p50 latency: 2306
    p90 latency: 3042
    p95 latency: 3061
    p99 latency: 3094
```

### FUSION_OP_RGB_LETTERBOX_CVTCOLOR_NORM_TENSOR_EXT

```bash
python3 buildin_op_prof.py \
--op_config rgb_letterbox_cvtcolor_ext.json \
--device_ids [0] \
--instance 5 \
--iterations 8000 \
--percentiles "[50,90,95,99]" \
--input_host 0 
```
结果示例
```bash
- number of instances: 5
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 1574.082948355669
  latency (us):
    avg latency: 3174
    min latency: 2171
    max latency: 5228
    p50 latency: 2680
    p90 latency: 4894
    p95 latency: 4910
    p99 latency: 4944
```




























