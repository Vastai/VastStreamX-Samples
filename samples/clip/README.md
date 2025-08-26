# CLIP Sample 

本 sample 提供 clip 算法的基本用法，clip 算法主要功能是，提供一张图片和多个字符串，计算各字符串与图片的匹配程度，分数越高，对应的字符串与图片匹配越高。 clip 算法 有两个模型，一个是 image 模型，一个 text 模型。     

C++ 与 Python Sample 均需要安装 python clip 包，安装方法 pip3 install  git+https://github.com/openai/CLIP.git


## 模型信息
|    模型信息   |  值       |
|-----------|-----------|
|    来源   | [github](https://github.com/openai/CLIP.git)  [modelzoo](http://gitlabdev.vastai.com/VastML/algorithm_modelzoo/-/tree/develop/classification/vision_transformer) |
|  输入 shape | image: [ (1,3,224,224) ]   text: [(1,77)]  |
| INT8量化方式 |   -          |
|  官方精度 |  -      |
|  VACC FP16  精度 | top1_rate: 55.62 top5_rate: 82.602 |
|  VACC INT8  精度 | -  |

## 数据准备
下载模型 clip_image-fp16-none-1_3_224_224-vacc  到 /opt/vastai/vaststreamx/data/models 里  
下载模型 clip_text-fp16-none-1_77-vacc  到 /opt/vastai/vaststreamx/data/models 里  
下载数据集 ILSVRC2012_img_val 到 /opt/vastai/vaststreamx/data/datasets 里  


## C++ Sample

### clip_sample 命令行参数说明
```bash
options:
      --imgmod_prefix          image model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod])
      --imgmod_hw_config       hw-config file of the model suite (string [=])
      --norm_elf               normalize op elf file (string [=/opt/vastai/vaststreamx/data/elf/normalize])
      --space2depth_elf        space_to_depth op elf file (string [=/opt/vastai/vaststreamx/data/elf/space_to_depth])
      --txtmod_prefix          model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod])
      --txtmod_hw_config       hw-config file of the model suite (string [=])
      --txtmod_vdsp_params     vdsp preprocess parameter file (string [=../data/configs/clip_txt_vdsp.json])
  -d, --device_id              device id to run (unsigned int [=0])
      --label_file             npz filelist of input strings (string [=])
      --npz_files_path         npz filelist of input strings (string [=])
      --input_file             input file (string [=../data/images/CLIP.png])
      --dataset_filelist       input dataset filelist (string [=])
      --dataset_root           input dataset root (string [=])
      --dataset_output_file    dataset output file (string [=])
  -?, --help                   print this message
```

### clip_sample 命令行示例    
在build目录里执行  
```bash
# 测试 ../data/images/CLIP.png 与 "a diagram", "a dog", "a cat" 三个字符串的匹配分数
# 1. 先将  "a diagram", "a dog", "a cat" 转成 clip_text 模型所需的 token tensor。
# 1.1 将  "a diagram", "a dog", "a cat" 写入一个 txt，每个字符串一行，参考 ../samples/clip/test_label.txt 
# 1.2 运行脚本 ../samples/clip/make_input_npz.py 得到每个字符串对应的npz文件，并保存于 npz_files_path

python3 ../samples/clip/make_input_npz.py \
--label_file ../samples/clip/test_label.txt \
--npz_files_path npz_files

# 2. 执行 clip_sample 命令，计算三个字符与CLIP.png的匹配分数
./vaststreamx-samples/bin/clip_sample \
--imgmod_prefix /opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod \
--norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
--space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
--txtmod_prefix /opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod \
--txtmod_vdsp_params ../data/configs/clip_txt_vdsp.json \
--device_id 0 \
--label_file ../samples/clip/test_label.txt \
--input_file ../data/images/CLIP.png \
--npz_files_path npz_files

#输出如下结果
Top5:
0th, string: a diagram, score: 0.99674
1th, string: a dog, score: 0.00232988
2th, string: a cat, score: 0.000929735
# 说明 "a diagram"与CLIP.png最匹配


# 用 imagenet 数据集测试精度

# 1.对 imagenet 数据集的标签字符串，生成对应的 npz 文件
python3 ../samples/clip/make_input_npz.py \
--label_file ../data/labels/imagenet.txt \
--npz_files_path imagenet_label_npz_files

# 2. 执行 clip_sample 命令，生成分类结果 clip_result.txt
./vaststreamx-samples/bin/clip_sample \
--imgmod_prefix /opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod \
--norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
--space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
--txtmod_prefix /opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod \
--txtmod_vdsp_params ../data/configs/clip_txt_vdsp.json \
--device_id 0 \
--label_file ../data/labels/imagenet.txt \
--npz_files_path imagenet_label_npz_files \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ILSVRC2012_img_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file clip_result.txt

# 3. 统计精度
python3 ../evaluation/classification/eval_topk.py  clip_result.txt  


# 4. 精度统计结果
[VACC]:  top1_rate: 55.62 top5_rate: 82.602
```

### clip_image 模型性能测试

clip_image_prof 命令行参数说明
```bash
options:
  -m, --model_prefix       model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod])
      --hw_config          hw-config file of the model suite (string [=])
      --norm_elf           normalize op elf file (string [=/opt/vastai/vaststreamx/data/elf/normalize])
      --space2depth_elf    space_to_depth op elf file (string [=/opt/vastai/vaststreamx/data/elf/space_to_depth])
  -d, --device_ids         device id to run (string [=[0]])
  -b, --batch_size         profiling batch size of the model (unsigned int [=1])
  -i, --instance           instance number for each device (unsigned int [=1])
  -s, --shape              model input shape (string [=])
      --iterations         iterations count for one profiling (int [=10240])
      --percentiles        percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host         cache input data into host memory (bool [=0])
  -q, --queue_size         aync wait queue size (unsigned int [=1])
  -?, --help               print this message
```
clip_image_prof 命令行示例 
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/clip_image_prof \
-m /opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod \
--norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
--space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 1000 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1


# 测试最小时延
./vaststreamx-samples/bin/clip_image_prof \
-m /opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod \
--norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
--space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 800 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0

```
clip_image_prof 命令行结果示例 
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 192.84
  latency (us):
    avg latency: 15466
    min latency: 7760
    max latency: 17841
    p50 latency: 15492
    p90 latency: 15521
    p95 latency: 15533
    p99 latency: 15556

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 140.033
  latency (us):
    avg latency: 7140
    min latency: 7009
    max latency: 7641
    p50 latency: 7134
    p90 latency: 7191
    p95 latency: 7196
    p99 latency: 7235
```

### clip_text 模型性能测试
clip_text_prof 命令行参数说明
```bash
options:
  -m, --model_prefix      model prefix of the model suite files (string [=/opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod])
      --hw_config         hw-config file of the model suite (string [=])
      --vdsp_params       vdsp preprocess parameter file (string [=../data/configs/clip_txt_vdsp.json])
  -d, --device_ids        device id to run (string [=[0]])
  -b, --batch_size        profiling batch size of the model (unsigned int [=1])
  -i, --instance          instance number or range for each device (unsigned int [=1])
      --iterations        iterations count for one profiling (int [=1024])
      --percentiles       percentiles of latency (string [=[50, 90, 95, 99]])
      --input_host        cache input data into host memory (bool [=0])
  -q, --queue_size        aync wait queue size (unsigned int [=2])
      --test_input_npz    test input npz file (string)
  -?, --help              print this message
```
clip_text_prof 命令行示例 
```bash
# 测试最大吞吐
./vaststreamx-samples/bin/clip_text_prof  \
-m /opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod \
--vdsp_params ../data/configs/clip_txt_vdsp.json \
--test_input_npz "./imagenet_label_npz_files/Afghan hound, Afghan.npz" \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 1500 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1


# 测试最小时延
./vaststreamx-samples/bin/clip_text_prof  \
-m /opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod \
--vdsp_params ../data/configs/clip_txt_vdsp.json \
--test_input_npz "./imagenet_label_npz_files/Afghan hound, Afghan.npz" \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 1000 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0

```
clip_text_prof 命令行结果示例 
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [ 0 ]
  queue size: 1
  batch size: 1
  throughput (qps): 357.636
  latency (us):
    avg latency: 8333
    min latency: 3980
    max latency: 9422
    p50 latency: 8347
    p90 latency: 8372
    p95 latency: 8381
    p99 latency: 8405

# 测试最小时延
- number of instances: 1
  devices: [ 0 ]
  queue size: 0
  batch size: 1
  throughput (qps): 282.91
  latency (us):
    avg latency: 3533
    min latency: 3507
    max latency: 3820
    p50 latency: 3522
    p90 latency: 3580
    p95 latency: 3589
    p99 latency: 3629
```


## Python Sample 
在当前文档所在目录执行

### clip_sample.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  --imgmod_prefix IMGMOD_PREFIX
                        image model prefix of the model suite files
  --imgmod_hw_config IMGMOD_HW_CONFIG
                        image model hw-config file of the model suite
  --norm_elf NORM_ELF   image model elf file
  --space2depth_elf SPACE2DEPTH_ELF
                        image model elf file
  --txtmod_prefix TXTMOD_PREFIX
                        text model prefix of the model suite files
  --txtmod_hw_config TXTMOD_HW_CONFIG
                        text model hw-config file of the model suite
  --txtmod_vdsp_params TXTMOD_VDSP_PARAMS
                        text model vdsp preprocess parameter file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input file
  --label_file LABEL_FILE
                        label file
  --dataset_filelist DATASET_FILELIST
                        input dataset filelist
  --dataset_root DATASET_ROOT
                        input dataset root
  --dataset_output_file DATASET_OUTPUT_FILE
                        dataset output file
  --strings STRINGS     test strings, split by ","
```

### clip_sample.py 命令行示例
```bash
# 测试 a diagram,a dog,a cat 与 CLIP.png 的匹配分数
python3 clip_sample.py \
--imgmod_prefix /opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod \
--norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
--space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
--txtmod_prefix /opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod \
--txtmod_vdsp_params ../../data/configs/clip_txt_vdsp.json \
--device_id 0 \
--input_file ../../data/images/CLIP.png \
--strings "[a diagram,a dog,a cat]"

#结果示例
Top3:
0th, string: a diagram, score: 0.9967411160469055
1th, string: a dog, score: 0.002329525537788868
2th, string: a cat, score: 0.0009294250630773604
#说明 “a diagram” 与 图片 CLIP.png 最匹配

#跑数据集
python3 clip_sample.py \
--imgmod_prefix /opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod \
--norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
--space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
--txtmod_prefix /opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod \
--txtmod_vdsp_params ../../data/configs/clip_txt_vdsp.json \
--label_file ../../data/labels/imagenet.txt \
--device_id 0 \
--dataset_filelist /opt/vastai/vaststreamx/data/datasets/ILSVRC2012_img_val_filelist.txt \
--dataset_root /opt/vastai/vaststreamx/data/datasets/ \
--dataset_output_file clip_result.txt

# 统计精度
python3 ../../evaluation/classification/eval_topk.py  clip_result.txt  

# 统计精度结果
[VACC]:  top1_rate: 55.622 top5_rate: 82.598
```
### clip_image 模型性能分析

clip_image_prof.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_PREFIX, --model_prefix MODEL_PREFIX
                        model prefix of the model suite files
  --hw_config HW_CONFIG
                        hw-config file of the model suite
  --norm_elf NORM_ELF   normalize op elf file
  --space2depth_elf SPACE2DEPTH_ELF
                        space_to_depth op elf file
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
clip_image_prof.py 命令行示例
```bash
# 测试最大吞吐
python3 clip_image_prof.py \
-m /opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod \
--norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
--space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 1000 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1

# 测试最小时延
python3 clip_image_prof.py \
-m /opt/vastai/vaststreamx/data/models/clip_image-fp16-none-1_3_224_224-vacc/mod \
--norm_elf /opt/vastai/vaststreamx/data/elf/normalize \
--space2depth_elf /opt/vastai/vaststreamx/data/elf/space_to_depth \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 800 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```
clip_image_prof.py 命令行结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 193.86
  latency (us):
    avg latency: 15426
    min latency: 10336
    max latency: 18026
    p50 latency: 15425
    p90 latency: 15503
    p95 latency: 15523
    p99 latency: 15558

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 136.62
  latency (us):
    avg latency: 7318
    min latency: 7200
    max latency: 7878
    p50 latency: 7314
    p90 latency: 7371
    p95 latency: 7380
    p99 latency: 7446
```

### clip_text 模型性能分析

clip_text_prof.py 命令行参数说明
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
  --iterations ITERATIONS
                        iterations count for one profiling
  --queue_size QUEUE_SIZE
                        aync wait queue size
  --percentiles PERCENTILES
                        percentiles of latency
  --input_host INPUT_HOST
                        cache input data into host memory
```
clip_text_prof.py 命令行示例
```bash
# 测试最大吞吐
python3 clip_text_prof.py \
-m /opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod \
--vdsp_params ../../data/configs/clip_txt_vdsp.json \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 1500 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 1


# 测试最小时延
python3 clip_text_prof.py \
-m /opt/vastai/vaststreamx/data/models/clip_text-fp16-none-1_77-vacc/mod \
--vdsp_params ../../data/configs/clip_txt_vdsp.json \
--device_ids  [0] \
--batch_size  1 \
--instance 1 \
--iterations 1200 \
--percentiles "[50,90,95,99]" \
--input_host 1 \
--queue_size 0
```
clip_text_prof.py 命令行结果示例
```bash
# 测试最大吞吐
- number of instances: 1
  devices: [0]
  queue size: 1
  batch size: 1
  throughput (qps): 356.43
  latency (us):
    avg latency: 8361
    min latency: 4377
    max latency: 8988
    p50 latency: 8363
    p90 latency: 8418
    p95 latency: 8434
    p99 latency: 8462

# 测试最小时延
- number of instances: 1
  devices: [0]
  queue size: 0
  batch size: 1
  throughput (qps): 278.85
  latency (us):
    avg latency: 3585
    min latency: 3563
    max latency: 4027
    p50 latency: 3580
    p90 latency: 3594
    p95 latency: 3639
    p99 latency: 3653
```
