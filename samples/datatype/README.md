# Data Type Sample

本例程介绍如何创建vsx::Image vsx::Tensor，以及它们的常用方法。 主要实现了 Image Tensor 从文件中读取，写入到文件； 复制到 device、复制到host，修改 Image Tensor的值等示例。

## C++ Sample

### image 命令行参数说明

```bash
  -d, --device_id      device id to run (unsigned int [=0])
      --input_file     input image (string [=../data/images/dog.jpg])
      --output_file    output image (string [=image_out.jpg])
  -?, --help           print this message

```

### image 命令行示例
在build目录下执行  

```bash
./vaststreamx-samples/bin/image  \
--device_id 0 \
--input_file ../data/images/dog.jpg \
--output_file image_out.jpg
```


### tensor 命令行参数说明

```bash
usage: ./vaststreamx-samples/bin/tensor [options] ... 
options:
  -d, --device_id     device id to run (unsigned int [=0])
      --input_npz     input npz file (string [=])
      --output_npz    output npz file (string [=tensor_out.npz])
  -?, --help          print this message
```

### tensor 命令行示例
在build目录下执行  

```bash
./vaststreamx-samples/bin/tensor  \
--device_id 0 \
--input_npz /opt/vastai/vaststreamx/data/datasets/SQuAD_1.1/val_npz_6inputs/test_0.npz \
--output_npz tensor_out.npz
```


## Python Sample


### image.py  命令行参数说明
```bash
  -h, --help            show this help message and exit
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file
```
### image.py 命令行示例
```bash
python3 image.py \
--device_id 0 \
--input_file ../../data/images/dog.jpg \
--output_file image_out.jpg
```

### tensor.py 命令行参数说明
```bash
  -h, --help            show this help message and exit
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_npz INPUT_NPZ
                        input npz
  --output_npz OUTPUT_NPZ
                        output file
```
### tensor.py 命令行示例
```bash
 python3 tensor.py \
 --device_id 0 \
--input_npz /opt/vastai/vaststreamx/data/datasets/SQuAD_1.1/val_npz_6inputs/test_0.npz \
--output_npz tensor_out.npz
```





