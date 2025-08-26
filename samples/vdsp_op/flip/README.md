# Flip Op

Flip op 用于图像翻转，支持水平翻转与垂直翻转

flip_type -- x : 上下翻转
flip_type -- y : 左右翻转

对 Flip op 的性能测试，请看 buildin_op_prof 目录

注： flip op 当前仅支持 YUV_NV12格式

## C++ Sample 

### flip 命令行参数说明
```bash
options:
  -d, --device_id      device id to run (unsigned int [=0])
      --input_file     input_image (string [=../data/images/dog.jpg])
      --output_file    output image (string [=crop_result.jpg])
      --flip_type      flip type x or y (string [=x])
  -?, --help           print this message
```

### flip 命令行示例
```bash
./vaststreamx-samples/bin/flip \
--input_file ../data/images/dog.jpg \
--flip_type y \
--output_file flip_result.jpg
```
结果保存为 flip_result.jpg

## Python Sample 

### flip.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input image
  --output_file OUTPUT_FILE
                        output image
  --flip_type FLIP_TYPE
                        flip type x or y
```

### flip.py 命令行示例
```bash
python3 flip.py \
--input_file ../../../data/images/dog.jpg \
--flip_type x \
--output_file flip_result.jpg
```
结果保存为 flip_result.jpg


