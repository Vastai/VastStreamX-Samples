# Resize Op

resize.cpp 展示 SINGLE_OP_RESIZE 的用法, 用于将输入图片 resize 到指定尺寸。 resize_bgr888_to_bgr888 函数用于展示 单纯的resize操作，输入输出格式一样。resize_rgb888_to_rgb_planar 函数展示了 resize 并 cvtcolor的操作，resize的过程中，顺便输出的格式调整为 rgb_planar。


对 resize op 的性能测试，请看 buildin_op_prof 目录

## C++ Sample 

### resize 命令行参数说明
```bash
options:
  -d, --device_id      device id to run (unsigned int [=0])
      --input_file     input image (string [=../data/images/dog.jpg])
      --output_size    input image (string [=[256,256]])
      --output_file    output image (string [=resize_result.jpg])
  -?, --help           print this message
```

### resize 命令示例
在build目录里运行    
```bash
./vaststreamx-samples/bin/resize \
--device_id 0 \
--input_file ../data/images/dog.jpg \
--output_size "[512,512]"   \
--output_file resize_result.jpg
```
最终结果保存为 resize_result.jpg


## Python Sample 

resize.py 展示 SINGLE_OP_RESIZE 的用法，将 原图 resize 到指定尺寸。

### resize.py 命令行参数说明  

```bash
optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input image
  --output_size OUTPUT_SIZE
                        resize output size
  --output_file OUTPUT_FILE
                        output image
```

### resize.py 命令示例 
在本目录下运行   
```bash
python3 resize.py  \
 --device_id 0 \
 --input_file ../../../data/images/cat.jpg \
 --output_size "[600,800]" \
 --output_file resize_result.jpg
```
结果保存为  resize_result.jpg

