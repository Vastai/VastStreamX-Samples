# SCALE Op

scale op 与 resize op 功能类似，区别是 scale op 支持多个 size 输出， 而 resize op 仅支持一个输出


对 scale op 的性能测试，请看 buildin_op_prof 目录

scale op 当前仅支持 YUV_NV12 格式
## C++ Sample 

### scale 命令行参数说明
```bash
options:
  -d, --device_id       device id to run (unsigned int [=0])
      --input_file      input image (string [=../data/images/dog.jpg])
      --output_size1    output size1 [w,h] (string [=[256,256]])
      --output_size2    output size2 [w,h] (string [=[320,320]])
      --output_file1    output image1 (string [=scale_result1.jpg])
      --output_file2    output image2 (string [=scale_result2.jpg])
  -?, --help            print this message
```

### scale 命令行示例

```bash
./vaststreamx-samples/bin/scale \
--device_id 0 \
--input_file ../data/images/dog.jpg \
--output_size1 [512,512] \
--output_size2 [600,800] \
--output_file1 scale_result1.jpg \
--output_file2 scale_result2.jpg 
```

结果保存为 scale_result1.jpg ，scale_result2.jpg 

## Python Sample 

### scale.py 命令行参数说明
```bash
  -h, --help            show this help message and exit
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input image
  --output_size1 OUTPUT_SIZE1
                        resize output size [w,h]
  --output_size2 OUTPUT_SIZE2
                        resize output size [w,h]
  --output_file1 OUTPUT_FILE1
                        output image
  --output_file2 OUTPUT_FILE2
                        output image
```


### scale.py 命令行示例

```bash
python3 scale.py \
--device_id 0 \
--input_file ../../../data/images/dog.jpg \
--output_size1 [512,512] \
--output_size2 [600,800] \
--output_file1 scale_result1.jpg \
--output_file2 scale_result2.jpg 
```

结果保存为 scale_result1.jpg ，scale_result2.jpg 
