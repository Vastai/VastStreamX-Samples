# CopyMakeBorder Op

CopyMakeBorder op 实现的是 letterbox功能，其先resize 再 padding 


对 CopyMakeBorder op 的性能测试，请看 buildin_op_prof 目录

## C++ Sample 

### copy_make_border 命令行参数说明
```bash
options:
  -d, --device_id      device id to run (unsigned int [=0])
      --input_file     input_image (string [=../data/images/dog.jpg])
      --output_file    output image (string [=flip_result.jpg])
      --output_size    output size [width,height] (string [=x])
  -?, --help           print this message
```


### copy_make_border 命令行示例

```bash
./vaststreamx-samples/bin/copy_make_border \
--device_id 0 \
--input_file ../data/images/cat.jpg \
--output_file copy_make_border_result.jpg \
--output_size "[640,640]"
```
## Python Sample 

### copy_make_border.py 命令行参数说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input image
  --output_file OUTPUT_FILE
                        output image 1
  --output_size OUTPUT_SIZE
                        output size [width,height]
```
### copy_make_border.py 命令行示例
```bash
python3 copy_make_border.py \
--device_id 0 \
--input_file ../../../data/images/cat.jpg \
--output_file copy_make_border_result.jpg \
--output_size "[640,640]"
```


