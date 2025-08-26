# Crop Op

Crop op 用于裁剪图片，输入一张图与要裁剪的区域，保存裁剪结果

对 Crop op 的性能测试，请看 buildin_op_prof 目录

## C++ Sample 

### crop 命令行参数说明
```bash
options:
  -d, --device_id      device id to run (unsigned int [=0])
      --input_file     input_image (string [=../data/images/dog.jpg])
      --output_file    output image (string [=crop_result.jpg])
      --crop_rect      output size [x,y,w,h] (string [=[33,65,416,416]])
  -?, --help           print this message
```

### crop 命令行示例

```bash
./vaststreamx-samples/bin/crop \
--device_id 0 \
--input_file ../data/images/dog.jpg \
--output_file crop_result.jpg \
--crop_rect [33,65,416,416]
```

结果保存于 crop_result.jpg


## Python Sample 

### crop.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input image
  --output_file OUTPUT_FILE
                        output image 1
  --crop_rect CROP_RECT
                        crop rect [x,y,w,h]
```

### crop.py 命令行示例

```bash
python3 crop.py \
--input_file ../../../data/images/dog.jpg \
--output_file crop_result.jpg \
--crop_rect [33,65,416,320]
```

结果保存于 crop_result.jpg
