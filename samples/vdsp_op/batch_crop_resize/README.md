# BatchCropResize Op

BatchCropResize 的功能为，根据输入的一组 rectangle, 在同一张照片上进行 crop 操作，并 resize 到指定的 统一的 size 。

对 BatchCropResize op 的性能测试，请看 buildin_op_prof 目录


## C++ Sample 

### batch_crop_resize 命令行参数说明
```bash
options:
  -d, --device_id       device id to run (unsigned int [=0])
      --input_file      input image (string [=../data/images/dog.jpg])
      --output_file1    output image (string [=batch_crop_resize_result1.jpg])
      --output_file2    output image (string [=batch_crop_resize_result2.jpg])
      --output_size     output size [w,h] (string [=[512,512]])
      --crop_rect1      crop rect1 [x,y,w,h] (string [=[50,70,131,230]])
      --crop_rect2      crop rect1 [x,y,w,h] (string [=[60,90,150,211]])
  -?, --help            print this message
```

### batch_crop_resize 命令行示例
```bash
./vaststreamx-samples/bin/batch_crop_resize \
--device_id 0 \
--input_file ../data/images/dog.jpg \
--output_size [512,512] \
--crop_rect1 [50,70,131,230] \
--crop_rect2 [60,90,150,211] \
--output_file1 batch_crop_resize_result1.jpg \
--output_file2 batch_crop_resize_result2.jpg 
```
最终结果保存为 batch_crop_resize_result1.jpg  batch_crop_resize_result2.jpg


## Python Sample 

### batch_crop_resize.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input image
  --output_size OUTPUT_SIZE
                        resize output size [w,h]
  --output_file1 OUTPUT_FILE1
                        output image 1
  --output_file2 OUTPUT_FILE2
                        output image 2
  --crop_rect1 CROP_RECT1
                        crop rect [x,y,w,h]
  --crop_rect2 CROP_RECT2
                        crop rect [x,y,w,h]
```
### batch_crop_resize.py 命令行示例

```bash
python3 batch_crop_resize.py \
--device_id 0 \
--input_file ../../../data/images/dog.jpg \
--output_size [512,512] \
--crop_rect1 [50,70,131,230] \
--crop_rect2 [60,90,150,211] \
--output_file1 batch_crop_resize_result1.jpg \
--output_file2 batch_crop_resize_result2.jpg 
```

最终结果保存为 batch_crop_resize_result1.jpg  batch_crop_resize_result2.jpg
