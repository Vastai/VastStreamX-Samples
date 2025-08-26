# image process samples

本目录提供基于vsx ImageProcess API 开发的sample,主要的API有:   
cvtcolor: 颜色空间转换   
resize：图片缩放     
crop: 图片裁剪     
yuvflip: yuv图片翻转     
warpaffine: 图片仿射变换     
resize_copy_make_border:  图片缩放，并在图片外侧添加border    
batch_crop_resize: 图片根据输入多个rectangle进行多次裁剪，并对裁剪结果resize到统一的size
scale： 图片根据输入的多个shape，缩放成多个size的图片   

## C++ Sample
### image_process 命令行格式
```bash
  -d, --device_id      device id to run (unsigned int [=0])
      --input_file     input image file (string [=../data/images/dog.jpg])
      --output_file    output image file (string [=./image_process_result.jpg])
  -?, --help           print this message
```
### image_process 命令示例
在 build 目录里执行  
```bash
./vaststreamx-samples/bin/image_process \
--device_id 0 \
--input_file ../data/images/dog.jpg \
--output_file ./image_process_result.jpg
```

### image_process 结果示例

```bash
image_rgb_planar format is RGB_PLANAR
image_yuv_nv12 format is YUV_NV12
image_gray format is GRAY
image_416_416 size is ( 416 x 416 )
image_600_800 size is ( 600 x 800 )
image_crop_224_224 size is ( 224 x 224 )
image_flip_x size is ( 768 x 576 )
image_flip_y size is ( 768 x 576 )
image_warpaffine size is ( 768 x 576 )
image_resize_copy_make_border size is ( 512 x 512 )
image_resize_copy_make_border[0] size is ( 640 x 640 )
image_resize_copy_make_border[1] size is ( 640 x 640 )
images_scale[0] size is ( 224 x 224 )
images_scale[1] size is ( 416 x 416 )
images_scale[2] size is ( 800 x 600 )
```

## Python Sample

### image_process.py 命令行格式
```bash
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input file
```

### image_process.py 命令示例

```bash
python3 image_process.py \
--device_id 0 \
--input_file ../../data/images/dog.jpg \
--output_file image_process_result.jpg
```

### image_process.py 结果示例

```bash
vsx_image_rgb_planar format is ImageFormat.RGB_PLANAR
vsx_image_yuv_nv12 format is ImageFormat.YUV_NV12
vsx_image_gray format is ImageFormat.GRAY
vsx_image_resize_416_416 size is ( 416 x  416 )
vsx_image_resize_416_416 size is ( 600 x  800 )
vsx_image_resize_416_416 size is ( 224 x  224 )
vsx_image_yuvflip_x_axis size is ( 768 x  576 )
vsx_image_yuvflip_y_axis size is ( 768 x  576 )
vsx_image_warpaffine size is ( 768 x  576 )
vsx_image_resize_copy_make_border size is ( 512 x  512 )
vsx_images_batch_crop_resize[0] size is ( 224 x  224 )
vsx_images_batch_crop_resize[1] size is ( 224 x  224 )
vsx_images_batch_crop_resize[0] size is ( 224 x  224 )
vsx_images_batch_crop_resize[1] size is ( 416 x  416 )
vsx_images_batch_crop_resize[2] size is ( 800 x  600 )
```