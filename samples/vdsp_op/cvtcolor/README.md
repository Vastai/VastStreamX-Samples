# CvtColor Op

本目录展示 cvtcolor的用法， 性能测试请参考 build_in_op_prof 目录

## C++ Sample

cvtcolor.cpp 展示 SINGLE_OP_CVT_COLOR 的用法，将 BGR_INTERLEAVE 格式转为 RGB_PLANAR 格式。最后在转为cv::Mat类型时，又转回bgr_interleave格式，保存的图片与原图相同。

### cvtcolor 命令行参数说明
```bash
options:
  -d, --device_id        device id to run (unsigned int [=0])
      --input_file       input image (string [=../data/images/dog.jpg])
      --output_file      output image (string [=cvtcolor_result.jpg])
      --cvtcolor_code    cvtcolor code (string [=bgr2rgb_interleave2planar])
  -?, --help             print this message
```

### cvtcolor 命令示例   
在build目录里运行    
```bash
./vaststreamx-samples/bin/cvtcolor \
--device_id 0 \
--input_file ../data/images/dog.jpg \
--output_file cvtcolor_result.jpg \
--cvtcolor_code rgb2yuv_nv12_planar 
```
最终结果保存为 cvtcolor_result.jpg


## Python Sample

cvtcolor.py 展示 SINGLE_OP_CVT_COLOR 的用法，将 BGR_INTERLEAVE 格式转为 RGB_PLANAR 格式。

### cvtcolor.py 命令行参数说明
```bash
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file

```


###  cvtcolor.py 命令行示例 
在本目录下运行   
```bash
python3 cvtcolor.py \
--device_id 0 \
--input_file ../../../data/images/dog.jpg \
--output_file cvtcolor_result.jpg \
--cvtcolor_code rgb2yuv_nv12_planar 
```

最后将图片保存到 cvtcolor_result.jpg
