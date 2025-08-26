# Warpaffine op

Warpaffine op实现图像的仿射变换，需要输入仿射变换矩阵和输出的size

对 Warpaffine op 的性能测试，请看 buildin_op_prof 目录

Warpaffine op当前支持的格式是YUV_NV12 GRAY
## C++ Sample

### warpaffine 命令行参数说明
```bash
options:
  -d, --device_id      device id to run (unsigned int [=0])
      --input_file     input image (string [=../data/images/dog.jpg])
      --output_file    output image (string [=scale_result.jpg])
      --output_size    output size [w,h] (string [=[320,320]])
      --matrix         warpaffine matirx, [x0,x1,x2,y0,y1,y2] (string [=[0.7890625, -0.611328125, 56.0, 0.611328125, 0.7890625, -416.0]])
  -?, --help           print this message
```

### warpaffine 命令行示例
```bash
./vaststreamx-samples/bin/warpaffine \
--device_id 0 \
--input_file ../data/images/dog.jpg \
--matrix [0.7890625,-0.611328125,56.0,0.611328125,0.7890625,-416.0] \
--output_size [640,640] \
--output_file warpaffine_reusult.jpg
```

结果保存在 warpaffine_reusult.jpg


## Python Sample
### warpaffine.py 命令行参数说明
```bash
optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input image
  --output_size OUTPUT_SIZE
                        output size [w,h]
  --matrix MATRIX       warp affine matrix, [x0,x1,x2,y0,y1,y2]
  --output_file OUTPUT_FILE
                        output image
```

### warpaffine.py 命令行示例

```bash
python3 warpaffine.py \
--device_id 0 \
--input_file ../../../data/images/dog.jpg \
--matrix [0.7890625,-0.611328125,56.0,0.611328125,0.7890625,-416.0] \
--output_size [640,640] \
--output_file warpaffine_reusult.jpg
```

结果保存在 warpaffine_reusult.jpg
