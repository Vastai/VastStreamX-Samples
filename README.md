# VaststreamX Samples

VaststreamX Samples 是基于 VaststreamX API 开发的示例程序，用户可以参考其中的 Sample 学习如何使用 VaststreamX API。在每个 Sample 目录下有 `README` 文件，用户可以通过 `README` 文件了解 Sample 的功能和用法。


## 支持的模型与开发语言
|   类别             |  模型                                              | C++  | Python |
|--------------------|---------------------------------------------------|------|--------|
|  图片分类           | resnet50  vit-base  swin-transformer mobile-vit   | ✓    |  ✓    |
|  目标检测           | yolov5m  detr_r50 grounding_dino  yolo_world      | ✓    |  ✓    |
|  人脸分割           | bisenet                                           | ✓    |  ✓    |
|  文字检测           |  dbnet                                            | ✓    |  ✓    |
|  语义分割           | fcn                                               | ✓    |  ✓    |
|  实例分割           | yolov8_seg    mask2former                         | ✓    |  ✓    |
|  OCR               | resnet34_vd     ppocr_v4                          | ✓    |  ✓    |
|  人脸检测           | retinaface_resnet50                               | ✓    |  ✓    |
|  人脸特征           | facenet                                           | ✓    |  ✓    |
|  图像超分           | rcan       edsr                                   | ✓    |  ✓    |
|  行人跟踪           | bytetrack                                         | ✓    |  ✓    |
|  人脸增强           | gpen                                              | ✓    |  ✓    |
| 人脸关键点检测       | hih                                               | ✓    |  ✓    |
|  动态模型            | yolov5s_dynamic                                   | ✓    |  ✓    |
| 显著性目标检测        | u2net       isnet                                 | ✓    |  ✓    |
| 3D目标检测           |  point-pillar                                      | ✓    |  ✓    |
|  CLIP               | clip                                               | ✓    |  ✓    |
| 视觉基础模型         | dinov2                                             | ✓    |  ✓    |
|  图像压缩解压        | elic                                                |     |  ✓    |


**各模型的精度与性能，请查看对应sample的readme. 在不同的卡或不同的频率下，性能会有差异**    
各sample的readme里列出的性能数据，除特别指出频率外，均是在 OCLK=835MHz DCLK=650MHz ECLK=200MHz下测试出来的   

## 支持的功能
|   类别             |     例子                                                                       | C++  | Python |
|--------------------|--------------------------------------------------------------------------------|------|--------|
|   图片处理API      |   CvtColor  Resize Crop WarpAffine ResizeCopyMakeBorder BatchCropResize        | ✓    |  ✓    |
|   内置 VDSP 算子   |   cvtcolor resize scale flip warpaffine crop copy_make_border batch_crop_rezie | ✓    |  ✓    |
|   自定义算子       |   argmax brightness norma_tensor_3ch                                           | ✓    |  ✓    |
|   jpeg编解码      |  Jpeg_Decode Jpeg_Encode                                                        | ✓    |  ✓    |
|   h264 h265编解码  | Video_Decode Video_Encode                                                      | ✓    |  ✓    |
|   视频拉流        |  VideoCapture  video_writer                                                     | ✓    |  ✓    |
|   AI + 编解码     |  decode + detection  decode + detection + encode                                | ✓    |  ✓    |
|   卡状态获取      |   card_info                                                                      | ✓    |  ✓    |


      
## 版本要求

Vastai Compiler: 2.8.0   
Vaststream SDK: 2.5.0   
gcc/g++: 9.4.0  
python: 3.8.10   
Vaststreamx: 2.8.3  
cmake: 3.22  

## 依赖项
```bash
opencv: 3.4.10
glog
openblas: 0.3.28
libtorch: 2.4.0
```

## 编译 C++ Samples

```bash
cd vaststreamx-samples
source scripts/activate.sh
mkdir build && cd build
cmake ..
make -j
make install
```
## 运行 C++ Samples

参考各 sample 的 readme.md 

## 运行 Python Samples


参考各个 sample 目录下的 readme ，执行对应的指令




## Update
- 2025-03-11
    - 增加 UT 与 code style check 
    - custom op 运行方式 由 execute 转为 run_sync 

- 2025-01-20   
    - 新增模型: grounding_dino  yolo_world dinov2 elic mask2former  
    - bert_qa 与 market_bot_r50 有bug，暂未修复 



