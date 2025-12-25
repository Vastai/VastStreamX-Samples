<div id=top align="center">

![logo](./data/images/logo.png)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![company](https://img.shields.io/badge/vastaitech.com-blue)](https://www.vastaitech.com/)

</div>

---

VaststreamX Samples 是基于 VaststreamX API 开发的示例程序，用户可以参考其中的 Sample 学习如何使用 VaststreamX API。在每个 Sample 目录下有 `README` 文件，用户可以通过 `README` 文件了解 Sample 的功能和用法。

VastStreamX 提供了 Stream 管理、内存管理、模型加载与执行、算子加载与执行、多媒体数据处理等 C++ 或 Python API，用于实现目标识别、图像分类、视频增强等功能。VastStreamX 详细说明可参考《VastStreamX 用户手册》。

## 依赖软件

使用瀚博半导体的AI加速卡测试 VaststreamX Samples 前， 需联系销售代表获取部署软件包。

- 版本号：[VVI-25.11](https://developer.vastaitech.com/downloads/delivery-center?&version_uid=484479150992789504)

## 快速安装

获取部署软件包后安装流程如下所示。详细安装及使用说明可参考对应组件的文档。

其中，xxx表示版本相关信息，请根据实际情况替换。 

<details><summary><b>步骤 1.</b> 安装驱动。</summary>

1. 查询是否安装加速卡。

```shell
lspci -d:0100 |wc -l
```

2. 查询是否安装驱动。

```shell
lsmod | grep -i vastai_pci
```

3. 查询驱动版本。

```shell
cat /dev/vastai0_version | grep "Driver"
```

4. 安装驱动。


```shell   
sudo ./vastai_driver_install_xxx.run install
```
</details>

<details><summary><b>步骤 2.</b> 设置加速卡参数。</summary>

1. 查询加速卡信息。

```shell
sudo vasmi list
```


2. 根据业务情况设置加速卡Bbox模式。

```shell
sudo vasmi setcardmode <Card Mode> -d <Device ID> -y
```
> Card Mode可根据 `sudo vasmi setcardmode --help` 查询获取。

3. 使能日志记录等监控功能。
```shell
nohup sudo valogger &
```

</details>

<details><summary><b>步骤 3.</b> 部署模型运行环境（ARM/X86）。</summary>



1. 安装 VastStream。

```shell
sudo ./ai-xxx.bin
```
2. 安装 VAMC。
```shell
pip install vamc-xxx.whl
```
3. 安装 VastStreamX。

   - Python：`pip install vaststreamx-xxx.whl`
   - C++：`sudo ./vaststreamx-xxx.bin`

</details>


## 支持的模型与开发语言

|       类别       |                                 模型                                 | C++ | Python |
| ---------------- | -------------------------------------------------------------------- | --- | ------ |
| 图片分类         | <ul><li>resnet50</li><li>vit-base</li><li>swin-transformer</li><li>mobile-vit</li></ul>                     | ✓   | ✓      |
| 目标检测         | <ul><li>yolov5m</li><li>detr_r50</li><li>grounding_dino</li><li>yolo_world</li><li>rtdetr</li><li>alpr_yolov10n | ✓   | ✓      |
| 人脸分割         | bisenet</li></ul>                                                               | ✓   | ✓      |
| 文字检测         | dbnet                                                                | ✓   | ✓      |
| 语义分割         | fcn                                                                  | ✓   | ✓      |
| 实例分割         | <ul><li>yolov8_seg</li><li>mask2former</li></ul>                                               | ✓   | ✓      |
| OCR              | <ul><li>resnet34_vd</li><li>ppocr_v4</li></ul>                                                 | ✓   | ✓      |
| 人脸检测         | retinaface_resnet50                                                  | ✓   | ✓      |
| 人脸特征         | facenet                                                              | ✓   | ✓      |
| 图像超分         | <ul><li>rcan</li><li>edsr</li></ul>                                                            | ✓   | ✓      |
| 行人跟踪         | bytetrack                                                            | ✓   | ✓      |
| 人脸增强         | gpen                                                                 | ✓   | ✓      |
| 人脸关键点检测   | hih                                                                  | ✓   | ✓      |
| 动态模型         | yolov5s_dynamic                                                      | ✓   | ✓      |
| 显著性目标检测   | <ul><li>u2net</li><li>isnet</li></ul>                                                          | ✓   | ✓      |
| 3D目标检测       | point-pillar                                                         | ✓   | ✓      |
| 多模态预训练模型 | <ul><li>clip</li><li>siglip</li></ul>                                                          | ✓   | ✓      |
| 视觉基础模型     | dinov2                                                               | ✓   | ✓      |
| 图像压缩解压     | <ul><li>elic</li><li>mlic++</li></ul>                                                         |     | ✓      |
| 人体姿态识别     | yolov8-pose                                                          |     | ✓      |
| 异常检测         | efficient_ads                                                        |     | ✓      |



基于 Sample 运行得到的各模型的精度和性能，请查看对应Sample目录下的Readme 说明。 不同的加速卡或不同的频率下，Sample 运行所得到的性能结果可能存在差异。

各 Sample 文档列出的性能数据，如无特别说明，均基于 OCLK=835MHz DCLK=650MHz ECLK=200MHz的配置测得。

## 支持的功能

|      类别       |                                         示例                                          | C++ | Python |
| --------------- | ------------------------------------------------------------------------------------- | --- | ------ |
| 图片处理API     | <ul><li>CvtColor</li><li>Resize</li><li>Crop</li><li>WarpAffine</li><li>ResizeCopyMakeBorder</li><li>BatchCropResize</li></ul>             | ✓   | ✓      |
| 内置 VDSP 算子  | <ul><li>cvtcolor</li><li>resize</li><li>scale</li><li>flip</li><li>warpaffine</li><li>crop</li><li>copy_make_border</li><li>batch_crop_rezie</li></ul>   | ✓   | ✓      |
| 自定义算子      | <ul><li>argmax</li><li>brightness</li><li>norma_tensor_3ch</li></ul>                                                  | ✓   | ✓      |
| JPEG 编解码     | <ul><li>Jpeg_Decode</li><li>Jpeg_Encode</li></ul>                                                              | ✓   | ✓      |
| H264/H265编解码 | <ul><li>Video_Decode</li><li>Video_Encode</li></ul>                                                              | ✓   | ✓      |
| 视频拉流        | <ul><li>VideoCapture</li><li>video_writer</li></ul>                                                             | ✓   | ✓      |
| AI + 编解码     | <ul><li>decode + detection</li><li>decode + detection + encode</li><li>decode + pose</li><li>decode + mot + encode | ✓   | ✓      |
| 卡状态获取      | card_info                                                                             | ✓   | ✓      |



## 依赖项


- gcc/g++: 9.4.0  
- python: 3.8.10 
- cmake: 3.22  
- opencv: 3.4.10
- glog
- openblas: 0.3.28
- libtorch: 2.4.0


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

参考各 Sample 的 Readme，执行对应的指令。

## 运行 Python Samples

参考各个 Sample 目录下的 Readme，执行对应的指令。

## 更新说明

- 2025-10-28
  - 增加sample: decode_detection_encode_multi

- 2025-03-11
  - 增加 UT 与 code style check
  - custom op 运行方式 由 execute 转为 run_sync

- 2025-01-20
  - 新增模型: grounding_dino  yolo_world dinov2 elic mask2former  
  - bert_qa 与 market_bot_r50 有bug，暂未修复
