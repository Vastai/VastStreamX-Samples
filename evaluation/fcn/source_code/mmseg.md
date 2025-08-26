## mmsegmentation

### step.1 获取预训练模型

```
link: https://github.com/open-mmlab/mmsegmentation
branch: v1.0.0rc2
commit: 8a611e122d67b1d36c7929331b6ff53a8c98f539
```

使用mmseg转换代码[pytorch2torchscript.py](https://github.com/open-mmlab/mmsegmentation/blob/master/tools/pytorch2torchscript.py)，命令如下
```bash
python tools/pytorch2torchscript.py  \
    configs/fcn/fcn_r50-d8_512x512_20k_voc12aug.py \
    --checkpoint ./pretrained/mmseg/ann/fcn_r50-d8_512x512_20k_voc12aug_20200617_010715-52dc5306.pth \
    --output-file ./onnx/mmseg/ann/torchscript/fcn_r50_d8_20k-512.torchscript.pt \
    --shape 512 512
```
> onnx在build时会报错

### step.2 准备数据集
- 下载[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)数据集，解压，使用[image2npz.py](../vacc_code/vdsp_params/image2npz.py)，提取val图像数据集和转换为npz格式
- 处理好的数据集
  - 测试图像：[JPEGImages_val](http://10.23.4.235:8080/datasets/seg/VOCdevkit/VOC2012/JPEGImages_val/?download=zip)
  - 测试图像npz：[JPEGImages_val_npz](http://10.23.4.235:8080/datasets/seg/VOCdevkit/VOC2012/JPEGImages_val_npz/?download=zip)
  - 测试图像npz_datalist.txt：[npz_datalist.txt](http://10.23.4.235:8080/datasets/seg/VOCdevkit/VOC2012/npz_datalist.txt)
  - 测试图像对应Mask：[SegmentationClass](http://10.23.4.235:8080/datasets/seg/VOCdevkit/VOC2012/SegmentationClass/?download=zip)


### step.3 模型转换
1. 获取[vamc](../../../docs/doc_vamc.md)模型转换工具
2. 根据具体模型修改配置文件，[mmseg_config.yaml](../vacc_code/build/mmseg_config.yaml)
3. 执行转换

   ```bash
    vamc build ../vacc_code/build/mmseg_config.yaml
   ```

### step.4 模型推理
1. runmodel
   > `engine.type: debug`
  - 方式1：[mmseg_sample_runmodel.py](../vacc_code/runmodel/mmseg_sample_runmodel.py)，进行runmodel推理和eval评估，计算miou
  - 方式2：也可使用vamc的run功能
    - 确保[mmseg_config.yaml](../vacc_code/build/mmseg_config.yaml)内dataset.path为验证集数据集路径，数据量为全部，执行以下命令后会在三件套同级目录下生成推理结果nzp目录
    ```bash
        vamc run ../vacc_code/build/mmseg_config.yaml
    ```
    - 使用[mmseg_vamp_eval.py](../vacc_code/vdsp_params/mmseg_vamp_eval.py)解析npz，绘图并统计精度（保证上面跑完全量的验证集）：
    ```bash
        python ../vacc_code/vdsp_params/mmseg_vamp_eval.py \
        --src_dir VOC2012/JPEGImages_val \
        --gt_dir VOC2012/SegmentationClass \
        --input_npz_path npz_datalist.txt \
        --out_npz_dir deploy_weights/fcn_r50_d8_20k-int8-kl_divergence-3_512_512-debug-result \
        --input_shape 512 512 \
        --draw_dir npz_draw_result
    ```

2. vsx
    > `engine.type: vacc`

- [doc_vsx.md](../../../docs/doc_vsx.md)，参考文档安装推理`vsx`工具
- [vsx_inference.py](../vacc_code/vsx/mmseg_vsx_inference.py)，配置相关参数，执行进行runstream推理及获得精度指标



### step.5 性能精度
1. 获取[vamp](../../../docs/doc_vamp.md)工具
2. 基于[image2npz.py](../vacc_code/vdsp_params/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`，注意只转换`VOC2012/ImageSets/Segmentation/val.txt`对应的验证集图像（配置相应路径）：
    ```bash
    python ../vacc_code/vdsp_params/image2npz.py \
    --dataset_path VOC2012/JPEGImages \
    --target_path  VOC2012/JPEGImages_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[mmseg-fcn_r50_d8_20k-vdsp_params.json](../vacc_code/vdsp_params/mmseg-fcn_r50_d8_20k-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/fcn_r50_d8_20k-int8-kl_divergence-3_512_512-vacc/fcn_r50_d8_20k \
    --vdsp_params ../vacc_code/vdsp_params/mmseg-fcn_r50_d8_20k-vdsp_params.json \
    -i 2 p 2 -b 1 -s [3,512,512]
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/fcn_r50_d8_20k-int8-kl_divergence-3_512_512-vacc/fcn_r50_d8_20k \
    --vdsp_params vacc_code/vdsp_params/mmseg-fcn_r50_d8_20k-vdsp_params.json \
    -i 2 p 2 -b 1 -s [3,512,512] \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [mmseg_vamp_eval.py](../vacc_code/vdsp_params/mmseg_vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/mmseg_vamp_eval.py \
    --src_dir VOC2012/JPEGImages_val \
    --gt_dir VOC2012/SegmentationClass \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir ./npz_output \
    --input_shape 512 512 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```

### Tips
- onnx在build时报错：` /jenkins/workspace/02_SW_DEV_AI_automation_test_daily/ai-compiler-test-scripts/vastai/vaststream/tvm/src/relay/pass/vacc/check_outside_pipeline_ops.cc/CheckOutsidePipelineOps:78: max, subtract, sum, divide,  operators is outside of pipeline 已放弃 (核心已转储)`
- `1.5.2 SP1`版本解决torchscript，fp16 run问题，~~[OP#16455](http://openproject.vastai.com/projects/model-debug/work_packages/16455/activity)~~