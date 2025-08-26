
## coincheung版本

### step.1 获取预训练模型
```
# CoinCheung
link: https://github.com/CoinCheung/BiSeNet
branch: master
commit: f2b901599752ce50656d2e50908acecd06f7eb47
```

基于仓库脚本转换onnx和torchscript：[tools/export_onnx.py](https://github.com/CoinCheung/BiSeNet/blob/master/tools/export_onnx.py)

```python
# https://github.com/CoinCheung/BiSeNet/blob/master/tools/export_onnx.py#L42
# 修改配置参数和尺寸信息等，增加以下代码增加导出torchscript

scripted_model = torch.jit.trace(net, dummy_input ).eval()
scripted_model.save(checkpoint.replace(".pth", ".torchscript.pt"))
scripted_model = torch.jit.load(checkpoint.replace(".pth", ".torchscript.pt"))
```

> Tips
> 
> 模型forward内存在argmax，当前提供的onnx等已移除
> 

### step.2 准备数据集
- 下载[cityscapes](https://www.cityscapes-dataset.com/)数据集，解压
- 处理好的数据集
  - 测试图像：[leftImg8bit/val](http://10.23.4.235:8080/datasets/seg/cityscapes/leftImg8bit/val/?download=zip)
  - 测试图像npz：[leftImg8bit/val_npz](http://10.23.4.235:8080/datasets/seg/cityscapes/leftImg8bit/val_npz/?download=zip)
  - 测试图像npz_datalist.txt：[npz_datalist.txt](http://10.23.4.235:8080/datasets/seg/cityscapes/npz_datalist.txt)
  - 测试图像对应Mask：[gtFine/val](http://10.23.4.235:8080/datasets/seg/cityscapes/gtFine/val/?download=zip)

### step.3 模型转换
1. 获取[vamc](../../../docs/doc_vamc.md)模型转换工具
2. 根据具体模型修改配置文件，[coincheung_config.yaml](../vacc_code/build/coincheung_config.yaml)
3. 执行转换

   ```bash
    vamc build ../vacc_code/build/coincheung_config.yaml
   ```
### step.4 模型推理
1. runmodel
   > `engine.type: debug`
  - 方式1：[coincheung_sample_runmodel.py](../vacc_code/runmodel/coincheung_sample_runmodel.py)，进行runmodel推理和eval评估，计算miou
  - 方式2：也可使用vamc的run功能
    - 确保[coincheung_config.yaml](../vacc_code/build/coincheung_config.yaml)内dataset.path为验证集数据集路径，数据量为全部，执行以下命令后会在三件套同级目录下生成推理结果nzp目录
    ```bash
    vamc run ../vacc_code/build/coincheung_config.yaml
    ```
    - 使用[coincheung_vamp_eval.py](../vacc_code/vdsp_params/coincheung_vamp_eval.py)解析npz，绘图并统计精度（保证上面跑完全量的验证集）：
    ```bash
    python ../vacc_code/vdsp_params/coincheung_vamp_eval.py \
    --src_dir cityscapes/leftImg8bit/val \
    --gt_dir cityscapes/gtFine/val \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir deploy_weights/bisenetv2-fp16-none-3_736_960-debug-result \
    --input_shape 736 960 \
    --draw_dir npz_draw_result
    ```

2. vsx
    > `engine.type: vacc`

- [doc_vsx.md](../../../docs/doc_vsx.md)，参考文档安装推理`vsx`工具
- [coincheung_vsx_inference.py](../vacc_code/vsx/coincheung_vsx_inference.py)，配置相关参数，执行进行runstream推理及获得精度指标


### step.5 性能精度
1. 获取[vamp](../../../docs/doc_vamp.md)工具
2. 基于[image2npz.py](../../common/utils/image2npz.py)，将评估数据集转换为npz格式，生成对应的`npz_datalist.txt`
    ```bash
    python ../../common/utils/image2npz.py \
    --dataset_path cityscapes/leftImg8bit/val \
    --target_path  cityscapes/leftImg8bit/val_npz \
    --text_path npz_datalist.txt
    ```
3. 性能测试，配置vdsp参数[coincheung-bisenetv2-vdsp_params.json](../vacc_code/vdsp_params/coincheung-bisenetv2-vdsp_params.json)，执行：
    ```bash
    vamp -m deploy_weights/bisenetv2-int8-kl_divergence-3_736_960-vacc/bisenetv2 \
    --vdsp_params ../vacc_code/vdsp_params/coincheung-bisenetv2-vdsp_params.json \
    -i 1 p 1 -b 1
    ```
4. 精度测试，推理得到npz结果：
    ```bash
    vamp -m deploy_weights/bisenetv2-int8-kl_divergence-3_736_960-vacc/bisenetv2 \
    --vdsp_params vacc_code/vdsp_params/coincheung-bisenetv2-vdsp_params.json \
    -i 1 p 1 -b 1 \
    --datalist npz_datalist.txt \
    --path_output npz_output
    ```
5. [coincheung_vamp_eval.py](../vacc_code/vdsp_params/coincheung_vamp_eval.py)，解析npz结果，绘图并统计精度：
   ```bash
    python ../vacc_code/vdsp_params/coincheung_vamp_eval.py \
    --src_dir cityscapes/leftImg8bit/val \
    --gt_dir cityscapes/gtFine/val \
    --input_npz_path npz_datalist.txt \
    --out_npz_dir ./npz_output \
    --input_shape 736 960 \
    --draw_dir npz_draw_result \
    --vamp_flag
   ```


### Tips
- BiseNetv2需基于指定版本：1.5.1_rc1
- 性能验证环境：vamp 2.1.0，vastpipe: 2.2.5，新版vamp需要的vdsp参数格式有变化
- 
    <details><summary>eval metrics</summary>

    ```
    bisenetv1_city_new.pth
    validation pixAcc: 94.125, mIoU: 67.629

    bisenetv1_city0629-fp16-none-3_736_960-vacc
    validation pixAcc: 94.099, mIoU: 67.457

    bisenetv1_city0629-int8-kl_divergence-3_736_960-vacc
    validation pixAcc: 93.604, mIoU: 64.615

    bisenetv2_city.pth
    validation pixAcc: 94.713, mIoU: 69.778

    bisenetv2_city0629-fp16-none-3_736_960-vacc
    validation pixAcc: 94.719, mIoU: 69.769

    bisenetv2_city0629-int8-kl_divergence-3_736_960-vacc
    validation pixAcc: 94.503, mIoU: 68.111
    ```
    </details>
