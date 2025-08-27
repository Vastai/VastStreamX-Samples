# MLIC++ Sample

本目录提供基于 mlic 模型的图像压缩与解压sample

## 模型信息

| 模型信息       | 值                                                           |
| -------------- | ------------------------------------------------------------ |
| 来源           | [github](https://github.com/JiangWeibeta/MLIC) [modelzoo](-) |
| 输入 shape     | [ (1,3,512,768) ]                                            |
| INT8量化方式   | -                                                            |
| 官方精度       | "PSNR": 35.8066                                              |
| VACC FP16 精度 | "PSNR": 35.3577                                              |
| VACC INT8 精度 | -                                                            |

> PSNR 精度基于 [kodak](https://www.kaggle.com/datasets/sherylmehta/kodak-dataset) 数据集中的 768x512 图片运行得到。

## 数据集准备

下载 mlic 模型  到 /opt/vastai/vaststreamx/data/models 里并解压
下载 图片集 kodak_768_512.tar.gz 到 /opt/vastai/vaststreamx/data/datasets 里并解压
下载 mlic 预训练模型  mlicpp_mse_q5_2960000.pth.tar  到 /opt/vastai/vaststreamx/data/pre-trained 里，不用解压

## Python Sample

### 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### mlic_inference.py 脚本命令行说明

```bash
optional arguments:
  -h, --help            show this help message and exit
  --gaha_model_prefix GAHA_MODEL_PREFIX
                        ga_ha model prefix of the model suite files
  --gaha_hw_config GAHA_HW_CONFIG
                        hw-config file of the model suite
  --gaha_vdsp_params GAHA_VDSP_PARAMS
                        vdsp preprocess parameter file
  --hs_model_prefix HS_MODEL_PREFIX
                        h_s model prefix of the model suite files
  --hs_hw_config HS_HW_CONFIG
                        hs_hw-config file of the model suite
  --gs_model_prefix GS_MODEL_PREFIX
                        g_s model prefix of the model suite files
  --gs_hw_config GS_HW_CONFIG
                        gs_hw-config file of the model suite
  --torch_model TORCH_MODEL
                        torch model file
  --tensorize_elf_path TENSORIZE_ELF_PATH
                        tensorize elf file
  -d DEVICE_ID, --device_id DEVICE_ID
                        device id to run
  --input_file INPUT_FILE
                        input file
  --output_file OUTPUT_FILE
                        output file
  --dataset_path DATASET_PATH
                        input dataset path
  --dataset_output_path DATASET_OUTPUT_PATH
                        dataset output path
  --patch PATCH         padding patch size (default: 64)
```

### elic_inference.py 脚本命令示例

```bash
# 测试单张图片
python3 mlic_inference.py  \
--gaha_model_prefix  /opt/vastai/vaststreamx/data/models/mlic/compress_ga_ha_sim_512_768_vacc_runmodel \
--gaha_vdsp_params  ../../../data/configs/mlic_compress_gaha_rgbplanar.json \
--hs_model_prefix  /opt/vastai/vaststreamx/data/models/mlic/compress_hs_sim_512_768_vacc_runmodel \
--gs_model_prefix   /opt/vastai/vaststreamx/data/models/mlic/decompress_gs_sim_512_768_vacc_runmodel \
--torch_model  /opt/vastai/vaststreamx/data/pre-trained/mlicpp_mse_q5_2960000.pth.tar \
--device_id  0 \
--input_file  /opt/vastai/vaststreamx/data/dataset/kodak/kodim01.png \
--output_file  mlic_result.png
# 结果保存为 mlic_result.png

# 测试数据集
python3 mlic_inference.py  \
--gaha_model_prefix  /opt/vastai/vaststreamx/data/models/mlic/compress_ga_ha_sim_512_768_vacc_runmodel \
--gaha_vdsp_params  ../../../data/configs/mlic_compress_gaha_rgbplanar.json \
--hs_model_prefix  /opt/vastai/vaststreamx/data/models/mlic/compress_hs_sim_512_768_vacc_runmodel \
--gs_model_prefix   /opt/vastai/vaststreamx/data/models/mlic/decompress_gs_sim_512_768_vacc_runmodel \
--torch_model  /opt/vastai/vaststreamx/data/pre-trained/mlicpp_mse_q5_2960000.pth.tar \
--device_id  0 \
--dataset_path /opt/vastai/vaststreamx/data/dataset/kodak \
--dataset_output_path dataset_outputs

# 结果保存到 dataset_outputs 文件夹，并输出PSNR。压缩与解压时间受CPU影响较大，重点关注psnr值
# CPU min MHz:         800.0000
    Ave Compress time:596.7426829867892 ms
    Ave Decompress time:648.1502321031359 ms
    Ave PNSR:35.357761837544075

```
