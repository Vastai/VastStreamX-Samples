# Test Scripts for CICD 

## Build Docker Image 

```bash
sudo docker build . -f Dockerfile -t vsx-samples:unit-test 
```


## Build Cpp Code 

```bash

cd vaststreamx-samples

sudo docker run --rm --privileged --name=vsx-samples -it -v `pwd`:/work vsx-samples:unit-test  bash

# in container
mkdir -p build
cd build
cmake ..
make -j
make install

```


## Run Test Scripts 


```bash

cd vaststreamx-samples

sudo docker run --rm --privileged --name=vsx-samples -it -v `pwd`:/work  vsx-samples:unit-test  bash

# in container

# 所有测试用例被分为五类: fast slow codec ai_integration codec_integration
# fast: 表示耗时很短的 测试用例，很快能跑完，不包括 codec
# slow: 表示耗时很长的 测试用例，至少耗时半小时以上，不包括 codec
# codec： 表示编解码相关的 测试用例，耗时都比较小，不包括: slow fast
# ai_integration: 表示 CICD 时需要执行的 AI 测试用例
# codec_integration: 表示 CICD 时需要执行的 CODEC 测试用例
# fast slow codec 互相之间没有交集


# 脚本 run_tests.py 命令参数说明
# --folder  Python脚本所在目录，以test_开头的Python文件会被执行
# --device_ids 选择哪些 die id 作为执行测试脚本的 die ，有多少个die，就会创建多少个 进程
# --report_dir 用来保存测试报告的文件夹
# --merged_html 每个测试脚本都会生成一个测试报告，但最后会merge成一个html文件，merged_html则是最终的报告
# --case_type 即上述的 fast slow codec ai_integration codec_integration 五类


# test ai_integration case 
rm -f test_reports/*
python tests/run_tests.py \
--folder tests/tests  \
--device_ids [0,1,2,3]  \
--merged_html vsx_samples_integration_merge_report.html \
--report_dir test_reports \
--case_type ai_integration

# test codec_intergration case 
rm -f test_reports/*
python tests/run_tests.py \
--folder tests/tests  \
--device_ids [0,1,2,3]  \
--merged_html vsx_samples_codec_integration_report.html \
--report_dir test_reports \
--case_type codec_integration

# test fast case 
rm -f test_reports/*
python tests/run_tests.py \
--folder tests/tests  \
--device_ids [0,1,2,3]  \
--merged_html vsx_samples_fast_merge_report.html \
--report_dir test_reports \
--case_type fast

# test slow case 
rm -f test_reports/*
python tests/run_tests.py \
--folder tests/tests  \
--device_ids [0,1,2,3]  \
--merged_html vsx_samples_slow_merge_report.html \
--report_dir test_reports \
--case_type slow

# test codec case 
rm -f test_reports/*
python tests/run_tests.py \
--folder tests/tests  \
--device_ids [0,1,2,3]  \
--merged_html vsx_samples_codec_report.html \
--report_dir test_reports \
--case_type codec


# test one case 
pytest tests/tests/test_ocr_e2e.py -m ai_integration --html=one-test-report.html --self-contained-html --device_id=2
pytest tests/tests/test_elic.py --html=one-test-report.html --self-contained-html --device_id=2
pytest tests/tests/test_video_decode.py --html=one-test-report.html --self-contained-html --device_id=2
pytest tests/tests/test_buildin_op_prof.py --html=one-test-report.html --self-contained-html --device_id=2

pytest tests/tests/test_mask2former.py  --html=mask2former-report.html --self-contained-html --device_id=2

# test multi cases
pytest tests/tests/test_detr.py tests/tests/test_buildin_op_prof.py tests/tests/test_classification.py --html=one-test-report.html --self-contained-html --device_id=2

pytest tests/tests/test_yolov5m-640.py tests/tests/test_buildin_op_prof.py tests/tests/test_classification.py --html=one-test-report.html --self-contained-html --device_id=2

pytest -m fast tests/tests/test_yolo_world.py  tests/tests/test_decode_detection_encode.py tests/tests/test_video_decode.py tests/tests/test_batch_crop_resize_op.py tests/tests/test_buildin_op_prof.py tests/tests/test_crnn.py  --device_id=0



# test one api
pytest tests/tests/test_video_capture.py::test_video_capture_cpp --html=one-test-report.html --self-contained-html --device_id=1





```
