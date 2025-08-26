FROM ubuntu:20.04
ENV TZ=Asia/Shanghai  
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone  

RUN cd /etc/apt/ && mv sources.list sources.list.bak  \
    && echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse" >> sources.list \
    && echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse" >> sources.list \
    && echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse" >> sources.list \
    && echo "deb http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse" >> sources.list 

RUN apt -y update && apt -y upgrade
RUN apt install -y  git make gcc-9 g++-9 unzip libgl1-mesa-dev libglib2.0-dev wget  python3-pip libeigen3-dev
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9 
RUN cd /usr/bin && ln -s python3.8 python
RUN apt install -y libboost-all-dev 

RUN pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
RUN pip config set global.trusted-host mirrors.ustc.edu.cn


RUN git config --global http.proxy http://192.168.30.26:808
RUN git config --global https.proxy http://192.168.30.26:808


RUN pip install git+https://github.com/openai/CLIP.git

RUN pip install  protobuf==3.20.3 torch compressai==1.1.5  scipy==1.7.3 \
    numpy  motmetrics  shapely  pyclipper   tqdm scikit-learn pycocotools \
    ipython cython yacs faiss-cpu torchvision  openpyxl tabulate pysodmetrics cython_bbox \
    transformers  lap pybind11 mmcv-full  tensorflow easydict lvis terminaltables pytest \
    pytest-html pytest_html_merger crayons  pytest-xdist



WORKDIR /root

# install cmake
RUN wget http://cee-release.vastai.com:32482/vastpipe-samples/data/third_party/cmake-3.22.0-linux-x86_64.tar.gz \
    && tar xf cmake-3.22.0-linux-x86_64.tar.gz -C /opt/ \
    && rm cmake-3.22.0-linux-x86_64.tar.gz
ENV PATH=/opt/cmake-3.22.0-linux-x86_64/bin:$PATH 


# install third_party
RUN wget http://cee-release.vastai.com:32482/vastpipe-samples/data/third_party/root_3rdparty_static_x86_ubuntu.tar.gz \
    && tar xf root_3rdparty_static_x86_ubuntu.tar.gz \
    && mv root_3rdparty /opt/3rdparty \
    && rm root_3rdparty_static_x86_ubuntu.tar.gz
ENV CMAKE_PREFIX_PATH=/opt/3rdparty:$CMAKE_PREFIX_PATH
ENV LD_LIBRARY_PATH=/opt/3rdparty/lib:$LD_LIBRARY_PATH

# install libtorch 
RUN wget http://cee-release.vastai.com:32482/vastpipe-samples/data/third_party/libtorch-cxx11-abi-shared-with-deps-2.4.0+cpu.zip \
    && unzip libtorch-cxx11-abi-shared-with-deps-2.4.0+cpu.zip \
    && mv libtorch /opt/ && rm libtorch-cxx11-abi-shared-with-deps-2.4.0+cpu.zip
ENV CMAKE_PREFIX_PATH=/opt/libtorch:$CMAKE_PREFIX_PATH
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH


# install openblas
RUN wget http://cee-release.vastai.com:32482/vastpipe-samples/data/third_party/OpenBLAS-0.3.28.tar.gz \
    && tar xf OpenBLAS-0.3.28.tar.gz && cd OpenBLAS-0.3.28 && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/opt/openblas \
    && make -j && make install -j && cd ../.. && rm -rf OpenBLAS-0.3.28.tar.gz OpenBLAS-0.3.28
ENV CMAKE_PREFIX_PATH=/opt/openblas:$CMAKE_PREFIX_PATH
ENV LD_LIBRARY_PATH=/opt/openblas/lib:$LD_LIBRARY_PATH    


# install vaststream
RUN wget -O 'ai-v2.8.0-20250124-121-linux-x86_64-sdk2-python3.8.bin'  \
    http://devops.vastai.com/kapis/artifact.kubesphere.io/v1alpha1/artifact?artifactid=4768 \
    && bash ./ai-v2.8.0-20250124-121-linux-x86_64-sdk2-python3.8.bin \
    && rm ai-v2.8.0-20250124-121-linux-x86_64-sdk2-python3.8.bin
ENV VACM_LOG_CFG=/home

# install odsp_plugin
RUN wget -O 'odsp_plugin-v1.0-20250124-121-linux-x86_64.tar.gz'  \
    http://devops.vastai.com/kapis/artifact.kubesphere.io/v1alpha1/artifact?artifactid=4767 \
    && mkdir -p /opt/odsp_plugin && tar xf odsp_plugin-v1.0-20250124-121-linux-x86_64.tar.gz -C /opt/odsp_plugin \
    && cd /opt/odsp_plugin/vastai && bash build.sh && cd /root \
    && rm odsp_plugin-v1.0-20250124-121-linux-x86_64.tar.gz
ENV LD_LIBRARY_PATH=/opt/odsp_plugin/vastai/lib:/opt/odsp_plugin/protobuf/lib:$LD_LIBRARY_PATH


# install vaststreamx c++
RUN wget http://cee-release.vastai.com:32482/vaststreamx/2.8.3-set/2.8.3/vaststreamx-2.8.3-8eddef9-Ubuntu18.04-x86_64.bin \
    && bash ./vaststreamx-2.8.3-8eddef9-Ubuntu18.04-x86_64.bin \
    && rm vaststreamx-2.8.3-8eddef9-Ubuntu18.04-x86_64.bin

ENV CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/opt/vastai/vaststreamx/vaststreamx
ENV LD_LIBRARY_PATH=/opt/vastai/vaststreamx/vaststreamx/lib:$LD_LIBRARY_PATH
ENV GLOG_minloglevel=2
ENV GLOG_logtostderr=1
ENV VSX_DISABLE_DEEPBIND=1 

# install vaststreamx python
RUN wget http://cee-release.vastai.com:32482/vaststreamx/2.8.3-set/2.8.3/python/ubuntu/vaststreamx-2.8.3-cp38-cp38-linux_x86_64.whl \
    && pip3 install vaststreamx-2.8.3-cp38-cp38-linux_x86_64.whl \
    && rm vaststreamx-2.8.3-cp38-cp38-linux_x86_64.whl

ENV LD_LIBRARY_PATH=/opt/vastai/vaststream/lib:$LD_LIBRARY_PATH

# get datasets
RUN cd /opt/vastai/vaststreamx/ && mkdir -p data && cd data \
    # videos images elf pre-trained
    && wget \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/videos.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/images.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/elf.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/pre-trained.tar.gz \
    && find . -name "*.tar.gz" -print0 | xargs -0 -I {} tar -zxvf {} && rm *.tar.gz \
    # tokenizer
    && mkdir -p tokenizer && cd tokenizer && wget \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/tokenizer/clip-vit-base-patch32.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/tokenizer/bert-base-uncased.tar.gz \
    && find . -name "*.tar.gz" -print0 | xargs -0 -I {} tar -zxvf {} && rm *.tar.gz && cd .. \
    # datasets
    && mkdir -p datasets && cd datasets && wget \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/bisenet.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/ch4_test_images.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/CUTE80.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/det_coco_val.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/ECSSD.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/fastreid.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/GPEN.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/ILSVRC2012_img_val.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/kitti_val.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/Kodak-512.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/Kodak.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/Kodak_1280_2048.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/lfw_mtcnnpy_160.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/mot17.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/segmentation.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/SQuAD_1.1.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/sr4k.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/WFLW.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/widerface_val.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/datasets/oxbuild_images-v1.tar.gz \
    && find . -name "*.tar.gz" -print0 | xargs -0 -I {} tar -zxvf {} && rm *.tar.gz 


# get  models
RUN cd /opt/vastai/vaststreamx/data && mkdir -p models && cd models && wget \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/bisenet-int8-kl_divergence-1_3_512_512-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/bytetrack_m_mot17-int8-percentile-1_3_800_1440-vacc-pipeline.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/clip_image-fp16-none-1_3_224_224-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/clip_text-fp16-none-1_77-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/dbnet_resnet50_vd-int8-kl_divergence-1_3_736_1280-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/detr_res50-fp16-none-1_3_1066_800-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/dinov2_vitl14_reg4-fp16-none-1_3_224_224-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/edsr_x2-int8-percentile-1_3_256_256-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/elic.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/facenet_vggface2-int8-percentile-1_3_160_160-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/fcn16s_vgg16-int8-kl_divergence-1_3_320_320-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/gpen-int8-mse-1_3_512_512-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/groundingdino.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/hih_2s-int8-percentile-1_3_256_256-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/isnet-int8-kl_divergence-1_3_320_320-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/mask2former-fp16-none-1_3_1024_1024-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/mobilevit_s-fp16-none-1_3_224_224-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/pointpillar-int8-max-16000_32_10_3_16000_1_16000-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/ppocr-v4.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/rcan-int8-max-1_3_1080_1920-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/resnet34_vd-int8-max-1_3_32_100-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/resnet50-int8-percentile-1_3_224_224-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/retinaface_resnet50-int8-kl_divergence-1_3_640_640-vacc-pipeline.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/rtdetr-fp16-none-1_3_640_640-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/swin-b-fp16-none-1_3_224_224-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/torch-yolov5s_coco-int8-percentile-Y-Y-2-none.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/u2net-int8-percentile-1_3_320_320-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/vit-b-fp16-none-1_3_224_224-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/yolov5m-int8-percentile-1_3_640_640-vacc-pipeline.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/yolov8m-int8-percentile-1_3_640_640-vacc-pipeline.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/yolov8m-seg-int8-percentile-1_3_640_640-vacc-pipeline.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/yolo_world_image-fp16-none-1_3_1280_1280_1203_512-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/yolo_world_text-fp16-none-1_16_1_16-vacc.tar.gz \
    http://cee-release.vastai.com:32482/vastpipe-samples/data/x86_linux_models.ai.2.8.0/yolox_m-int8-percentile-1_3_640_640-vacc-pipeline.tar.gz \
    && find . -name "*.tar.gz" -print0 | xargs -0 -I {} tar -zxvf {} \
    && rm *.tar.gz 

# install torchvision 
RUN wget http://cee-release.vastai.com:32482/vastpipe-samples/data/third_party/torchvision_0.19.tar.gz \
    && tar xvf torchvision_0.19.tar.gz \
    && mv  vision /opt/ && rm torchvision_0.19.tar.gz
RUN apt-get install libpng-dev  libjpeg-dev -y
RUN mkdir /opt/vision/build && cd /opt/vision/build && cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/Release \
    && make install -j8 
ENV CMAKE_PREFIX_PATH=/opt/vision/build/Release:$CMAKE_PREFIX_PATH
ENV LD_LIBRARY_PATH=/opt/vision/build/Release/lib:$LD_LIBRARY_PATH

RUN mkdir -p /work
WORKDIR /work