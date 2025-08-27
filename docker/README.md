# Build Docker Image

```bash

sudo docker build . -f Dockerfile -t vsx-samples:v0.4 

```

注：Dockerfile 搭建的是 整个仓库所有 samples 所需的环境。 若只是想运行某个sample，可以注释掉不需要的依赖库，模型，数据集。 对于Dockerfile里无法通过链接下载的资源，用户改成copy本地文件或者在启动容器后，copy 到 容器里。
