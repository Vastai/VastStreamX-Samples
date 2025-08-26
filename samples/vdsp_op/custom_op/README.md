# Custom Op 

本目录介绍 Custom Op 的用法

使用Custom Op 的步骤为
1. 定义好自定义算子配置参数所需的结构体 
2. 通过 vsx::CustomOperator 加载 自定义算子
3. 设置好配置参数并通过 run_sync 调用算子