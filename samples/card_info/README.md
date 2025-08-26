# Get Card Info Sample

本 sample 主要展示了如何通过 vaststreamx API 获得常见的卡 状态信息，更多信息 请参考 API文档与头文件

## C++ Sample
在build 目录下执行
```bash
./vaststreamx-samples/bin/card_info 

#输出示例
Find 2 cards in system.
0th card info:
        UUID: FCA12CD00038
        Card type: VA1-16G
        Die ID: 0, 1, 
1th card info:
        UUID: FCA12CD00053
        Card type: VA1-16G
        Die ID: 2, 3, 
Device id 0 status: 
        Temperature: 34.43 ℃.
        Power: 30.9582 W.
        Memory total: 8 GB.
        Memory free: 7.36157 GB.
        Memory used: 0.638428 GB.
        Memory usage rate: 7.98% .
        AI usage rate: 0% .
        VDSP usage rate: 0% .
        DEC usage rate: 0% .
        ENC usage rate: 0% .
```


## Python Sample
在当前目录执行
```bash
python3 card_info.py 

#输出示例
Find 2 cards in system.
0th card info:
        UUID: FCA12CD00038
        Card type:: VA1-16G
        Die ID: 0, 1, 
1th card info:
        UUID: FCA12CD00053
        Card type:: VA1-16G
        Die ID: 2, 3, 
Device id 0 status: 
        Temperature: 34.480000000000004  ℃.
        Power: 30.993935999999998 W.
        Memory total: 8.000 GB.
        Memory free: 7.362 GB.
        Memory used: 0.638 GB.
        Memory usage rate: 7.98%.
        AI usage rate: 0.00%.
        VDSP usage rate: 0.00%.
        DEC usage rate: 0.00%.
        NEC usage rate: 0.00%.
```



