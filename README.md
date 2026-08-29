# yccstego —— YCC(YCbCr) 亮度通道 nsF5 隐写工具

在 JPEG 压缩域中对 YCbCr 的 **Y（亮度）通道量化 DCT 系数** 实施 nsF5 伴随式矩阵编码隐写。
自实现标准 JPEG(DCT+量化+Huffman) 编解码，保证量化系数在“保存→解析”后逐位一致，
从而实现压缩域无失真往返嵌入。

## 安装
```bash
pip install -e .
```

## 用法（CLI）
```bash
# 嵌入（消息 UTF-8，中英文均可）
yccstego embed in.png out.jpg -m "你好，ycc stego" -p 3 -k 口令
# 消息超容量时按 UTF-8 安全截断（默认则报错）
yccstego embed small.png out.jpg -m "很长很长的中文…" -p 3 -k 口令 --truncate
# 解码
yccstego extract out.jpg -p 3 -k 口令
# 隐写分析
yccstego analyze out.jpg
```

## 结构
- `yccstego/color.py`  RGB↔YCbCr(BT.601) 与 4:2:0 子采样
- `yccstego/dct.py`   8×8 分块 DCT/IDCT、量化表、之字扫描
- `yccstego/huffman.py` 标准 JPEG DC/AC Huffman 编解码
- `yccstego/jpeg_codec.py` 图像↔量化系数↔.jpg 位流
- `yccstego/nsf5.py`   Y 亮度量化 DCT 系数上的 nsF5 嵌入/提取(伴随式+湿纸+块置乱+图像哈希自同步)
- `yccstego/steganalysis.py` YCC 域盲隐写分析
- `yccstego/cli.py`   命令行入口