# yccstego —— YCC(YCbCr) 亮度通道 nsF5 隐写工具

<p align="center">
  <a href="https://pypi.org/project/yccstego/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/yccstego"></a>
  <a href="https://pypi.org/project/yccstego/"><img alt="PyPI - Python" src="https://img.shields.io/pypi/pyversions/yccstego?cacheseconds=86400"></a>
  <a href="https://pypi.org/project/yccstego/"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/yccstego?cacheseconds=86400"></a>
  <a href="https://github.com/Yukinoshita-lin/yccstego/releases"><img alt="GitHub - Release" src="https://img.shields.io/github/v/release/Yukinoshita-lin/yccstego"></a>
  <a href="https://github.com/Yukinoshita-lin/yccstego/blob/main/LICENSE"><img alt="GitHub - License" src="https://img.shields.io/github/license/Yukinoshita-lin/yccstego"></a>
</p>

在 JPEG 压缩域中对 YCbCr 的 **Y（亮度）通道量化 DCT 系数** 实施 nsF5 伴随式矩阵编码隐写。
自实现标准 JPEG(DCT+量化+Huffman) 编解码，保证量化系数在“保存→解析”后逐位一致，
从而实现压缩域无失真往返嵌入。

**0.2.0 起嵌入逐字节确定**：湿纸求解的遍历种子由（干点集合，目标伴随式）派生，
相同输入得到完全相同的输出字节（跨进程/跨机器一致），实验可复现、可写进回归测试；
旧版（≤0.1.4）生成的含密图仍可正常解码。

## 安装
```bash
# 从 PyPI 安装
pip install yccstego
```
```bash
# 或本地源码开发模式安装
pip install -e .
```
> 控制台命令 `yccstego` 依赖 Python 的 `Scripts` 目录在 PATH 中；
> 若未配置可用 `python -m yccstego.cli`。

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

## 确定性与修改轨迹（0.2.0）
```python
import yccstego.api as api
jpg1, _ = api.embed_bytes("cover.png", "同一段话", p=3)
jpg2, _ = api.embed_bytes("cover.png", "同一段话", p=3)
assert jpg1 == jpg2                       # 逐字节一致

jpg, rep = api.embed_bytes("cover.png", "看算法", p=3, trace=True)
for ch in rep["changes"]:                 # 每个被修改系数的轨迹
    print(ch["block"], ch["rc"], f'{ch["from"]}->{ch["to"]}', ch["kind"], ch["pool"])
# kind: shrink=减幅 / wet=湿纸方程解 / boost=升幅兜底; pool: head=认证头 / body=正文
```
`report` 同时新增 `wet_points`（嵌入前载体中 |c|=1 的湿点个数）。

## 与 nsf5stego 主线的关系
本项目保持独立仓库/独立发版；自 [nsf5stego](https://github.com/Yukinoshita-lin/nsf5-steganography)
v1.9.0 起被其作为依赖集成（Python≥3.10 自动安装），经桥接层提供 CLI `--jpeg`、
GUI"JPEG 域"模式与实验档案 repro 的字节级重跑校验。两侧算法行为保持一致。

## 结构
- `yccstego/color.py`  RGB↔YCbCr(BT.601) 与 4:2:0 子采样
- `yccstego/dct.py`   8×8 分块 DCT/IDCT、量化表、之字扫描
- `yccstego/huffman.py` 标准 JPEG DC/AC Huffman 编解码
- `yccstego/jpeg_codec.py` 图像↔量化系数↔.jpg 位流
- `yccstego/nsf5.py`   Y 亮度量化 DCT 系数上的 nsF5 嵌入/提取(伴随式+湿纸+块置乱+图像哈希自同步)
- `yccstego/steganalysis.py` YCC 域盲隐写分析
- `yccstego/cli.py`   命令行入口

## 许可

本项目基于 **Apache License 2.0** 发布，详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。