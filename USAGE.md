# yccstego 使用说明（USAGE）

在 JPEG 压缩域中对 YCbCr 的 **Y（亮度通道）量化 DCT 系数** 实施 nsF5 伴随式矩阵编码隐写。自实现标准 JPEG（DCT+量化+Huffman）编解码，保证量化系数“保存→解析”后逐位一致，可无失真往返嵌入。

---

## 1. 安装

在项目目录 `F:\Steganography\yccstego` 下：

```bash
pip install -e .
```

依赖：`numpy`、`Pillow`（可选绘制结果图时另需 `matplotlib`、`pandas`、`joblib` 等）。

安装完成后获得命令入口 `yccstego`，等价于 `python -m yccstego.cli`。

---

## 2. 快速上手

生成一张测试图：

```bash
python -c "import numpy as np; from PIL import Image; \
Image.fromarray(np.random.default_rng(3).integers(0,255,(192,256,3),dtype=np.uint8)).save('cover.png')"
```

**嵌入**（UTF-8，中英文均可）：

```bash
yccstego embed cover.png stego.jpg -m "你好，YCC 隐写测试 42" -p 3 -k 口令
```

**解码**（需与嵌入相同的 `-p` 与 `-k`）：

```bash
yccstego extract stego.jpg -p 3 -k 口令
```

**盲隐写分析**：

```bash
yccstego analyze stego.jpg
```

> 说明：分析合成噪声图会提示“AC 系数本底随机、无法可靠判定”——这是预期行为。真实照片的低频结构更明显，magnitude-1 指纹更能体现嵌入痕迹。

---

## 3. 命令参考

### embed —— 嵌入

```
yccstego embed <infile> <outfile> -m <消息> [-p 1..8] [-k 口令] [-q 1..100] [-t] [--json]
```

- `-m/--message`：要嵌入的文本（UTF-8，中英文均可，必填）。
- `-p`：矩阵编码参数，默认 `3`。越大嵌入效率越高、但单块容量越小（见 `n=2^p-1`）。
- `-k/--password`：口令（可选）。决定隐藏路径，提取端必须一致。
- `-q/--quality`：JPEG 质量，默认 `85`。越高可嵌入容量越大。
- `-t/--truncate`：消息超容量时按 UTF-8 安全边界截断；默认超容量则报错退出。
- `--json`：以 JSON 输出报告（含 cover_hash / 容量 / 是否截断等），便于脚本处理。

### extract —— 提取

```
yccstego extract <infile> [-p 1..8] [-k 口令] [--json]
```

- 输出解密消息、覆盖图像哈希、篡改标记。
- 口令错误、参数不一致或文件被改动 → 提示失败并给出原因（篡改 / 非本工具生成）。

### analyze —— 盲隐写分析

```
yccstego analyze <infile> [-s 灵敏度] [--json]
```

- `-s/--sensitivity`：判定灵敏度（预设项，可查 `--help`），默认 `均衡`。
- 主信号为 DCT magnitude-1 指纹（nsF5/F5 把 |c|=2 减幅到 1 的特征），辅以卡方佐证。

---

## 4. Python 库接口

```python
import yccstego.api as api

# 嵌入：接受 ndarray / 图片路径 / bytes / Pillow 对象
jpg_bytes, rep = api.embed_bytes(
    "cover.png", "你好，YCC 隐写", p=3, password="口令", quality=85,
    truncate=False,          # 超容量时报错；True 则按 UTF-8 安全截断
)

# 提取
msg, cover_hash, tampered, head_match = api.extract_bytes(
    jpg_bytes, p=3, password="口令",
)

# 隐写分析（返回 DCT 域 + 像素域两套结论）
ana = api.analyze_bytes(jpg_bytes, sensitivity="均衡")
```

---

## 5. 关键参数速查

| 参数 | 作用 | 备注 |
|------|------|------|
| `-p` | 矩阵编码参数 `1..8` | 越大嵌入效率越高，容量越小，默认 3 |
| `-q` | JPEG 质量 `1..100` | 越高可嵌入容量越大，默认 85 |
| `-k` | 口令 | 决定键控隐藏路径，提取端必须一致 |
| `-t` | 超容量自动截断 | 默认超容量报错 |
| `--json` | 结构化输出 | 便于脚本/自动化 |

---

## 6. 常见问题（FAQ）

- **提示“正文载体不足 / 图像太小”**：图太小、消息太长或 `-p` 过大。
  解决：换更大的图、调高 `-q`、减小 `-p`，或加 `--truncate` 自动截断。
- **提取失败“口令错误或非本工具生成”**：`-p`/`-k` 与嵌入时不一致，或文件被改动触发篡改感知。
- **中文比英文“占地方”**：UTF-8 下每个汉字 3 字节，同容量下可装中文约为英文的 1/3；长中文优先用 `-t` 截断。
- **分析“无法可靠判定”**：图像 AC 系数本底随机（合成噪声图会如此），改用真实照片结果更显著。

---

## 7. 评测与绘图（开发用）

生成基准数据并绘制结果图：

```bash
python tools/benchmark.py          # 写出 output/bench_yccstego.csv
python tools/plot_results.py       # 写出 output/fig_*.png
```

相关图：嵌入效率 vs 码族、容量 vs 质量/尺寸、检测率 vs 密度、改动代价 vs 密度。