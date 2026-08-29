"""yccstego 命令行入口：embed / extract / analyze。

例：
    yccstego embed cover.png out.jpg -m "hello" -p 3 -k 口令
    yccstego extract out.jpg -p 3 -k 口令
    yccstego analyze out.jpg
"""
from __future__ import annotations

import argparse
import json
import sys

from .steganalysis import _SENS


def _out_jpg(path, data: bytes):
    with open(path, "wb") as f:
        f.write(data)


def cmd_embed(args):
    try:
        jpg_bytes, rep = _embed_impl(args.infile, args.message, args.p, args.password,
                                     args.quality, args.truncate)
    except Exception as e:
        print(f"[错误] {e}", file=sys.stderr); sys.exit(2)
    _out_jpg(args.outfile, jpg_bytes)
    print(f"已嵌入 → {args.outfile}  ({len(jpg_bytes)} 字节)")
    print(f"  正文载体数：{rep['body_pool']}  画像哈希：{rep['cover_hash']}")
    if rep.get("truncated"):
        print(f"  提示：消息超容量，已按 UTF-8 安全截断为 {rep['embedded_chars']} 字符")
    if args.json:
        print(json.dumps({**rep, "outfile": args.outfile}, ensure_ascii=False))


def _embed_impl(infile, message, p, password, quality, truncate=False):
    from .api import embed_bytes
    return embed_bytes(infile, message, p=p, password=password, quality=quality,
                       truncate=truncate)


def cmd_extract(args):
    try:
        msg, cov, tampered, head_match = _extract_impl(args.infile, args.p, args.password)
    except Exception as e:
        print(f"[错误] {e}", file=sys.stderr); sys.exit(2)
    if msg is None:
        print(f"提取失败：{'发现篡改' if tampered else '口令错误或非本工具生成'} "
              f"(header哈希匹配={head_match})")
        sys.exit(1)
    print(f"解密消息：{msg}")
    print(f"覆盖图像哈希：{cov}  篡改标记：{tampered}")
    if args.json:
        print(json.dumps({"message": msg, "cover_hash": cov, "tampered": tampered,
                          "header_match": head_match}, ensure_ascii=False))


def _extract_impl(infile, p, password):
    from .api import extract_bytes
    data = _read(infile)
    return extract_bytes(data, p=p, password=password)


def cmd_analyze(args):
    try:
        res = _analyze_impl(args.infile, args.sensitivity)
    except Exception as e:
        print(f"[错误] {e}", file=sys.stderr); sys.exit(2)
    print(f"文件：{args.infile}  AC系数数：{res.get('n_ac')}")
    print(f"  DCT 卡方 p 值    : {res.get('dct_chi2_pvalue')}")
    print(f"  奇数幅值占比      : {res.get('parity_odd')}   (基线 {res.get('base_parity')})")
    print(f"  DCT 隐写概率      : {res.get('stego_probability_dct'):.3f}")
    print(f"  DCT 判定          : {res.get('verdict')}")
    if res.get("pix_probability") is not None:
        print(f"  [像素域佐证] 概率 : {res.get('pix_probability'):.3f}  判定：{res.get('pix_verdict')}")
    if args.json:
        print(json.dumps(res, ensure_ascii=False, default=str))


def _analyze_impl(infile, sensitivity):
    from .api import analyze_bytes
    data = _read(infile)
    return analyze_bytes(data, sensitivity=sensitivity)


def _read(path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def build_parser():
    ap = argparse.ArgumentParser(prog="yccstego",
                                 description="JPEG 压缩域(Y 亮度 DCT) nsF5 隐写工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="向图像嵌入文本")
    e.add_argument("infile", help="输入图像（png/jpg/...）")
    e.add_argument("outfile", help="输出 .jpg")
    e.add_argument("-m", "--message", required=True, help="要嵌入的文本（UTF-8，中英文均可）")
    e.add_argument("-p", "--p", type=int, default=3, help="矩阵编码参数 p（1..8，默认3）")
    e.add_argument("-k", "--password", default="", help="口令（可选）")
    e.add_argument("-q", "--quality", type=int, default=85, help="JPEG 质量（1..100）")
    e.add_argument("-t", "--truncate", action="store_true",
                   help="消息超容量时按 UTF-8 安全截断（默认超容量则报错）")
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_embed)

    x = sub.add_parser("extract", help="从 .jpg 提取消息")
    x.add_argument("infile", help="输入 .jpg")
    x.add_argument("-p", "--p", type=int, default=3)
    x.add_argument("-k", "--password", default="")
    x.add_argument("--json", action="store_true")
    x.set_defaults(func=cmd_extract)

    a = sub.add_parser("analyze", help="盲隐写分析")
    a.add_argument("infile", help="输入 .jpg")
    a.add_argument("-s", "--sensitivity", choices=list(_SENS), default="均衡",
                   help="判定灵敏度")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_analyze)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()