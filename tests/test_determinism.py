"""0.2.0 确定性与 trace 测试。

确定性是 0.2.0 的核心升级: 湿纸权重-2 求解的遍历种子由 (干点集合, 目标伴随式)
派生, 嵌入逐字节复现 —— 主项目 nsf5stego 的实验档案 repro 契约因此能对 JPEG
域启用字节级校验。trace 则把每个被修改系数的 from/to 暴露给可视化。
"""
import numpy as np

from yccstego import api


def _cover():
    rng = np.random.default_rng(7)
    yy = np.linspace(0, 200, 96)[None, :]
    xx = np.linspace(0, 80, 96)[:, None]
    base = np.clip(110 + yy + xx, 0, 255)
    gray = np.clip(base + rng.integers(-8, 9, (96, 96)), 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


MSG = "确定性 42"


def test_embed_byte_deterministic():
    jpg1, rep1 = api.embed_bytes(_cover(), MSG, p=3, quality=85)
    jpg2, rep2 = api.embed_bytes(_cover(), MSG, p=3, quality=85)
    assert jpg1 == jpg2, "相同输入必须得到完全相同的输出字节"
    assert rep1["carriers_changed"] == rep2["carriers_changed"]


def test_deterministic_across_global_rng_state():
    """0.1.4 的病灶: 湿纸权重-2 用全局 RNG, 外部扰动会改变嵌入结果。"""
    jpg1, _ = api.embed_bytes(_cover(), MSG, p=3)
    np.random.seed(123)
    np.random.rand(4096)                       # 扰动全局随机状态
    jpg2, _ = api.embed_bytes(_cover(), MSG, p=3)
    assert jpg1 == jpg2, "嵌入不得依赖全局随机状态"


def test_trace_changes():
    jpg, rep = api.embed_bytes(_cover(), "trace me", p=3, quality=85, trace=True)
    ch = rep["changes"]
    assert ch, "真实嵌入必有修改轨迹"
    need = {"cell", "block", "rc", "pool", "kind", "from", "to"}
    assert all(need <= set(c) for c in ch)
    assert all(c["kind"] in ("shrink", "wet", "boost") for c in ch)
    assert all(c["pool"] in ("head", "body") for c in ch)
    # 减幅语义: to = from - sign(from); 唯一例外是湿点无解时的升幅兜底
    for c in ch:
        assert (c["to"] == c["from"] - (1 if c["from"] > 0 else -1)
                or c["kind"] == "boost"), c
    assert len(ch) == rep["carriers_changed"], "轨迹条数 == 改动系数数"
    assert {c["pool"] for c in ch} == {"head", "body"}, "头池/正文池都有修改"
    # trace 不改变嵌入结果: 带 trace 与不带 trace 的输出字节一致
    jpg2, _ = api.embed_bytes(_cover(), "trace me", p=3, quality=85)
    assert jpg == jpg2


def test_wet_points_reported():
    _, rep = api.embed_bytes(_cover(), MSG, p=3)
    assert isinstance(rep["wet_points"], int) and rep["wet_points"] >= 0
