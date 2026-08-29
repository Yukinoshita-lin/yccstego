"""Y 域 nsF5 往返/篡改检测测试。独立可运行（python tests/test_nsf5.py），也兼容 pytest。"""
import os, sys, numpy as np
import contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yccstego import jpeg_codec as J
from yccstego import nsf5

try:
    import pytest
    parametrize = pytest.mark.parametrize
    raises = pytest.raises
    _RUNNER = "pytest"
except Exception:                       # 非 pytest 环境：自行驱动
    def _parametrize(argnames, argvalues):
        def deco(fn):
            fn._params = (argnames, argvalues); return fn
        return deco
    def _raises(exc):
        return contextlib.nullcontext() if exc is None else pytest_raises(exc)
    class _Raises:
        def __init__(self, exc): self.exc = exc; self.value = None
        def __enter__(self):
            class C:
                def __exit__(self_, et, ev, tb):
                    assert et and issubclass(et, self.exc), f"expected {self.exc}, got {et}"
                    self.value = ev; return True
            return C()
        def __exit__(self, *a): pass
    def pytest_raises(exc): return _Raises(exc)
    parametrize = _parametrize
    raises = _raises
    _RUNNER = "manual"


def make_rgb(h=96, w=128, seed=1):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, w)[None, :]
    yy = np.linspace(0, 1, h)[:, None]
    base = np.sin(3 * x) * np.cos(5 * yy) * 60 + 120
    R = np.clip(base + 3 * x * 255 * np.random.uniform(0.5, 1.5, (h, w)), 0, 255)
    G = np.clip(120 + 60 * np.sin(8 * x) * np.exp(-(yy - 0.5) ** 2 * 20) + rng.normal(0, 8, (h, w)), 0, 255)
    B = np.clip(base - 40 * x + rng.normal(0, 8, (h, w)), 0, 255)
    return np.stack([R, G, B], -1).astype(np.uint8)


def embed_roundtrip(msg, p, password, rgb=None):
    rgb = rgb if rgb is not None else make_rgb(128, 160)
    c0 = J.YCC(rgb, 85)
    y_new, rep = nsf5.embed_into_y(c0.Y, c0.Cb, c0.Cr, msg, p=p, password=password)
    c1 = J.YCC.__new__(J.YCC)
    c1.h, c1.w, c1.orig_shape, c1.quality = c0.h, c0.w, c0.orig_shape, 85
    c1.qlum, c1.qchr = c0.qlum, c0.qchr
    c1.Y, c1.Cb, c1.Cr = y_new, c0.Cb, c0.Cr
    return J.YCC.from_bytes(c1.to_bytes()), rep     # 压缩域无失真往返


P_LIST = [1, 2, 3, 4]
MSG = ("Hello YCC nsF5 roundtrip 0123456789 abcdef ") * 2


def test_roundtrip_ascii(p=3):
    back, rep = embed_roundtrip(MSG, p, "pw")
    out, cov, tampered, hm = nsf5.extract_from_y(back.Y, back.Cb, back.Cr, p=p, password="pw")
    assert out == MSG, f"p={p} message mismatch"
    assert tampered is False
    assert hm is True


def test_wrong_password_fails():
    back, _ = embed_roundtrip("secret-message", 3, "pw-a")
    out, *_ = nsf5.extract_from_y(back.Y, back.Cb, back.Cr, p=3, password="pw-b")
    assert out is None


def test_chroma_tamper_detected():
    c0 = J.YCC(make_rgb(128, 160), 85)
    y_new, _ = nsf5.embed_into_y(c0.Y, c0.Cb, c0.Cr, "hello", p=3, password="pw")
    bad_cb = c0.Cb.copy(); bad_cb.reshape(-1)[7] += 1
    _, _, tampered, hm = nsf5.extract_from_y(y_new, bad_cb, c0.Cr, p=3, password="pw")
    assert hm is False and tampered is True


def test_y_body_tamper_detected():
    c0 = J.YCC(make_rgb(128, 160), 85)
    y_new, _ = nsf5.embed_into_y(c0.Y, c0.Cb, c0.Cr, "hello", p=3, password="pw")
    bad = y_new.copy(); bad.reshape(-1)[80] += 1
    out, *_ = nsf5.extract_from_y(bad, c0.Cb, c0.Cr, p=3, password="pw")
    assert out is None


def test_capacity_error():
    rgb = make_rgb(16, 16)
    c0 = J.YCC(rgb, 85)
    try:
        nsf5.embed_into_y(c0.Y, c0.Cb, c0.Cr, "x" * 5000, p=3, password="pw")
        raise AssertionError("expected CapacityError")
    except nsf5.CapacityError:
        pass


def _main():
    failures = 0
    for p in P_LIST:
        try:
            test_roundtrip_ascii(p); print(f"PASS test_roundtrip_ascii p={p}")
        except AssertionError as e:
            failures += 1; print(f"FAIL test_roundtrip_ascii p={p}: {e}")
    for name in ["test_wrong_password_fails", "test_chroma_tamper_detected",
                 "test_y_body_tamper_detected", "test_capacity_error"]:
        try:
            globals()[name](); print(f"PASS {name}")
        except Exception as e:
            failures += 1; print(f"FAIL {name}: {e!r}")
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    _main()