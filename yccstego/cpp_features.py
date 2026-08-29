"""ctypes 封装 yccstego/cpp/dcffeatures.dll，提供与 steganalysis.analyze_y 对齐的特征。

DLL 缺失时降级：本模块可用但 *_dll 函数返回 None，供上层短路。
"""
from __future__ import annotations

import ctypes
import os

_DLL = None
_dll_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "cpp", "dcffeatures.dll")


def _load():
    global _DLL
    if _DLL is None and os.path.exists(_dll_path):
        try:
            _DLL = ctypes.CDLL(_dll_path)
            _DLL.dcffeatures_analyze_y.argtypes = [
                ctypes.POINTER(ctypes.c_int16), ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_double)]
            _DLL.dcffeatures_analyze_y.restype = None
        except OSError:
            _DLL = None
    return _DLL


def available() -> bool:
    return _load() is not None


def dll_path() -> str:
    return _dll_path


def analyze_y_dll(y):  # -> dict | None
    """用 C++ DLL 计算 Y 量化系数特征。y 排布需为块主序 (R,C,8,8)。"""
    lib = _load()
    if lib is None:
        return None
    import numpy as np
    arr = np.ascontiguousarray(y, dtype=np.int16).reshape(-1)
    R, C = int(y.shape[0]), int(y.shape[1])
    if arr.size != R * C * 64:
        raise ValueError(f"Y 系数长度 {arr.size} 与 {R}x{C} 块不符")
    out = np.zeros(7, dtype=np.float64)
    lib.dcffeatures_analyze_y(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                              R, C,
                              out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return {
        "n_ac": int(out[0]),
        "parity_odd": out[1],
        "unit_frac": out[2],
        "chi_stat": out[3],
        "chi_p": out[4],
        "df": int(out[5]),
        "ac_plane_entropy": out[6],
    }