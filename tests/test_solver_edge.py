"""热启动插值与奇异算子的边界单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.discretize import discretize
from app.core.models import Problem, Zone
from app.core.reactions import interpolate_flux
from app.core.solver import SingularOperatorError, source_iteration


def test_interpolate_flux_handles_changed_mesh():
    old = np.linspace(0.0, 10.0, 5)          # 0,2.5,5,7.5,10
    phi = np.array([0.0, 2.5, 5.0, 2.5, 0.0])  # 线性“三角”形
    new = np.array([0.0, 1.0, 5.0, 9.0, 12.0])  # 含越界点
    out = interpolate_flux(old, phi, new)
    np.testing.assert_allclose(out, [0.0, 1.0, 5.0, 1.0, 0.0], atol=1e-12)
    assert np.all(np.isfinite(out))


def test_interpolate_preserves_zero_at_vacuum_face():
    # 旧解在真空边界端通量为低小值，插值到更细网格端点不外推出负/外推值
    old = np.array([0.5, 1.5, 2.5])
    phi = np.array([0.1, 0.5, 0.2])
    new = np.array([0.25, 1.5, 2.75])
    out = interpolate_flux(old, phi, new)
    assert np.all(out >= 0.0)


def test_reflective_nonabsorbing_raises_singular():
    # 两端全反射、零吸收：算子有常数零空间，应明确报奇异而不是 500/NaN
    p = Problem((Zone(20.0, 1.0, 0.0, 0.1, 20),), "reflect", "reflect")
    m = discretize(p)
    with pytest.raises(SingularOperatorError):
        source_iteration(m, tol_k=1e-10, tol_phi=1e-10, max_iter=100)


def test_solve_rejects_bad_hot_start_shape():
    p = Problem((Zone(10.0, 1.0, 0.1, 0.12, 20),), "zero", "zero")
    m = discretize(p)
    with pytest.raises(ValueError):
        source_iteration(m, tol_k=1e-8, tol_phi=1e-8, max_iter=10,
                         phi0=np.zeros(19))  # 长度对不上


def test_k_inf_limit_for_thick_pure_reflecting_core():
    # 单区燃料、两端反射但有吸收：有界解，厚板 k -> k_inf（从下方逼近）
    ks = []
    for a in (50.0, 200.0, 1000.0):
        p = Problem((Zone(a, 1.0, 0.1, 0.12, int(a)),),
                    "reflect", "reflect")
        m = discretize(p)
        out = source_iteration(m, tol_k=1e-11, tol_phi=1e-11,
                               max_iter=200000)
        assert out.converged
        ks.append(out.k)
    # 50cm 已远大于扩散长度，k 随厚度上升并逼近 k_inf=1.2
    # （三者都在机器精度上贴着 1.2，用容差容纳浮点末位抖动）
    assert ks[1] + 1e-9 >= ks[0] and ks[2] + 1e-9 >= ks[1]
    assert all(k < 1.2 + 1e-9 for k in ks)
    assert abs(ks[2] - 1.2) < 1e-3
    assert ks[0] > 1.0  # 厚反射板应超临界
