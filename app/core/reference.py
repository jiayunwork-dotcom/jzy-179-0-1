"""均匀裸板的解析参考解。

单群扩散、均匀裸堆：

        k_eff = k_inf / (1 + L^2 B^2),  其中 L^2 = D/Sigma_a

边界对应的几何曲率 B（a 为物理厚度，delta = 2.13 D）：

* 两端都在物理表面 phi=0（zero/zero）：基模 cos，B = pi / a；
* 一端真空、一端对称面：B = pi / (2 a_e)
  （zero+reflect 时 a_e=a；extrap+reflect 时 a_e=a+delta）；
* 两端 extrap（Robin 条件 -D dphi/dn = D/delta * phi_s）：
  基模 phi=cos(B(x-a/2))，代入表面条件得精确本征方程

        tan(B a / 2) = 1 / (B delta)

  用二分法求最小正根。注意 pi/(a+2delta) 只是“零通量线性外推”的近似，
  不是 Robin 条件的精确 B，二者相差 O(delta^3) 量级但不可忽略；
  二阶离散收敛的是精确 Robin 根，测试以此为准。
"""
from __future__ import annotations

import math

from app.core.models import Problem

EXTRAPOLATION_FACTOR = 2.13


def _extrap_delta(zone) -> float:
    return EXTRAPOLATION_FACTOR * zone.d


def _robin_buckling_both(a: float, delta: float) -> float:
    """tan(B a/2) = 1/(B delta) 的最小正根。根在 (0, pi/a) 内。"""
    def f(b: float) -> float:
        return math.tan(b * a * 0.5) - 1.0 / (b * delta)

    lo, hi = 1e-12, math.pi / a * (1.0 - 1e-12)
    # f 在 (0, pi/a) 单调增（tan 增、-1/(B delta) 也增）
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _robin_buckling_half(a: float, delta: float) -> float:
    """对称半板：x=0 反射，x=a 为 Robin。基模 phi=cos(Bx)，
    表面条件 D B sin(Ba) = D/delta cos(Ba) => tan(Ba) = 1/(B delta)。"""
    def f(b: float) -> float:
        return math.tan(b * a) - 1.0 / (b * delta)

    lo, hi = 1e-12, 0.5 * math.pi / a * (1.0 - 1e-12)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def geometric_buckling(problem: Problem) -> float:
    if len(problem.zones) != 1:
        raise ValueError("解析解只适用于单区均匀裸板")
    z = problem.zones[0]
    a = z.thickness
    delta = _extrap_delta(z)
    left, right = problem.left_bc, problem.right_bc

    if "reflect" in (left, right):
        if left == right:
            raise ValueError("两端全反射的裸板无离散特征值（B=0），不适用裸板公式")
        vac = left if left != "reflect" else right
        if vac == "zero":
            return math.pi / (2.0 * a)
        return _robin_buckling_half(a, delta)

    if left == "zero" and right == "zero":
        return math.pi / a

    if "extrap" in (left, right):
        # 对称组合：两端外推距离相同（同一材料）
        return _robin_buckling_both(a, delta)

    # 其余（zero+extrap 的不对称组合）裸板参考：用等效长度近似
    d_left = delta if left == "extrap" else 0.0
    d_right = delta if right == "extrap" else 0.0
    return math.pi / (a + d_left + d_right)


def analytic_k_eff(problem: Problem) -> float:
    z = problem.zones[0]
    if z.sigma_a <= 0.0:
        raise ValueError("Sigma_a 为 0 时 L^2 无定义")
    l2 = z.d / z.sigma_a
    k_inf = z.nu_sigma_f / z.sigma_a
    b2 = geometric_buckling(problem) ** 2
    return k_inf / (1.0 + l2 * b2)
