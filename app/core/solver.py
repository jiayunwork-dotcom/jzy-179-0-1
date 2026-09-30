"""特征值求解：自写三对角追赶（Thomas）+ 源迭代。

控制方程离散为

        A phi = (1/k) F phi

A 为三对角矩阵（吸收 + 界面导纳 + 边界导纳，均在对角块上），
F 是对角的裂变权重矩阵。源迭代（功率迭代）：

    1. 取归一化裂变源 s = F phi；
    2. 解三对角方程 A phi_raw = s；
    3. 用瑞利比更新 k：k_new = <F phi_raw,1> / <s,1>
       （源已归一化，未归一化解的裂变源净放大倍数就是 k 的新估计）；
    4. 归一化 phi_raw 使总裂变率等于目标值；
    5. 同时检查 k 的相对变化与通量的相对 L2 残差，两个判据都满足才收敛。

到上限未收敛时返回 converged=False，并带最后一步残差，上层据此显式报失败，
绝不返回半成品当结果。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .discretize import Mesh


class SingularOperatorError(RuntimeError):
    """三对角矩阵奇异（物理上正常工况不应出现）。"""


@dataclass
class SolveOutcome:
    converged: bool
    k: float
    phi: np.ndarray
    iterations: int
    residual_k: float
    residual_phi: float
    message: str = ""


def thomas_solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray,
                 rhs: np.ndarray) -> np.ndarray:
    """求解三对角方程组，算法手写（不调用 SciPy / np.linalg）。

    lower[0]、upper[N-1] 为占位。纯标量追赶以避免任何内置矩阵求解。
    """
    n = diag.size
    if n == 0:
        raise SingularOperatorError("空矩阵")

    cp = np.empty(n)
    dp = np.empty(n)

    if diag[0] == 0.0:
        raise SingularOperatorError("三对角矩阵首主元为零")
    inv = 1.0 / diag[0]
    cp[0] = upper[0] * inv
    dp[0] = rhs[0] * inv

    for i in range(1, n):
        denom = diag[i] - lower[i] * cp[i - 1]
        if denom == 0.0:
            raise SingularOperatorError(f"三对角矩阵在第 {i} 行主元为零")
        inv = 1.0 / denom
        cp[i] = upper[i] * inv
        dp[i] = (rhs[i] - lower[i] * dp[i - 1]) * inv

    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _normalize_fission(phi: np.ndarray, fission_weight: np.ndarray,
                       total_fission_rate: float) -> np.ndarray:
    total = float(np.dot(fission_weight, phi))
    if total <= 0.0:
        # 非裂变区域占多数时仍可用通量范数兜底，保证迭代能走下去
        total = float(np.sqrt(np.dot(phi, phi)))
        if total <= 0.0:
            phi = np.ones_like(phi)
            total = float(np.sqrt(phi.size))
    return phi * (total_fission_rate / total)


def source_iteration(
    mesh: Mesh,
    *,
    tol_k: float,
    tol_phi: float,
    max_iter: int,
    total_fission_rate: float = 1.0,
    phi0: np.ndarray | None = None,
    k0: float | None = None,
) -> SolveOutcome:
    """源迭代求 k_eff 与基模通量。

    约定：phi 始终保持“总裂变率 = total_fission_rate”的归一化形式。
    每一代：

        s = F phi                    （归一化裂变源）
        phi_raw = A^{-1} s           （三对角追赶，尚未归一化）
        k_new = <F phi_raw,1>/<s,1>  （原始放大率，直接是 k 的新估计，
                                        不要再乘 k_old——源已归一化）
        phi_new = normalize(phi_raw)
    """
    n = mesh.n
    f = mesh.fission

    if phi0 is None:
        phi = np.full(n, total_fission_rate / max(float(np.sum(f)), 1e-30))
    else:
        phi = np.array(phi0, dtype=float)
        if phi.shape != (n,):
            raise ValueError("热启动通量形状与当前网格不匹配")
        if np.any(~np.isfinite(phi)):
            raise ValueError("热启动通量含非有限值")
    phi = _normalize_fission(phi, f, total_fission_rate)

    # 初始 k：有热启动值用之，否则取 1
    k_old = float(k0) if k0 is not None and k0 > 0.0 else 1.0
    res_k = np.inf
    res_phi = np.inf

    for it in range(1, max_iter + 1):
        source = f * phi
        source_rate = float(np.sum(source))
        if source_rate <= 0.0:
            return SolveOutcome(False, k_old, phi, it - 1, res_k, res_phi,
                                "无裂变源，无法进行特征值迭代")

        phi_raw = thomas_solve(mesh.lower, mesh.diag, mesh.upper, source)
        if np.any(~np.isfinite(phi_raw)):
            return SolveOutcome(False, k_old, phi, it - 1, res_k, res_phi,
                                "三对角求解出现非有限值")

        new_source_rate = float(np.dot(f, phi_raw))
        if new_source_rate <= 0.0:
            return SolveOutcome(False, k_old, phi, it, np.inf, np.inf,
                                "新一代裂变源非正，系统无可维持的链式反应")
        # Rayleigh 商：未归一化解的裂变率 / 上代归一化源强。
        # 源已经归一化，这里直接是 k 的新估计，不能再乘 k_old。
        k_new = new_source_rate / source_rate

        phi_new = _normalize_fission(phi_raw, f, total_fission_rate)

        # 两个独立判据：k 的相对变化、归一化通量的相对 L2 变化
        res_k = abs(k_new - k_old) / max(abs(k_new), 1e-30)
        denom = float(np.sqrt(np.dot(phi_new, phi_new)))
        res_phi = float(np.sqrt(np.dot(phi_new - phi, phi_new - phi)) / denom)

        phi = phi_new
        k_old = k_new

        if res_k < tol_k and res_phi < tol_phi:
            return SolveOutcome(True, k_old, phi, it, res_k, res_phi,
                                "已收敛")

    return SolveOutcome(False, k_old, phi, max_iter, res_k, res_phi,
                        f"达到最大迭代步数 {max_iter} 仍未收敛")
