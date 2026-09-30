"""一维平板单群扩散的有限体积离散。

网格采用以“物理网格点”为单元中心的网格（cell-centered finite volume）。
对每个网格点 i 做积分平衡：

    J_{i-1/2} - J_{i+1/2} + Sigma_a,i * h_i * phi_i = (nuSigma_f,i * h_i / k) phi_i

其中界面净流 J = -D dphi/dx。关键点是区界面的处理：

* 界面两侧 phi 连续、J 连续（两侧 D 可以不同）；
* 不用“把两侧截面/D 取平均”的近似，而是从 J 连续严格消去界面未知通量，
  得到调和导纳

        t_{i,i+1} = 2 / (h_i/D_i + h_{i+1}/D_{i+1})

  均匀材料时它自然退化为 D/h。

边界（外推距离 delta = 2.13*D，Robin 条件 J = D/delta * phi_s）：
* zero    物理表面上 phi=0，表面导纳 2D/h（网格点到表面 h/2）；
* extrap  网格点距表面 h/2，表面与网格点间取线性梯度：

              t0 = D / (h/2 + delta)

          该“面元”导纳在网格加密时二阶收敛到 Robin 精确本征值
          tan(Ba/2)=1/(B delta)（不是近似公式 pi/(a+2delta)，
          解析参考见 reference.py）；delta->0 时回到 2D/h；
* reflect 零净流对称面，导纳 0。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Problem

EXTRAPOLATION_FACTOR = 2.13


@dataclass
class Mesh:
    n: int
    centers: np.ndarray          # 各网格点 x 坐标
    widths: np.ndarray           # 各网格点对应控制体宽度 h
    zone_slices: list[slice]     # 各区占用的下标
    lower: np.ndarray            # 三对角矩阵 A 的下对角（lower[0] 不用）
    diag: np.ndarray
    upper: np.ndarray            # 上对角（upper[N-1] 不用）
    fission: np.ndarray          # 裂变权重 F_i = nuSigma_f_i * h_i
    absorption_weight: np.ndarray  # Sigma_a_i * h_i
    conductance_left: float      # 左端表面导纳（泄漏 = 导纳 * phi_0）
    conductance_right: float     # 右端表面导纳
    d_edge_left: float           # 左端物理表面到零通量面的外推距离
    d_edge_right: float


def _interface_transmiss(h_left: float, d_left: float,
                         h_right: float, d_right: float) -> float:
    """由界面 J 连续推出的调和导纳 2/(h_i/D_i + h_{i+1}/D_{i+1})。"""
    return 2.0 / (h_left / d_left + h_right / d_right)


def discretize(problem: Problem) -> Mesh:
    zones = problem.zones
    widths_parts: list[np.ndarray] = []
    centers_parts: list[np.ndarray] = []
    slices: list[slice] = []
    x_cursor = 0.0
    start = 0
    for z in zones:
        h = z.thickness / z.n_mesh
        widths_parts.append(np.full(z.n_mesh, h, dtype=float))
        offsets = (np.arange(z.n_mesh, dtype=float) + 0.5) * h
        centers_parts.append(x_cursor + offsets)
        slices.append(slice(start, start + z.n_mesh))
        x_cursor += z.thickness
        start += z.n_mesh

    widths = np.concatenate(widths_parts)
    centers = np.concatenate(centers_parts)
    n = widths.size

    d_per_cell = np.empty(n)
    sa_per_cell = np.empty(n)
    nuf_per_cell = np.empty(n)
    for z, sl in zip(zones, slices):
        d_per_cell[sl] = z.d
        sa_per_cell[sl] = z.sigma_a
        nuf_per_cell[sl] = z.nu_sigma_f

    # 内部界面导纳
    internal_t = 2.0 / (widths[:-1] / d_per_cell[:-1]
                        + widths[1:] / d_per_cell[1:])

    # 边界导纳
    h0, d0 = widths[0], d_per_cell[0]
    hn, dn = widths[-1], d_per_cell[-1]
    if problem.left_bc == "zero":
        g_left = 2.0 * d0 / h0
        d_edge_left = 0.0
    elif problem.left_bc == "extrap":
        d_edge_left = EXTRAPOLATION_FACTOR * d0
        g_left = d0 / (0.5 * h0 + d_edge_left)
    else:  # reflect
        g_left = 0.0
        d_edge_left = 0.0
    if problem.right_bc == "zero":
        g_right = 2.0 * dn / hn
        d_edge_right = 0.0
    elif problem.right_bc == "extrap":
        d_edge_right = EXTRAPOLATION_FACTOR * dn
        g_right = dn / (0.5 * hn + d_edge_right)
    else:
        g_right = 0.0
        d_edge_right = 0.0

    lower = np.zeros(n)
    upper = np.zeros(n)
    lower[1:] = -internal_t
    upper[:-1] = -internal_t

    diag = sa_per_cell * widths.copy()
    diag[0] += g_left
    diag[-1] += g_right
    diag[1:] += internal_t
    diag[:-1] += internal_t

    return Mesh(
        n=n,
        centers=centers,
        widths=widths,
        zone_slices=slices,
        lower=lower,
        diag=diag,
        upper=upper,
        fission=nuf_per_cell * widths,
        absorption_weight=sa_per_cell * widths,
        conductance_left=float(g_left),
        conductance_right=float(g_right),
        d_edge_left=float(d_edge_left),
        d_edge_right=float(d_edge_right),
    )
