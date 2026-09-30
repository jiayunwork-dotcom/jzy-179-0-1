"""后处理：分区反应率、两端泄漏率、中子平衡核算；以及热启动初值插值。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .discretize import Mesh
from .models import Problem


@dataclass
class ReactionReport:
    absorption_by_zone: list[float]
    fission_by_zone: list[float]      # nuSigma_f * phi * h（裂变中子产生率）
    leakage_left: float               # 正号表示向外流出
    leakage_right: float
    production_over_k: float          # 全场裂变产生率 / k
    total_loss: float                 # 吸收 + 两端泄漏
    balance_relative_error: float


def reaction_rates(problem: Problem, mesh: Mesh, phi: np.ndarray,
                   k: float) -> ReactionReport:
    absorption = [
        float(np.dot(mesh.absorption_weight[sl], phi[sl]))
        for sl in mesh.zone_slices
    ]
    fission = [
        float(np.dot(mesh.fission[sl], phi[sl]))
        for sl in mesh.zone_slices
    ]
    leakage_left = mesh.conductance_left * float(phi[0])
    leakage_right = mesh.conductance_right * float(phi[-1])

    production = sum(fission)
    production_over_k = production / k
    total_loss = sum(absorption) + leakage_left + leakage_right
    denom = max(abs(production_over_k), abs(total_loss), 1e-30)
    balance_err = abs(production_over_k - total_loss) / denom

    return ReactionReport(
        absorption_by_zone=absorption,
        fission_by_zone=fission,
        leakage_left=float(leakage_left),
        leakage_right=float(leakage_right),
        production_over_k=float(production_over_k),
        total_loss=float(total_loss),
        balance_relative_error=float(balance_err),
    )


def interpolate_flux(old_centers: np.ndarray, old_phi: np.ndarray,
                     new_centers: np.ndarray) -> np.ndarray:
    """新网格上构造热启动初值：逐点线性插值，越界处夹住（零通量边界自动落 0）。

    选择“按新网格插值”而不是原样沿用：厚度或网格数一变，旧向量长度都对不上；
    也不做外推加速——外推初值若过冲成负值反而拖慢甚至破坏单调收敛。
    """
    return np.interp(new_centers, old_centers, old_phi,
                     left=old_phi[0], right=old_phi[-1])
