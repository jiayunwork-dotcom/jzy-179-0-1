"""离散/求解使用的核心数据结构（与 HTTP 层的 pydantic 模型分开）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# 三种边界条件：
#   zero      物理表面处通量直接为零
#   extrap    物理表面外 2.13*D 处通量为零（马绍克真空边界的外推距离形式）
#   reflect   全反射（零流入 / 零净流），用于表示对称面
BoundaryType = Literal["zero", "extrap", "reflect"]
BOUNDARY_TYPES = ("zero", "extrap", "reflect")


@dataclass(frozen=True)
class Zone:
    thickness: float          # cm
    d: float                  # 扩散系数 D，cm
    sigma_a: float            # cm^-1
    nu_sigma_f: float         # cm^-1
    n_mesh: int               # 均匀网格数


@dataclass(frozen=True)
class Problem:
    """一个不可变的一维平板单群扩散工况版本。"""

    zones: tuple[Zone, ...]
    left_bc: BoundaryType
    right_bc: BoundaryType

    @property
    def total_thickness(self) -> float:
        return float(sum(z.thickness for z in self.zones))

    @property
    def total_mesh(self) -> float:
        return sum(z.n_mesh for z in self.zones)
