"""HTTP 接口的 pydantic 模型（入参/出参）。

边界类型等“取值合法性 + 逐项字段报错”统一交给 core.validation 处理，
这里只约束类型，避免 pydantic 提前返回与本服务不一致的错误结构。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ZoneIn(BaseModel):
    thickness: float = Field(description="区厚度 cm，必须 > 0")
    d: float = Field(description="扩散系数 D cm，必须 > 0")
    sigma_a: float = Field(description="宏观吸收截面 cm^-1，>= 0")
    nu_sigma_f: float = Field(description="nu*Sigma_f cm^-1，>= 0")
    n_mesh: int = Field(description="该区均匀网格数，正整数")


class CaseCreate(BaseModel):
    name: str
    left_bc: str
    right_bc: str
    zones: list[ZoneIn]


class ZonePatch(BaseModel):
    thickness: float
    d: float
    sigma_a: float
    nu_sigma_f: float
    n_mesh: int


class SolveOptions(BaseModel):
    start_mode: Literal["cold", "hot"] = "cold"
    total_fission_rate: float = 1.0
    tol_k: float | None = None
    tol_phi: float | None = None
    max_iter: int | None = None


class SearchRequest(BaseModel):
    target_zone: int = Field(ge=0, description="区号，从 0 开始")
    target_field: Literal["thickness", "nu_sigma_f"]
    low: float
    high: float
    max_steps: int = 60
    tol_k: float | None = None
    tol_phi: float | None = None
    max_iter: int | None = None
