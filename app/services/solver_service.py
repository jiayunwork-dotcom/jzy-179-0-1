"""求解编排：问题组装、冷/热启动选择、留档。"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from app.core.discretize import discretize
from app.core.models import Problem, Zone
from app.core.reactions import interpolate_flux, reaction_rates
from app.core.solver import source_iteration
from app.services.storage import Storage


class SolveFailure(Exception):
    """源迭代到上限未收敛（或算不下去），带残差与步数，API 层显式报失败。"""

    def __init__(self, message: str, iterations: int,
                 residual_k: float, residual_phi: float):
        self.message = message
        self.iterations = iterations
        self.residual_k = residual_k
        self.residual_phi = residual_phi
        super().__init__(message)


class HotStartUnavailable(Exception):
    """指定热启动但上一版没有可沿用的收敛解。"""


def build_problem(zones: list[Zone], left_bc: str, right_bc: str) -> Problem:
    return Problem(zones=tuple(zones), left_bc=left_bc, right_bc=right_bc)


def patch_zone(problem: Problem, idx: int, new_zone: Zone) -> Problem:
    """只替换某一区，返回一个新的不可变 Problem（边界及其余区不动）。"""
    zones = list(problem.zones)
    zones[idx] = new_zone
    return replace(problem, zones=tuple(zones))


def solve_version(
    storage: Storage,
    *,
    case_id: str,
    version_id: int,
    start_mode: str = "cold",
    total_fission_rate: float = 1.0,
    tol_k: float = 1e-10,
    tol_phi: float = 1e-10,
    max_iter: int = 20000,
    archive: bool = True,
):
    """对指定参数版本求解。

    start_mode:
      cold       平坦初值（裂变区 phi=1），k0 取 1
      hot        取同一工况“上一版”最近一次收敛解，按新网格插值沿用
      hot_from   内部使用：显式给定 (旧网格中心, 旧phi, 旧k)
    """
    version = storage.get_version(version_id)
    if version is None:
        raise KeyError(f"version {version_id} 不存在")
    problem: Problem = version["problem"]
    mesh = discretize(problem)

    phi0 = None
    k0 = None
    actual_mode = "cold"

    if start_mode in ("hot", "hot_from"):
        if start_mode == "hot":
            prev_id = storage.previous_version_id(version_id)
            if prev_id is None:
                raise HotStartUnavailable("该版本是首版，没有上一版解可热启动")
            prev_solve = storage.latest_solve(prev_id)
            if prev_solve is None:
                raise HotStartUnavailable("上一版还没有收敛解，无法热启动")
            prev_version = storage.get_version(prev_id)
            old_mesh = discretize(prev_version["problem"])
            old_phi, old_k = prev_solve["flux"], prev_solve["k_eff"]
        phi0 = interpolate_flux(old_mesh.centers, old_phi, mesh.centers)
        # 插值后非裂变区可能为正小值，夹一下确保不出现负通量初值
        phi0 = np.maximum(phi0, 0.0)
        k0 = old_k
        actual_mode = "hot"

    outcome = source_iteration(
        mesh, tol_k=tol_k, tol_phi=tol_phi, max_iter=max_iter,
        total_fission_rate=total_fission_rate, phi0=phi0, k0=k0)

    if not outcome.converged:
        # 不留档、不返回半成品
        raise SolveFailure(outcome.message, outcome.iterations,
                           outcome.residual_k, outcome.residual_phi)

    if np.any(outcome.phi < -1e-12):
        raise SolveFailure("求解出现负通量", outcome.iterations,
                           outcome.residual_k, outcome.residual_phi)
    # 数值噪声夹零
    phi = np.maximum(outcome.phi, 0.0)

    report = reaction_rates(problem, mesh, phi, outcome.k)

    solve_id = None
    if archive:
        solve_id = storage.add_solve(
            case_id=case_id, version_id=version_id,
            started_from=actual_mode,
            total_fission_rate=total_fission_rate,
            tol_k=tol_k, tol_phi=tol_phi, max_iter=max_iter,
            k_eff=outcome.k, iterations=outcome.iterations,
            residual_k=outcome.residual_k, residual_phi=outcome.residual_phi,
            balance_error=report.balance_relative_error,
            phi=phi,
            reactions={
                "absorption_by_zone": report.absorption_by_zone,
                "fission_by_zone": report.fission_by_zone,
                "leakage_left": report.leakage_left,
                "leakage_right": report.leakage_right,
                "production_over_k": report.production_over_k,
                "total_loss": report.total_loss,
            })

    return {
        "solve_id": solve_id,
        "case_id": case_id,
        "version_id": version_id,
        "started_from": actual_mode,
        "k_eff": outcome.k,
        "iterations": outcome.iterations,
        "residual_k": outcome.residual_k,
        "residual_phi": outcome.residual_phi,
        "balance_relative_error": report.balance_relative_error,
        "total_fission_rate": total_fission_rate,
        "x": [float(v) for v in mesh.centers],
        "flux": [float(v) for v in phi],
        "zone_rates": {
            "absorption": report.absorption_by_zone,
            "fission": report.fission_by_zone,
        },
        "leakage": {"left": report.leakage_left, "right": report.leakage_right},
        "_mesh": mesh,
        "_problem": problem,
    }
