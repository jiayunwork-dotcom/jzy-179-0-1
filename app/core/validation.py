"""工况参数校验。非法输入按字段名逐项给出原因。"""
from __future__ import annotations

from typing import Any

from .models import BOUNDARY_TYPES, Zone

MAX_ZONES = 50
MAX_TOTAL_MESH = 20000


class ValidationError(ValueError):
    """errors: [(字段定位, 原因), ...]，一条请求可以同时报多个字段。"""

    def __init__(self, errors: list[tuple[str, str]]):
        self.errors = errors
        super().__init__("; ".join(f"{loc}: {msg}" for loc, msg in errors))


def _check_zone(idx: int, raw: Any) -> tuple[list[tuple[str, str]], Zone | None]:
    """校验单区输入；数值字段既检查类型又检查物理范围。"""
    errors: list[tuple[str, str]] = []
    loc = f"zones[{idx}]"

    if not isinstance(raw, dict):
        return [(f"zones[{idx}]", "必须是一个区参数对象")], None

    thickness = raw.get("thickness")
    d = raw.get("d")
    sigma_a = raw.get("sigma_a")
    nu_sigma_f = raw.get("nu_sigma_f")
    n_mesh = raw.get("n_mesh")

    if not isinstance(thickness, (int, float)) or isinstance(thickness, bool):
        errors.append((f"{loc}.thickness", "必须是数值"))
    elif not thickness > 0:  # 同时挡掉 NaN
        errors.append((f"{loc}.thickness", "厚度必须大于 0"))

    if not isinstance(d, (int, float)) or isinstance(d, bool):
        errors.append((f"{loc}.d", "必须是数值"))
    elif not d > 0:
        errors.append((f"{loc}.d", "扩散系数 D 必须大于 0"))

    if not isinstance(sigma_a, (int, float)) or isinstance(sigma_a, bool):
        errors.append((f"{loc}.sigma_a", "必须是数值"))
    elif sigma_a < 0:
        errors.append((f"{loc}.sigma_a", "Sigma_a 不允许为负"))

    if not isinstance(nu_sigma_f, (int, float)) or isinstance(nu_sigma_f, bool):
        errors.append((f"{loc}.nu_sigma_f", "必须是数值"))
    elif nu_sigma_f < 0:
        errors.append((f"{loc}.nu_sigma_f", "nuSigma_f 不允许为负"))

    if not isinstance(n_mesh, int) or isinstance(n_mesh, bool):
        errors.append((f"{loc}.n_mesh", "网格数必须是正整数"))
    elif n_mesh <= 0:
        errors.append((f"{loc}.n_mesh", "网格数必须大于 0"))

    if errors:
        return errors, None
    return [], Zone(float(thickness), float(d), float(sigma_a),
                    float(nu_sigma_f), int(n_mesh))


def validate_case_payload(payload: Any) -> tuple[list[Zone], str, str]:
    """校验整份工况创建/修改输入。"""
    errors: list[tuple[str, str]] = []

    if not isinstance(payload, dict):
        raise ValidationError([("body", "请求体必须是 JSON 对象")])

    left_bc = payload.get("left_bc", payload.get("leftBc"))
    right_bc = payload.get("right_bc", payload.get("rightBc"))
    raw_zones = payload.get("zones")

    if left_bc not in BOUNDARY_TYPES:
        errors.append(("left_bc", f"必须是 {list(BOUNDARY_TYPES)} 之一"))
    if right_bc not in BOUNDARY_TYPES:
        errors.append(("right_bc", f"必须是 {list(BOUNDARY_TYPES)} 之一"))

    zones: list[Zone] = []
    if not isinstance(raw_zones, list) or len(raw_zones) == 0:
        errors.append(("zones", "至少需要一个区"))
    else:
        if len(raw_zones) > MAX_ZONES:
            errors.append(("zones", f"区数不得超过 {MAX_ZONES}，收到 {len(raw_zones)}"))
        for i, raw in enumerate(raw_zones):
            z_errors, zone = _check_zone(i, raw)
            errors.extend(z_errors)
            if zone is not None:
                zones.append(zone)
        if zones and sum(z.n_mesh for z in zones) > MAX_TOTAL_MESH:
            errors.append(("zones",
                           f"各区网格总数不得超过 {MAX_TOTAL_MESH}，"
                           f"收到 {sum(z.n_mesh for z in zones)}"))
        if zones and all(z.nu_sigma_f <= 0.0 for z in zones):
            errors.append(("zones", "所有区都不含裂变材料（nuSigma_f 全为 0），"
                                    "不存在正的 k_eff 特征值"))

    if errors:
        raise ValidationError(errors)
    return zones, left_bc, right_bc


def validate_zone_patch(idx: int, n_zones: int, patch: Any
                        ) -> tuple[list[tuple[str, str]], Zone | None]:
    """校验“只改某一区”的局部输入，返回 (错误[(字段,原因)], 新区或None)。"""
    if not isinstance(patch, dict):
        return [("body", "必须是 JSON 对象")], None
    if not (0 <= idx < n_zones):
        return [(f"zones[{idx}]", f"区号越界，当前只有 {n_zones} 个区")], None
    return _check_zone(idx, patch)


def validate_solve_options(options: Any) -> list[tuple[str, str]]:
    """求解参数（容差/上限/归一化总裂变率）的校验。

    None 表示“用服务默认值”，不算非法；只校验用户实际给了值的字段。
    """
    errors: list[tuple[str, str]] = []
    if options is None:
        return errors
    if not isinstance(options, dict):
        return [("solve", "求解参数必须是对象")]
    for key in ("tol_k", "tol_phi", "total_fission_rate"):
        if key in options and options[key] is not None:
            val = options[key]
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not val > 0:
                errors.append((f"solve.{key}", "必须是正数"))
    if "max_iter" in options and options["max_iter"] is not None:
        val = options["max_iter"]
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            errors.append(("solve.max_iter", "必须是正整数"))
    return errors
