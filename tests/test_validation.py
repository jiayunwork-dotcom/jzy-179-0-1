"""非法输入逐项拒收测试。"""
from __future__ import annotations

import pytest

from app.core.validation import (MAX_TOTAL_MESH, MAX_ZONES, ValidationError,
                                 validate_case_payload, validate_zone_patch)

BASE_ZONE = {"thickness": 10.0, "d": 1.0, "sigma_a": 0.1,
             "nu_sigma_f": 0.12, "n_mesh": 10}


def payload(zones, left="zero", right="zero"):
    return {"left_bc": left, "right_bc": right, "zones": zones}


def fields_of(exc):
    return {loc for loc, _ in exc.value.errors}


def test_reject_nonpositive_thickness_and_d():
    with pytest.raises(ValidationError) as exc:
        validate_case_payload(payload([dict(BASE_ZONE, thickness=0.0)]))
    assert "zones[0].thickness" in fields_of(exc)

    with pytest.raises(ValidationError) as exc:
        validate_case_payload(payload([dict(BASE_ZONE, d=-2.0)]))
    assert "zones[0].d" in fields_of(exc)


def test_reject_negative_cross_sections():
    for field in ("sigma_a", "nu_sigma_f"):
        with pytest.raises(ValidationError) as exc:
            validate_case_payload(payload([dict(BASE_ZONE, **{field: -1e-9})]))
        assert f"zones[0].{field}" in fields_of(exc)


def test_reject_no_fissile_anywhere():
    with pytest.raises(ValidationError) as exc:
        validate_case_payload(payload(
            [dict(BASE_ZONE, nu_sigma_f=0.0),
             dict(BASE_ZONE, nu_sigma_f=0.0)]))
    assert any("裂变材料" in msg for _, msg in exc.value.errors)


def test_reject_too_many_zones():
    zones = [dict(BASE_ZONE, n_mesh=1) for _ in range(MAX_ZONES + 1)]
    with pytest.raises(ValidationError) as exc:
        validate_case_payload(payload(zones))
    assert "zones" in fields_of(exc)


def test_reject_too_many_mesh():
    with pytest.raises(ValidationError) as exc:
        validate_case_payload(payload(
            [dict(BASE_ZONE, n_mesh=MAX_TOTAL_MESH + 1)]))
    assert any("20000" in msg for _, msg in exc.value.errors)


def test_reject_bad_boundary():
    with pytest.raises(ValidationError) as exc:
        validate_case_payload({"left_bc": "vacuum", "right_bc": "zero",
                               "zones": [BASE_ZONE]})
    assert "left_bc" in fields_of(exc)


def test_multiple_field_errors_reported_together():
    bad = {"thickness": -1, "d": 0, "sigma_a": -2,
           "nu_sigma_f": -3, "n_mesh": 0}
    with pytest.raises(ValidationError) as exc:
        validate_case_payload(payload([bad]))
    assert fields_of(exc) == {
        f"zones[0].{k}" for k in
        ("thickness", "d", "sigma_a", "nu_sigma_f", "n_mesh")}


def test_valid_payload_passes():
    zones, left, right = validate_case_payload(payload([BASE_ZONE]))
    assert left == "zero" and right == "zero"
    assert zones[0].n_mesh == 10


def test_zone_patch_out_of_range():
    errs, zone = validate_zone_patch(3, 2, BASE_ZONE)
    assert zone is None
    assert errs[0][0] == "zones[3]"
    assert "越界" in errs[0][1]
    # 正常区内的字段错误仍逐字段定位
    errs2, zone2 = validate_zone_patch(0, 2, dict(BASE_ZONE, d=-1))
    assert zone2 is None
    assert {loc for loc, _ in errs2} == {"zones[0].d"}
