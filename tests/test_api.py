"""HTTP 层：工况 CRUD、版本留档、冷热启动一致、持久化、失败显式报错。"""
from __future__ import annotations

import math

from fastapi.testclient import TestClient


FUEL = {"thickness": 25.0, "d": 1.0, "sigma_a": 0.1,
        "nu_sigma_f": 0.12, "n_mesh": 50}
REFLECTOR = {"thickness": 12.0, "d": 1.0, "sigma_a": 0.0,
             "nu_sigma_f": 0.0, "n_mesh": 24}


def create_case(client: TestClient, name="demo", zones=None,
                left="zero", right="zero"):
    resp = client.post("/cases", json={
        "name": name, "left_bc": left, "right_bc": right,
        "zones": zones or [FUEL]})
    assert resp.status_code == 201, resp.text
    return resp.json()


def solve(client, case_id, mode="cold", **extra):
    body = {"start_mode": mode, "tol_k": 1e-10, "tol_phi": 1e-10}
    body.update(extra)
    resp = client.post(f"/cases/{case_id}/solve", json=body)
    return resp


# ---------------- 工况 CRUD 与持久化 ----------------

def test_case_crud_lifecycle(client):
    rec = create_case(client, "lifecase")
    cid = rec["case_id"]
    assert rec["version_no"] == 1

    listed = client.get("/cases").json()["cases"]
    assert any(c["id"] == cid for c in listed)

    got = client.get(f"/cases/{cid}")
    assert got.status_code == 200
    assert got.json()["current_version"]["version_no"] == 1

    assert client.delete(f"/cases/{cid}").status_code == 204
    assert client.get(f"/cases/{cid}").status_code == 404
    # 再删一次
    assert client.delete(f"/cases/{cid}").status_code == 404


def test_case_persistence_across_restart(tmp_path):
    db = tmp_path / "persist.db"
    from app.main import create_app
    app1 = create_app(db_path=str(db))
    with TestClient(app1) as c1:
        rec = create_case(c1, "persisted", zones=[FUEL, REFLECTOR])
        r = solve(c1, rec["case_id"])
        assert r.status_code == 200
        k1 = r.json()["k_eff"]

    app2 = create_app(db_path=str(tmp_path / "persist.db"))
    with TestClient(app2) as c2:
        got = c2.get(f"/cases/{rec['case_id']}").json()
        assert got["name"] == "persisted"
        assert len(got["versions"]) == 1
        solves = c2.get(f"/cases/{rec['case_id']}/solves").json()["solves"]
        assert len(solves) == 1
        assert math.isclose(solves[0]["k_eff"], k1, rel_tol=1e-12)
        detail = c2.get(f"/solves/{solves[0]['id']}").json()
        assert len(detail["flux"]) == 74


# ---------------- 修改一区 -> 新版本，旧结果不覆盖 ----------------

def test_patch_zone_creates_version_and_keeps_history(client):
    rec = create_case(client, "ver", zones=[FUEL, REFLECTOR])
    cid = rec["case_id"]
    r1 = solve(client, cid).json()

    # 加厚燃料区
    resp = client.post(f"/cases/{cid}/zones/0",
                       json={**FUEL, "thickness": 35.0})
    assert resp.status_code == 200, resp.text
    v2 = resp.json()
    assert v2["version_no"] == 2

    case = client.get(f"/cases/{cid}").json()
    assert case["current_version"]["params"]["zones"][0]["thickness"] == 35.0
    # 旧版本参数没动
    assert case["versions"][0]["params"]["zones"][0]["thickness"] == 25.0

    r2 = solve(client, cid).json()
    assert r2["version_id"] == v2["version_id"]
    assert r2["k_eff"] > r1["k_eff"]  # 燃料加厚

    solves = client.get(f"/cases/{cid}/solves").json()["solves"]
    versions = {s["version_id"] for s in solves}
    assert versions == {r1["version_id"], r2["version_id"]}
    # 旧版结果原样留档
    old = client.get(f"/solves/{r1['solve_id']}").json()
    assert math.isclose(old["k_eff"], r1["k_eff"], rel_tol=1e-14)


def test_patch_validation_error_is_field_specific(client):
    rec = create_case(client, "badpatch")
    cid = rec["case_id"]
    resp = client.post(f"/cases/{cid}/zones/0",
                       json={**FUEL, "d": -1.0, "sigma_a": -2.0})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    fields = {d["field"] for d in detail}
    assert any("d" in f for f in fields)
    # 仍然只有一个版本
    case = client.get(f"/cases/{cid}").json()
    assert len(case["versions"]) == 1


# ---------------- 冷/热启动一致（核心要求） ----------------

def test_cold_hot_consistency_after_thickness_change(client):
    rec = create_case(client, "hot", zones=[FUEL, REFLECTOR])
    cid = rec["case_id"]
    cold0 = solve(client, cid, "cold")
    assert cold0.status_code == 200

    client.post(f"/cases/{cid}/zones/0",
               json={**FUEL, "thickness": 30.0, "n_mesh": 60})

    cold = solve(client, cid, "cold").json()
    hot = solve(client, cid, "hot").json()

    tol = 1e-10
    # k 差落在收敛容差量级
    assert abs(cold["k_eff"] - hot["k_eff"]) < max(tol, 5e-10)
    # 通量分布一致（同网格，逐点比较）
    assert len(cold["flux"]) == len(hot["flux"])
    fc = _normalized(cold["flux"])
    fh = _normalized(hot["flux"])
    max_rel = max(abs(a - b) / max(a, 1e-30) for a, b in zip(fc, fh))
    assert max_rel < 1e-8, max_rel
    # 厚度大改动下功率迭代受主导比限制，热启动不强求省步（见 README 取舍说明），
    # 但热启动至少不能比冷启动多花离谱的步数
    assert hot["iterations"] <= cold["iterations"] + 5
    assert hot["started_from"] == "hot" and cold["started_from"] == "cold"
    # 两条路径都过中子平衡
    assert cold["balance_relative_error"] < 1e-8
    assert hot["balance_relative_error"] < 1e-8


def test_cold_hot_consistency_after_nuf_change_and_reaction_rates(client):
    rec = create_case(client, "hot2", zones=[
        {**FUEL, "thickness": 30.0, "n_mesh": 60}, REFLECTOR])
    cid = rec["case_id"]
    solve(client, cid, "cold")
    client.post(f"/cases/{cid}/zones/0",
               json={**FUEL, "thickness": 30.0, "n_mesh": 60,
                     "nu_sigma_f": 0.125})
    cold = solve(client, cid, "cold").json()
    hot = solve(client, cid, "hot").json()
    assert abs(cold["k_eff"] - hot["k_eff"]) < 1e-9
    # 截面类小改动：热启动应大幅省步（旧解形状几乎可直接沿用）
    assert hot["iterations"] < 0.25 * cold["iterations"], (
        cold["iterations"], hot["iterations"])
    for res in (cold, hot):
        za = sum(res["zone_rates"]["absorption"])
        zf = sum(res["zone_rates"]["fission"])
        loss = za + res["leakage"]["left"] + res["leakage"]["right"]
        assert abs(zf / res["k_eff"] - loss) / (zf / res["k_eff"]) < 1e-8


def test_hot_start_unavailable_on_first_version(client):
    rec = create_case(client, "nohot")
    resp = solve(client, rec["case_id"], "hot")
    assert resp.status_code == 409
    assert resp.json()["error"] == "hot_start_unavailable"


def test_flux_normalization_to_total_fission_rate(client):
    rec = create_case(client, "norm", zones=[FUEL, REFLECTOR])
    cid = rec["case_id"]
    r = solve(client, cid, "cold", total_fission_rate=5.0).json()
    fission_total = sum(r["zone_rates"]["fission"])
    assert abs(fission_total - 5.0) < 1e-8
    assert all(v >= 0.0 for v in r["flux"])


# ---------------- 求解失败显式报错（不返回半成品） ----------------

def test_solve_failure_reports_residual_and_iterations(client):
    # 接近临界极限的弱源系统 + 极小步数上限，逼它到顶
    weak = {"thickness": 5.0, "d": 5.0, "sigma_a": 0.5,
            "nu_sigma_f": 0.5001, "n_mesh": 10}
    rec = create_case(client, "hard", zones=[weak])
    resp = solve(client, rec["case_id"], "cold", max_iter=3)
    assert resp.status_code == 409
    body = resp.json()
    assert body["error"] == "solve_did_not_converge"
    assert body["iterations"] == 3
    assert "residual_k" in body and "residual_phi" in body
    # 失败不留档
    assert client.get(f"/cases/{rec['case_id']}/solves").json()["solves"] == []


def _normalized(vec):
    m = max(vec)
    return [v / m for v in vec]


def test_http_validation_errors_are_field_specific(client):
    # 坏边界 + 多个坏字段，错误结构统一为 {field, reason}
    resp = client.post("/cases", json={
        "name": "x", "left_bc": "vacuum", "right_bc": "zero",
        "zones": [{"thickness": 0, "d": -1, "sigma_a": -2,
                   "nu_sigma_f": -3, "n_mesh": 0}]})
    assert resp.status_code == 422
    fields = {d["field"] for d in resp.json()["detail"]}
    assert "left_bc" in fields
    for f in ("zones[0].thickness", "zones[0].d", "zones[0].sigma_a",
              "zones[0].nu_sigma_f", "zones[0].n_mesh"):
        assert f in fields, (f, fields)


def test_http_type_error_uses_unified_shape(client):
    resp = client.post("/cases", json={
        "name": "x", "left_bc": "zero", "right_bc": "zero",
        "zones": [{"thickness": 10, "d": 1, "sigma_a": 0.1,
                   "nu_sigma_f": 0.12, "n_mesh": "ten"}]})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "input_validation_failed"
    assert any("n_mesh" in d["field"] for d in body["detail"])


def test_duplicate_name_conflict(client):
    create_case(client, "dup")
    resp = client.post("/cases", json={
        "name": "dup", "left_bc": "zero", "right_bc": "zero",
        "zones": [FUEL]})
    assert resp.status_code == 409
    assert resp.json()["error"] == "name_conflict"


def test_missing_resources_return_404(client):
    assert client.get("/cases/nope").status_code == 404
    assert client.get("/solves/9999").status_code == 404
    assert client.get("/searches/nope").status_code == 404


def test_search_bad_zone_and_bad_bounds(client):
    rec = create_case(client, "bounds")
    cid = rec["case_id"]
    r1 = client.post(f"/cases/{cid}/searches", json={
        "target_zone": 5, "target_field": "thickness",
        "low": 1, "high": 2, "max_steps": 10})
    assert r1.status_code == 422
    assert any(d["field"] == "target_zone" for d in r1.json()["detail"])

    r2 = client.post(f"/cases/{cid}/searches", json={
        "target_zone": 0, "target_field": "thickness",
        "low": 10, "high": 5, "max_steps": 10})
    assert r2.status_code == 422
    assert any(d["field"] == "bounds" for d in r2.json()["detail"])
