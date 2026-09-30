"""异步临界搜索作业测试：成功、同号拒绝、取消、步数耗尽、参数版本锁定。"""
from __future__ import annotations

import time

from fastapi.testclient import TestClient


# 一个对厚度变化比较敏感的燃料板：薄板次临界、厚板超临界
FUEL = {"thickness": 10.0, "d": 1.0, "sigma_a": 0.15,
        "nu_sigma_f": 0.18, "n_mesh": 40}


def create(client, name="search-case", **zone_over):
    z = {**FUEL, **zone_over}
    resp = client.post("/cases", json={
        "name": name, "left_bc": "zero", "right_bc": "zero", "zones": [z]})
    assert resp.status_code == 201, resp.text
    return resp.json()


def wait_status(client, job_id, targets, timeout=30.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/searches/{job_id}").json()
        if last["status"] in targets:
            return last
        time.sleep(0.02)
    raise AssertionError(f"作业未在 {timeout}s 内到达 {targets}，最后状态 {last}")


def test_search_thickness_finds_critical_and_verifies(client):
    rec = create(client)
    resp = client.post(f"/cases/{rec['case_id']}/searches", json={
        "target_zone": 0, "target_field": "thickness",
        "low": 8.0, "high": 40.0, "max_steps": 45})
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    assert resp.json()["version_id"] == rec["version_id"]

    job = wait_status(client, job_id, {"succeeded", "failed"})
    assert job["status"] == "succeeded", job
    result = job["result"]
    assert abs(result["k_eff"] - 1.0) < 1e-6
    assert result["steps"] >= 1

    # 代回去独立冷启动复核 |k-1| < 1e-6（用提交版本自己的网格）
    from app.core.discretize import discretize
    from app.core.models import Problem, Zone
    from app.core.solver import source_iteration
    z = Zone(result["value"], FUEL["d"], FUEL["sigma_a"],
             FUEL["nu_sigma_f"], FUEL["n_mesh"])
    p = Problem((z,), "zero", "zero")
    check = source_iteration(discretize(p), tol_k=1e-11, tol_phi=1e-11,
                             max_iter=100000)
    assert check.converged
    assert abs(check.k - 1.0) < 1e-6


def test_search_nu_sigma_f(client):
    # 厚度固定，搜 nuSigma_f：取一个肯定次临界的厚度
    rec = create(client, "nuf-search", thickness=8.0, n_mesh=40)
    resp = client.post(f"/cases/{rec['case_id']}/searches", json={
        "target_zone": 0, "target_field": "nu_sigma_f",
        "low": 0.15, "high": 0.4, "max_steps": 45})
    assert resp.status_code == 202, resp.text
    job = wait_status(client, resp.json()["job_id"],
                      {"succeeded", "failed"})
    assert job["status"] == "succeeded", job
    assert abs(job["result"]["k_eff"] - 1.0) < 1e-6


def test_same_sign_endpoints_rejected_with_ks(client):
    # 很薄且 k_inf 低 -> 两端都次临界
    rec = create(client, "same-sign", thickness=5.0, nu_sigma_f=0.155,
                 n_mesh=20)
    resp = client.post(f"/cases/{rec['case_id']}/searches", json={
        "target_zone": 0, "target_field": "thickness",
        "low": 4.0, "high": 4.5, "max_steps": 10})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "bracketing_failed"
    assert body["k_low"] < 1.0 and body["k_high"] < 1.0


def test_search_exhausts_steps(client):
    # 区间极大、只给 2 步 -> 无法达到 1e-6
    rec = create(client, "exhaust", thickness=12.0, n_mesh=30)
    resp = client.post(f"/cases/{rec['case_id']}/searches", json={
        "target_zone": 0, "target_field": "thickness",
        "low": 8.0, "high": 60.0, "max_steps": 2})
    job_id = resp.json()["job_id"]
    job = wait_status(client, job_id, {"failed", "succeeded"})
    assert job["status"] == "failed"
    assert "未满足" in job["error"] or "二分" in job["error"]
    assert job["steps_done"] == 2
    # 回报最后区间与残差
    assert job["low"] >= 8.0 and job["high"] <= 60.0
    assert job["k_low"] is not None and job["k_high"] is not None


def test_cancel_search_actually_stops(client, tmp_path):
    # 大网格让每一步慢一些，给取消留出窗口
    fuel = {**FUEL, "thickness": 20.0, "n_mesh": 400}
    rec = create(client, "cancel", **fuel)
    resp = client.post(f"/cases/{rec['case_id']}/searches", json={
        "target_zone": 0, "target_field": "thickness",
        "low": 10.0, "high": 120.0, "max_steps": 200,
        "tol_k": 1e-12, "tol_phi": 1e-12})
    job_id = resp.json()["job_id"]
    # 等它确实跑起来
    time.sleep(0.3)
    cancel = client.post(f"/searches/{job_id}/cancel")
    assert cancel.status_code in (200, 202)
    job = wait_status(client, job_id, {"canceled", "succeeded", "failed"},
                      timeout=15)
    assert job["status"] == "canceled", job
    assert job["result"] is None
    # 再等一段时间，确认不会“死后复活”写入成功结果
    time.sleep(0.5)
    again = client.get(f"/searches/{job_id}").json()
    assert again["status"] == "canceled"


def test_cancel_finished_job_is_noop(client):
    rec = create(client, "cancel-done")
    job_id = client.post(f"/cases/{rec['case_id']}/searches", json={
        "target_zone": 0, "target_field": "thickness",
        "low": 8.0, "high": 40.0, "max_steps": 40}).json()["job_id"]
    job = wait_status(client, job_id, {"succeeded"})
    resp = client.post(f"/searches/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "succeeded"


def test_two_concurrent_jobs_and_version_lock(client):
    """同一工况挂两个搜索；期间工况改版，两个作业都只认提交时的版本。"""
    rec = create(client, "locked", thickness=12.0, n_mesh=60)
    cid = rec["case_id"]
    version1 = rec["version_id"]

    j1 = client.post(f"/cases/{cid}/searches", json={
        "target_zone": 0, "target_field": "thickness",
        "low": 8.0, "high": 80.0, "max_steps": 60}).json()["job_id"]
    j2 = client.post(f"/cases/{cid}/searches", json={
        "target_zone": 0, "target_field": "thickness",
        "low": 9.0, "high": 70.0, "max_steps": 60}).json()["job_id"]

    # 作业在跑时改工况（新版本）
    patched = client.post(f"/cases/{cid}/zones/0",
                          json={**FUEL, "thickness": 12.0, "n_mesh": 60,
                                "d": 1.5, "sigma_a": 0.12,
                                "nu_sigma_f": 0.13})
    assert patched.status_code == 200
    version2 = patched.json()["version_id"]
    assert version2 != version1

    done1 = wait_status(client, j1, {"succeeded", "failed"})
    done2 = wait_status(client, j2, {"succeeded", "failed"})
    assert done1["status"] == "succeeded", done1
    assert done2["status"] == "succeeded", done2
    # 版本锁定：结果属于 v1，而不是改版后的 v2
    assert done1["version_id"] == version1
    assert done2["version_id"] == version1

    # 关键校验：用提交版参数代回两个作业的解，都满足 |k-1|<1e-6；
    # 若作业误用了 v2 参数（D、截面不同），代回 v1 就不会临界
    from app.core.discretize import discretize
    from app.core.models import Problem, Zone
    from app.core.solver import source_iteration
    for job in (done1, done2):
        v = job["result"]["value"]
        z = Zone(v, FUEL["d"], FUEL["sigma_a"], FUEL["nu_sigma_f"], 60)
        out = source_iteration(discretize(Problem((z,), "zero", "zero")),
                               tol_k=1e-11, tol_phi=1e-11,
                               max_iter=100000)
        assert abs(out.k - 1.0) < 1e-6, (job["id"], v, out.k)


def test_search_progress_fields(client):
    rec = create(client, "progress", thickness=12.0, n_mesh=40)
    job_id = client.post(f"/cases/{rec['case_id']}/searches", json={
        "target_zone": 0, "target_field": "thickness",
        "low": 8.0, "high": 80.0, "max_steps": 60}).json()["job_id"]
    # 立即查询：应能拿到区间与端点 k
    info = client.get(f"/searches/{job_id}").json()
    assert info["progress"]["interval"] == {"low": 8.0, "high": 80.0}
    assert info["progress"]["k_at_interval"]["low"] is not None
    wait_status(client, job_id, {"succeeded"})


def test_stale_running_jobs_marked_interrupted_on_restart(tmp_path):
    db = tmp_path / "stale.db"
    from app.main import create_app
    from app.services.storage import Storage
    st = Storage(str(db))
    case = st.create_case("stale-case", _problem())
    st.create_job({
        "id": "j1", "case_id": case["id"], "version_id": case["version_id"],
        "status": "running", "target_zone": 0, "target_field": "thickness",
        "low": 8.0, "high": 80.0, "k_low": 0.9, "k_high": 1.1,
        "best_value": None, "best_k": None, "steps_done": 3,
        "max_steps": 60, "tol_k": 1e-10, "tol_phi": 1e-10,
        "max_iter": 10000, "error": None, "result": None,
        "created_at": st.utcnow(), "updated_at": st.utcnow()})
    st.close()

    from fastapi.testclient import TestClient
    app = create_app(db_path=str(db))
    with TestClient(app) as c:
        assert c.get("/health").json()["stale_running_jobs_reset_on_boot"] == 1
        job = c.get("/searches/j1").json()
        assert job["status"] == "interrupted"


def _problem():
    from app.core.models import Problem, Zone
    return Problem((Zone(12.0, 1.0, 0.15, 0.18, 40),), "zero", "zero")
