"""FastAPI 应用入口：工况 CRUD、改区出版本、求解、异步临界搜索。"""
from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api import schemas
from app.config import settings
from app.core.discretize import discretize
from app.core.models import Zone
from app.core.solver import SingularOperatorError, source_iteration
from app.core.validation import (ValidationError, validate_case_payload,
                                 validate_solve_options, validate_zone_patch)
from app.services.jobs import JobManager, apply_target
from app.services.solver_service import (HotStartUnavailable, SolveFailure,
                                         build_problem, patch_zone,
                                         solve_version)
from app.services.storage import Storage


def create_app(db_path: str | None = None) -> FastAPI:
    storage = Storage(db_path or settings.db_path)
    stale = storage.reset_stale_running_jobs()
    jobs = JobManager(storage, max_workers=settings.search_workers)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # 启动：残留 running 作业已在上面标记为 interrupted
        yield
        # 关停：通知后台作业取消并等待线程池退出，关闭数据库
        jobs.shutdown()
        storage.close()

    app = FastAPI(title="一维平板单群扩散临界后端", version="1.0.0",
                  lifespan=lifespan)
    app.state.storage = storage
    app.state.jobs = jobs
    app.state.stale_jobs_reset = stale

    # ---------------- 错误处理 ----------------

    @app.exception_handler(ValidationError)
    async def _validation_handler(_: Request, exc: ValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": "input_validation_failed",
                     "detail": [{"field": loc, "reason": msg}
                                for loc, msg in exc.errors]})

    @app.exception_handler(RequestValidationError)
    async def _pydantic_handler(_: Request, exc: RequestValidationError):
        # 把 pydantic 的类型错误也统一成 {field, reason} 结构，
        # 列表下标用 zones[0].field 风格，与物理校验的字段定位保持一致
        detail = []
        for err in exc.errors():
            parts = [p for p in err["loc"] if p != "body"]
            loc = ""
            for p in parts:
                if isinstance(p, int):
                    loc = f"{loc}[{p}]"
                else:
                    loc = f"{loc}.{p}" if loc else str(p)
            reason = f"类型或格式错误：{err['msg']}"
            detail.append({"field": loc or "body", "reason": reason})
        return JSONResponse(
            status_code=422,
            content={"error": "input_validation_failed", "detail": detail})

    class _NameConflict(Exception):
        def __init__(self, name: str):
            self.name = name

    @app.exception_handler(_NameConflict)
    async def _name_handler(_: Request, exc: _NameConflict):
        return JSONResponse(
            status_code=409,
            content={"error": "name_conflict",
                     "detail": f"工况名 {exc.name!r} 已存在"})

    @app.exception_handler(HotStartUnavailable)
    async def _hot_handler(_: Request, exc: HotStartUnavailable):
        return JSONResponse(status_code=409,
                            content={"error": "hot_start_unavailable",
                                     "detail": str(exc)})

    @app.exception_handler(SingularOperatorError)
    async def _singular_handler(_: Request, exc: SingularOperatorError):
        return JSONResponse(
            status_code=422,
            content={"error": "singular_operator",
                     "detail": f"离散算子奇异，无有限 k_eff（典型原因：两端全"
                               f"反射且系统无吸收/无泄漏）：{exc}"})

    @app.exception_handler(SolveFailure)
    async def _solve_handler(_: Request, exc: SolveFailure):
        return JSONResponse(
            status_code=409,
            content={"error": "solve_did_not_converge",
                     "detail": exc.message,
                     "iterations": exc.iterations,
                     "residual_k": exc.residual_k,
                     "residual_phi": exc.residual_phi})

    # ---------------- 工具 ----------------

    def _problem_from_case(case_id: str) -> tuple[dict, Problem]:
        case = storage.get_case(case_id)
        if case is None:
            raise KeyError(case_id)
        return case, build_problem(
            [Zone(**z) for z in case["current_version"]["params"]["zones"]],
            case["current_version"]["params"]["left_bc"],
            case["current_version"]["params"]["right_bc"])

    def _solve_tolerances(body: schemas.SolveOptions):
        return (body.tol_k if body.tol_k is not None else settings.default_tol_k,
                body.tol_phi if body.tol_phi is not None else settings.default_tol_phi,
                body.max_iter if body.max_iter is not None else settings.default_max_iter)

    # ---------------- 健康检查 ----------------

    @app.get("/health")
    async def health():
        return {"status": "ok",
                "stale_running_jobs_reset_on_boot": app.state.stale_jobs_reset}

    # ---------------- 工况 CRUD ----------------

    @app.post("/cases", status_code=201)
    async def create_case(body: schemas.CaseCreate):
        payload = body.model_dump()
        zones, left_bc, right_bc = validate_case_payload(payload)
        problem = build_problem(zones, left_bc, right_bc)
        try:
            rec = storage.create_case(body.name, problem)
        except sqlite3.IntegrityError:
            raise _NameConflict(body.name)
        return {"case_id": rec["id"], "name": body.name,
                "created_at": rec["created_at"], "version_no": 1,
                "version_id": rec["version_id"]}

    @app.get("/cases")
    async def list_cases():
        return {"cases": storage.list_cases()}

    @app.get("/cases/{case_id}")
    async def get_case(case_id: str):
        case = storage.get_case(case_id)
        if case is None:
            return JSONResponse(status_code=404,
                                content={"error": "not_found",
                                         "detail": f"工况 {case_id} 不存在"})
        case["solves"] = storage.list_solves(case_id)
        case["jobs"] = [
            {k: v for k, v in j.items()
             if k not in ("result", "error") or v is not None}
            for j in storage.list_jobs(case_id)]
        return case

    @app.delete("/cases/{case_id}", status_code=204)
    async def delete_case(case_id: str):
        if not storage.delete_case(case_id):
            return JSONResponse(status_code=404,
                                content={"error": "not_found",
                                         "detail": f"工况 {case_id} 不存在"})
        return None

    # ---------------- 改区 -> 追加不可变版本 ----------------

    @app.post("/cases/{case_id}/zones/{zone_idx}")
    async def patch_zone_endpoint(case_id: str, zone_idx: int,
                                  body: schemas.ZonePatch):
        case = storage.get_case(case_id)
        if case is None:
            return JSONResponse(status_code=404,
                                content={"error": "not_found", "detail": "工况不存在"})
        current = case["current_version"]["params"]
        problems_zones = [Zone(**z) for z in current["zones"]]
        errs, new_zone = validate_zone_patch(zone_idx, len(problems_zones),
                                             body.model_dump())
        if errs:
            raise ValidationError(errs)

        # 改完一区仍要守住全局规则（裂变材料/网格总量）
        candidate = problems_zones
        candidate[zone_idx] = new_zone
        check_payload = {"left_bc": current["left_bc"],
                         "right_bc": current["right_bc"],
                         "zones": [z.__dict__ for z in candidate]}
        validate_case_payload(check_payload)

        old_problem = build_problem(problems_zones, current["left_bc"],
                                   current["right_bc"])
        new_problem = patch_zone(old_problem, zone_idx, new_zone)
        rec = storage.add_version(case_id, new_problem)
        return {"case_id": case_id, **rec,
                "note": "已追加新版本，旧版本与历史求解结果保持不变"}

    # ---------------- 求解 ----------------

    @app.post("/cases/{case_id}/solve")
    async def solve(case_id: str, body: schemas.SolveOptions):
        case = storage.get_case(case_id)
        if case is None:
            return JSONResponse(status_code=404,
                                content={"error": "not_found", "detail": "工况不存在"})
        opt_errors = validate_solve_options(body.model_dump(exclude={"start_mode"}))
        if opt_errors:
            raise ValidationError(opt_errors)

        version_id = case["current_version"]["version_id"]
        tol_k, tol_phi, max_iter = _solve_tolerances(body)
        result = solve_version(
            storage, case_id=case_id, version_id=version_id,
            start_mode=body.start_mode,
            total_fission_rate=body.total_fission_rate,
            tol_k=tol_k, tol_phi=tol_phi, max_iter=max_iter)
        result.pop("_mesh", None)
        result.pop("_problem", None)
        return result

    @app.get("/cases/{case_id}/solves")
    async def list_solves(case_id: str):
        if storage.get_case(case_id) is None:
            return JSONResponse(status_code=404,
                                content={"error": "not_found", "detail": "工况不存在"})
        return {"solves": storage.list_solves(case_id)}

    @app.get("/solves/{solve_id}")
    async def get_solve(solve_id: int):
        rec = storage.get_solve(solve_id)
        if rec is None:
            return JSONResponse(status_code=404,
                                content={"error": "not_found", "detail": "求解记录不存在"})
        rec["flux"] = [float(v) for v in rec["flux"]]
        return rec

    # ---------------- 临界搜索（异步作业） ----------------

    @app.post("/cases/{case_id}/searches", status_code=202)
    async def create_search(case_id: str, body: schemas.SearchRequest):
        case = storage.get_case(case_id)
        if case is None:
            return JSONResponse(status_code=404,
                                content={"error": "not_found", "detail": "工况不存在"})
        params = case["current_version"]["params"]
        zones = [Zone(**z) for z in params["zones"]]
        if not (0 <= body.target_zone < len(zones)):
            raise ValidationError([(
                "target_zone",
                f"区号越界，当前工况有 {len(zones)} 个区（编号 0..{len(zones)-1}）")])
        if body.low <= 0 or body.high <= 0 or body.low >= body.high:
            raise ValidationError([(
                "bounds",
                "要求 0 < low < high（厚度与 nuSigma_f 物理上都必须为正）")])

        tol_k = body.tol_k if body.tol_k is not None else settings.default_tol_k
        tol_phi = body.tol_phi if body.tol_phi is not None else settings.default_tol_phi
        max_iter = body.max_iter if body.max_iter is not None else settings.default_max_iter
        for key, val in (("tol_k", tol_k), ("tol_phi", tol_phi)):
            if val <= 0:
                raise ValidationError([(key, "必须为正数")])
        if max_iter <= 0:
            raise ValidationError([("max_iter", "必须为正整数")])

        # 冻结提交那一刻的版本快照（作业只认这个版本）
        version_id = case["current_version"]["version_id"]
        problem = build_problem(zones, params["left_bc"], params["right_bc"])

        # 两端冷启动预检：任何一端失败都不建作业
        def _endpoint(value: float):
            probe = apply_target(problem, body.target_zone,
                                 body.target_field, value)
            return source_iteration(discretize(probe), tol_k=tol_k,
                                    tol_phi=tol_phi, max_iter=max_iter)

        out_low = _endpoint(body.low)
        if not out_low.converged:
            raise SolveFailure(f"区间下端 value={body.low} 处求解失败："
                               f"{out_low.message}", out_low.iterations,
                               out_low.residual_k, out_low.residual_phi)
        out_high = _endpoint(body.high)
        if not out_high.converged:
            raise SolveFailure(f"区间上端 value={body.high} 处求解失败："
                               f"{out_high.message}", out_high.iterations,
                               out_high.residual_k, out_high.residual_phi)

        if (out_low.k - 1.0) * (out_high.k - 1.0) > 0.0:
            return JSONResponse(
                status_code=422,
                content={"error": "bracketing_failed",
                         "detail": "区间两端 k-1 同号，区间内不能保证存在临界解，"
                                   "已拒绝该搜索",
                         "k_low": out_low.k, "k_high": out_high.k,
                         "low": body.low, "high": body.high})

        job_id = storage.new_id()
        storage.create_job({
            "id": job_id, "case_id": case_id, "version_id": version_id,
            "status": "running", "target_zone": body.target_zone,
            "target_field": body.target_field,
            "low": body.low, "high": body.high,
            "k_low": out_low.k, "k_high": out_high.k,
            "best_value": None, "best_k": None,
            "steps_done": 0, "max_steps": body.max_steps,
            "tol_k": tol_k, "tol_phi": tol_phi, "max_iter": max_iter,
            "error": None, "result": None,
            "created_at": storage.utcnow(), "updated_at": storage.utcnow()})
        jobs.submit(case_id=case_id, version_id=version_id,
                    target_zone=body.target_zone,
                    target_field=body.target_field,
                    low=body.low, high=body.high, max_steps=body.max_steps,
                    tol_k=tol_k, tol_phi=tol_phi, max_iter=max_iter,
                    endpoint_k_low=out_low, endpoint_k_high=out_high,
                    problem=problem, job_id=job_id)
        return {"job_id": job_id, "case_id": case_id, "version_id": version_id,
                "status": "running",
                "endpoint_k": {"low": out_low.k, "high": out_high.k}}

    @app.get("/searches/{job_id}")
    async def get_search(job_id: str):
        job = storage.get_job(job_id)
        if job is None:
            return JSONResponse(status_code=404,
                                content={"error": "not_found", "detail": "作业不存在"})
        if job["result"]:
            job["result"] = json.loads(job["result"])
        progress = {
            "steps_done": job["steps_done"],
            "max_steps": job["max_steps"],
            "interval": {"low": job["low"], "high": job["high"]},
            "k_at_interval": {"low": job["k_low"], "high": job["k_high"]},
            "best_value": job["best_value"],
            "best_k": job["best_k"],
        }
        job["progress"] = progress
        return job

    @app.post("/searches/{job_id}/cancel")
    async def cancel_search(job_id: str):
        job = storage.get_job(job_id)
        if job is None:
            return JSONResponse(status_code=404,
                                content={"error": "not_found", "detail": "作业不存在"})
        if job["status"] != "running":
            return {"job_id": job_id, "status": job["status"],
                    "detail": "作业已结束，取消请求不改变其状态"}
        jobs.request_cancel(job_id)
        # 等取消落地（很快，只做一次短暂让步即可）
        import time
        for _ in range(50):
            latest = storage.get_job(job_id)
            if latest["status"] != "running":
                return {"job_id": job_id, "status": latest["status"],
                        "detail": "作业已停止，未写入成功结果"}
            time.sleep(0.02)
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "status": "cancel_requested",
                     "detail": "取消信号已送达，作业将在当前步结束后停下"})

    return app


app = create_app()
