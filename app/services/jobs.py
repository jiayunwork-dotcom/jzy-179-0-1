"""异步临界搜索：二分法找使 k_eff=1 的某区厚度或 nuSigma_f。

设计要点：
* 提交即返回作业号，作业在线程池中后台跑；
* 提交时把 version_id 连同当时完整参数快照冻结下来，作业全程只认这个版本，
  之后工况改版、并发挂多个作业都不会张冠李戴；
* 区间两端先各冷启动求解一次，k-1 同号则直接失败并回报两端 k；
* 二分过程中每个中点沿用前一步的收敛解热启动以省时；
* 每一步检查取消标志，取消后真的停下，不再写入任何成功结果；
* 找到的解必须 |k-1| < 1e-6；步数用尽则以未收敛结束并回报最终区间与残差。
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import numpy as np

from app.core.discretize import discretize
from app.core.models import Problem
from app.core.reactions import interpolate_flux
from app.core.solver import SolveOutcome, source_iteration
from app.services.storage import Storage, utcnow

KEFF_TARGET_TOL = 1e-6


class SameSignError(ValueError):
    def __init__(self, k_low: float, k_high: float):
        self.k_low = k_low
        self.k_high = k_high
        super().__init__(
            f"区间两端 k-1 同号（k_low={k_low:.8f}, k_high={k_high:.8f}），"
            "区间内不能保证存在临界解，拒绝搜索")


class JobCancelled(Exception):
    pass


def apply_target(problem: Problem, target_zone: int, target_field: str,
                 value: float) -> Problem:
    """返回把指定区的厚度/nuSigma_f 替换为 value 的新问题（版本快照不变）。"""
    zones = list(problem.zones)
    z = zones[target_zone]
    if target_field == "thickness":
        zones[target_zone] = replace(z, thickness=float(value))
    else:
        zones[target_zone] = replace(z, nu_sigma_f=float(value))
    return replace(problem, zones=tuple(zones))


def _evaluate(problem: Problem, value: float, *, target_zone: int,
              target_field: str, tol_k: float, tol_phi: float,
              max_iter: int, hot: tuple | None,
              cancel_event: threading.Event
              ) -> tuple[SolveOutcome, Any]:
    """在待定量=value 处冷（或热）启动求解一次。"""
    if cancel_event.is_set():
        raise JobCancelled()
    mesh = discretize(apply_target(problem, target_zone, target_field, value))

    phi0 = k0 = None
    if hot is not None:
        old_centers, old_phi, old_k = hot
        phi0 = np.maximum(interpolate_flux(old_centers, old_phi,
                                           mesh.centers), 0.0)
        k0 = old_k

    return source_iteration(
        mesh, tol_k=tol_k, tol_phi=tol_phi, max_iter=max_iter,
        phi0=phi0, k0=k0), mesh


class JobManager:
    def __init__(self, storage: Storage, max_workers: int = 2):
        self.storage = storage
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="crit-search")
        self._cancel_events: dict[str, threading.Event] = {}
        self._events_lock = threading.Lock()

    def shutdown(self) -> None:
        with self._events_lock:
            for ev in self._cancel_events.values():
                ev.set()
        self._pool.shutdown(wait=False, cancel_futures=True)

    def request_cancel(self, job_id: str) -> bool:
        """请求取消。返回 False 表示作业不存在或已结束。"""
        with self._events_lock:
            ev = self._cancel_events.get(job_id)
        if ev is None:
            return False
        ev.set()
        return True

    def submit(self, *, case_id: str, version_id: int, target_zone: int,
               target_field: str, low: float, high: float,
               max_steps: int, tol_k: float, tol_phi: float,
               max_iter: int,
               endpoint_k_low: SolveOutcome, endpoint_k_high: SolveOutcome,
               problem: Problem, job_id: str) -> None:
        """作业已在 API 层做完两端预检并落库为 running，这里只负责后台跑。"""
        event = threading.Event()
        with self._events_lock:
            self._cancel_events[job_id] = event
        self._pool.submit(
            self._run, job_id, case_id, version_id, target_zone,
            target_field, low, high, max_steps, tol_k, tol_phi, max_iter,
            endpoint_k_low, endpoint_k_high, problem, event)

    # ------------------------------------------------------------------

    def _run(self, job_id: str, case_id: str, version_id: int,
             target_zone: int, target_field: str, low: float, high: float,
             max_steps: int, tol_k: float, tol_phi: float, max_iter: int,
             ep_low: SolveOutcome, ep_high: SolveOutcome,
             problem: Problem, cancel_event: threading.Event) -> None:
        try:
            k_low, k_high = ep_low.k, ep_high.k
            self.storage.update_job(
                job_id, status="running", low=low, high=high,
                k_low=k_low, k_high=k_high, steps_done=0)

            # 端点本身恰好临界
            for value, kval, outcome in ((low, k_low, ep_low),
                                         (high, k_high, ep_high)):
                if abs(kval - 1.0) < KEFF_TARGET_TOL:
                    self._succeed(job_id, value, kval, 0, outcome)
                    return

            if (k_low - 1.0) * (k_high - 1.0) > 0:
                # 理论上提交层已挡，这里是防线
                self._fail(job_id,
                           f"区间两端 k-1 同号：k_low={k_low:.10g},"
                           f" k_high={k_high:.10g}",
                           low, high, k_low, k_high, 0)
                return

            # 最近一次收敛解作为下一个中点的热启动来源
            hot_low = self._hot_tuple(problem, target_zone, target_field,
                                      low, ep_low)
            hot_high = self._hot_tuple(problem, target_zone, target_field,
                                       high, ep_high)

            outcome = mesh = None
            for step in range(1, max_steps + 1):
                if cancel_event.is_set():
                    self._canceled(job_id, low, high, k_low, k_high,
                                   step - 1)
                    return

                mid = 0.5 * (low + high)
                # 选离目标更近的一端的解做热启动
                hot = hot_low if abs(k_low - 1.0) <= abs(k_high - 1.0) \
                    else hot_high
                outcome, mesh = _evaluate(
                    problem, mid, target_zone=target_zone,
                    target_field=target_field, tol_k=tol_k, tol_phi=tol_phi,
                    max_iter=max_iter, hot=hot, cancel_event=cancel_event)

                if not outcome.converged:
                    self._fail(job_id,
                               f"第 {step} 步中点 value={mid:.10g} 处求解失败："
                               f"{outcome.message}（残差 k={outcome.residual_k:.3e},"
                               f" phi={outcome.residual_phi:.3e},"
                               f" 迭代 {outcome.iterations} 次）",
                               low, high, k_low, k_high, step)
                    return

                k_mid = outcome.k
                if abs(k_mid - 1.0) < KEFF_TARGET_TOL:
                    self._succeed(job_id, mid, k_mid, step, outcome)
                    return

                # 缩区间：保持两端异号
                if (k_low - 1.0) * (k_mid - 1.0) <= 0:
                    high, k_high, hot_high = mid, k_mid, self._hot_tuple(
                        problem, target_zone, target_field, mid, outcome)
                else:
                    low, k_low, hot_low = mid, k_mid, self._hot_tuple(
                        problem, target_zone, target_field, mid, outcome)

                self.storage.update_job(
                    job_id, low=low, high=high, k_low=k_low,
                    k_high=k_high, best_value=0.5 * (low + high),
                    best_k=None, steps_done=step)

            # 步数用尽：回报最后区间与最佳点残差
            mid = 0.5 * (low + high)
            self._fail(
                job_id,
                f"二分 {max_steps} 步后仍未满足 |k-1|<{KEFF_TARGET_TOL:g}；"
                f"最终区间 [{low:.10g}, {high:.10g}]，"
                f"k_low={k_low:.10g}, k_high={k_high:.10g}，"
                f"k-1 最小绝对值="
                f"{min(abs(k_low-1), abs(k_high-1)):.3e}",
                low, high, k_low, k_high, max_steps, best_value=mid)

        except JobCancelled:
            job = self.storage.get_job(job_id)
            if job and job["status"] == "running":
                self.storage.update_job(job_id, status="canceled",
                                        error="作业已被用户取消")
        except Exception as exc:  # 后台线程不能静默吞异常
            self.storage.update_job(job_id, status="failed",
                                    error=f"作业内部错误：{exc!r}")
        finally:
            with self._events_lock:
                self._cancel_events.pop(job_id, None)

    def _hot_tuple(self, problem: Problem, target_zone: int,
                   target_field: str, value: float,
                   outcome: SolveOutcome):
        """返回 (该待定量对应网格中心, phi, k) 供下一点插值热启动。"""
        mesh = discretize(apply_target(problem, target_zone,
                                       target_field, value))
        return mesh.centers, np.maximum(outcome.phi, 0.0), outcome.k

    def _succeed(self, job_id: str, value: float, k_eff: float,
                 steps: int, outcome: SolveOutcome) -> None:
        result = {
            "value": float(value),
            "k_eff": float(k_eff),
            "steps": steps,
            "iterations_at_solution": outcome.iterations,
            "residual_k": outcome.residual_k,
            "residual_phi": outcome.residual_phi,
            "found_at": utcnow(),
        }
        self.storage.update_job(
            job_id, status="succeeded", steps_done=steps,
            best_value=float(value), best_k=float(k_eff),
            result=json.dumps(result, ensure_ascii=False), error=None)

    def _fail(self, job_id: str, message: str, low: float, high: float,
              k_low: float, k_high: float, steps: int,
              best_value: float | None = None) -> None:
        self.storage.update_job(
            job_id, status="failed", error=message, low=low, high=high,
            k_low=k_low, k_high=k_high, steps_done=steps,
            best_value=best_value if best_value is not None
            else 0.5 * (low + high))

    def _canceled(self, job_id: str, low: float, high: float,
                  k_low: float, k_high: float, steps: int) -> None:
        self.storage.update_job(
            job_id, status="canceled", error="作业已被用户取消，停止计算",
            low=low, high=high, k_low=k_low, k_high=k_high,
            steps_done=steps)
