"""运行期配置：数据库位置、求解器默认参数都可由环境变量覆盖。"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    db_path: str = os.getenv("DIFFUSION_DB", "/data/diffusion.db")
    # 源迭代默认收敛容差与步数上限
    default_tol_k: float = float(os.getenv("DIFFUSION_TOL_K", "1e-10"))
    default_tol_phi: float = float(os.getenv("DIFFUSION_TOL_PHI", "1e-10"))
    default_max_iter: int = int(os.getenv("DIFFUSION_MAX_ITER", "20000"))
    # 临界搜索默认上限（二分步数）
    default_search_max_steps: int = int(os.getenv("DIFFUSION_SEARCH_STEPS", "60"))
    search_workers: int = int(os.getenv("DIFFUSION_WORKERS", "2"))


settings = Settings()
