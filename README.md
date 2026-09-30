# 一维平板单群扩散临界后端

常驻 FastAPI 后端，替代旧 Fortran 对答案程序：工况一等对象、参数版本留档、
源迭代求 k_eff 与基模通量、改区后可热启动省步、临界搜索做成可取消的异步作业。
仅通过 HTTP 对外，不带页面。

- 技术栈：FastAPI + Python 3.11 + NumPy（只做数组运算）
- 三对角追赶、源迭代、二分求根全部手写，不调用 SciPy / `np.linalg`
- 持久化：SQLite（WAL），数据随容器卷 `/data` 保留

---

## 1. 物理模型与离散

沿 x 排若干区，每区给厚度、`D`、`Σa`、`νΣf`、均匀网格数。
单群扩散特征值问题

```
-d/dx (D dφ/dx) + Σa φ = (1/k) νΣf φ
```

**有限体积（cell-centered）**，在每个网格点做积分平衡：

```
J_{i-1/2} - J_{i+1/2} + Σa_i h_i φ_i = (νΣf_i h_i / k) φ_i
```

### 区界面：严格净流连续，不取平均截面

界面两侧 φ 连续、J=−D dφ/dx 连续。从这两个条件消去界面未知通量，
得到调和导纳（transmissibility）

```
t_{i,i+1} = 2 / (h_i/D_i + h_{i+1}/D_{i+1})
```

它在均匀材料里自然退化为 `D/h`，跨材料界面也严格守恒——**不是把两侧
D 或截面取算术平均**糊过去。`tests/test_core_physics.py` 里
`test_interface_current_continuity` 用界面两侧单侧梯度重构净流，
验证二者在 1e-12 内相等。

### 两端边界（各自独立三选一）

| 边界 | 含义 | 表面导纳 |
|---|---|---|
| `zero` | 物理表面 φ=0 | `2D/h` |
| `extrap` | 表面外 δ=2.13D 处 φ=0（Robin：J=D/δ·φ_s） | `D/(h/2+δ)` |
| `reflect` | 全反射/对称面，零净流 | `0` |

> 注意：`extrap` 的离散格式在网格加密时二阶收敛到 Robin 精确本征方程
> `tan(Ba/2)=1/(Bδ)` 的根，**不是**近似公式 `B=π/(a+2δ)`。
> `app/core/reference.py` 用二分求该精确根作为收敛阶测试的对照。

### 特征值迭代（源迭代 / 功率迭代）

1. 归一化裂变源 `s = Fφ`（`F_i = νΣf_i·h_i`）；
2. 手写 Thomas 追赶解三对角 `A φ_raw = s`；
3. Rayleigh 商更新 `k_new = ⟨Fφ_raw,1⟩ / ⟨s,1⟩`（源已归一化，直接是 k 新估计）；
4. 归一化 `φ_raw` 使**全场总裂变率等于用户给定值**；
5. 双容差：`|Δk|/k < tol_k` **且** `‖φ_new−φ‖₂/‖φ_new‖₂ < tol_phi` 才判收敛。

默认 `tol_k = tol_phi = 1e-10`，`max_iter = 20000`，均可按请求覆盖。
到上限未收敛**显式报 409 失败**，带最后一步的 `residual_k`、`residual_phi`
与已迭代次数，且**不留档、不返回半成品**。

### 中子平衡

每区吸收率/裂变率、左右两端泄漏率单独返回。全场满足

```
总裂变产生率 / k = Σ各区吸收率 + 左端泄漏 + 右端泄漏
```

相对误差 `< 1e-8`（实测在 1e-15 量级），冷、热启动都守这条。

---

## 2. 热启动策略与真实取舍

**初值怎么取**：取同一工况**上一版**最近一次收敛解，把旧网格点通量
**逐点线性插值**到新网格（越界处夹住，零通量边界自然落到 0），
k 初值沿用旧 k；插值后夹掉可能的负小值。

- 不“原样沿用”：厚度或网格数一改变，旧向量长度对不上；
- 不做外推加速：过冲成负通量反而破坏功率迭代的单调收敛，且引入额外参数。

**停机怎么判**：k 相对变化与通量相对 L2 残差**两者都**满足容差（AND），
而不是只看 k。只看 k 会在通量形状还没站稳时提前停下，热启动尤其容易
带着旧解偏差误收敛——判据必须和冷启动完全相同，才能保证冷热殊途同归。

**实测的真实取舍**（tol=1e-10，燃料+反射层算例）：

| 改动 | 冷启动迭代 | 热启动迭代 | 节省 |
|---|---:|---:|---:|
| νΣf 0.120→0.122（网格不变） | 89 | **2** | 98% |
| Σa 0.100→0.105（网格不变） | 93 | **2** | 98% |
| D 1.0→0.95（网格不变） | 92 | 81 | 12% |
| 厚度 30→31（小幅） | 93 | 90 | 3% |
| 厚度 30→36（+20%，网格随比例变） | 132 | 133 | ≈0 |

**结论**：热启动在“反复调截面凑临界”（网格不变、形状变化小）时一步到位、
省 90%+；但功率迭代的尾部误差由**主导比**（相邻高阶谐波的衰减率）决定，
空间形状变化大的厚度改动会重新引入高阶谐波分量，热启动省下的只是前面的
瞬态，尾部滤除步数省不掉，所以大改厚度几乎不省步。这是功率迭代的固有性质，
不是实现缺陷——要进一步压尾部需上 Chebyshev/ Wielandt 移位加速，
会牺牲单调收敛保证，本版刻意不引入。冷热两条路径的 k 与归一化通量始终
一致（|Δk|~1e-12，通量 L2 偏差 <1e-9）。

测试 `test_cold_hot_consistency_*` 对同一次改动冷、热各算一遍比对。

---

## 3. 工况、版本与结果留档

- 工况有名字，可新建、查询、删除；服务重启后仍在（SQLite 卷）。
- **改某一区参数 → 追加一个不可变版本**，旧版本与其下所有求解结果永不覆盖。
- 每次收敛求解挂在求解那一刻的 `version_id` 下留档：k、通量、分区吸收率/
  裂变率、两端泄漏、迭代次数、残差、本次 `cold/hot`。
- 删除工况级联删除其版本、结果与作业。

---

## 4. 临界搜索（异步作业）

指定某区的 `thickness` 或 `nu_sigma_f` 与搜索区间，提交即返回 `job_id`，
后台线程池跑二分：

- 提交时**先在区间两端各冷启动求解一次**：
  - 任一端不收敛 → 不建作业，直接报该端失败；
  - 两端 `k−1` 同号 → **拒绝**（422）并回报 `k_low/k_high`。
- 二分中每个中点沿用前一步收敛解热启动；判据 `|k−1| < 1e-6`。
- `GET /searches/{id}` 查进度：已走步数、当前区间与两端 k、当前最优点。
- `POST /searches/{id}/cancel` 取消：每步前检查取消标志，取消后**真的停下、
  绝不写入成功结果**（status=`canceled`，result 为空）。
- 步数用尽仍未达标：status=`failed`，回报最终区间、两端 k 与残差。

**参数版本锁定**：作业记录里写死提交那一刻的 `version_id` 和完整参数快照，
全程只认它。同一工况挂多个作业、作业在跑时工况又改版，结果都不会张冠李戴
——见 `test_two_concurrent_jobs_and_version_lock`（作业结果代回提交版参数复核）。
服务重启时残留的 `running` 作业标记为 `interrupted`，不会假死或继续写结果。

---

## 5. 输入校验（逐项指名字段）

422 返回统一结构 `{"error":"input_validation_failed",
"detail":[{"field":...,"reason":...}]}`，一次可报多个字段。拒收：

- 厚度或 D ≤ 0；
- `Σa`、`νΣf` 任一为负；
- 所有区都不含裂变材料（νΣf 全为 0，无正 k 特征值）；
- 区数 > 50；
- 各区网格总数 > 20000。

---

## 6. 模块划分

```
app/
  config.py                 环境变量配置（DB 路径、默认容差/上限/工作线程）
  core/
    models.py               Zone / Problem / 边界类型（不可变数据类）
    validation.py           全部字段级输入校验
    discretize.py           有限体积离散、界面调和导纳、三种边界
    solver.py               手写 Thomas 追赶 + 源迭代（双容差）
    reactions.py            分区反应率、泄漏、中子平衡、热启动插值
    reference.py            裸板解析曲率与 k（Robin 精确本征根）
  services/
    storage.py              SQLite：cases/versions/solves/jobs 四张表
    solver_service.py       求解编排、冷热启动选择、留档
    jobs.py                 线程池作业调度、二分、取消、版本快照
  api/
    schemas.py              HTTP pydantic 模型
  main.py                   FastAPI 路由、统一错误处理、lifespan
tests/                      pytest：物理判据/冷热一致/取消/版本锁定/持久化
```

---

## 7. 运行

### 容器（推荐，卷挂 /data）

```bash
docker build -t diffusion-backend .
docker run -p 8000:8000 -v diffusion-data:/data diffusion-backend
```

### 本地

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 可选：DIFFUSION_DB=/path/to.db DIFFUSION_TOL_K=1e-10 ...
```

交互式 API 文档：FastAPI 自带 `GET /docs`（Swagger UI，开发联调用；
生产仅暴露 HTTP 接口本身，无自定义页面）。

### 测试

```bash
pip install -r requirements-dev.txt
pytest
```

---

## 8. HTTP 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/cases` | 建工况（首版参数） |
| GET | `/cases` / `/cases/{id}` | 列表 / 详情（含版本、结果、作业） |
| DELETE | `/cases/{id}` | 删除 |
| POST | `/cases/{id}/zones/{idx}` | 改某一区 → 追加新版本 |
| POST | `/cases/{id}/solve` | 对当前版本冷/热求解并留档 |
| GET | `/cases/{id}/solves` / `/solves/{sid}` | 结果列表 / 详情（含通量） |
| POST | `/cases/{id}/searches` | 提交临界搜索（202 + job_id） |
| GET | `/searches/{jid}` | 查进度/结果 |
| POST | `/searches/{jid}/cancel` | 取消 |
| GET | `/health` | 健康检查 |

### 建工况示例

```json
POST /cases
{
  "name": "fuel-plus-reflector",
  "left_bc": "zero", "right_bc": "zero",
  "zones": [
    {"thickness": 30, "d": 1.0, "sigma_a": 0.10, "nu_sigma_f": 0.12, "n_mesh": 60},
    {"thickness": 12, "d": 1.5, "sigma_a": 0.02, "nu_sigma_f": 0.0,  "n_mesh": 32}
  ]
}
```

### 热启动求解

```json
POST /cases/{id}/solve
{"start_mode": "hot", "total_fission_rate": 1.0,
 "tol_k": 1e-10, "tol_phi": 1e-10}
```

### 临界搜索

```json
POST /cases/{id}/searches
{"target_zone": 0, "target_field": "thickness",
 "low": 8.0, "high": 40.0, "max_steps": 50}
```
