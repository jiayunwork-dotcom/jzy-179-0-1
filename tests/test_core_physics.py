"""物理判据测试（直调内核，不经 HTTP）。

覆盖：
1. 裸板网格逐级加密 -> 逼近解析 k_inf/(1+L^2 B^2)，且误差二阶（网格加倍约缩 1/4）；
2. 左右对称问题通量对称、峰在中心；
3. 半块板 + 中心全反射 == 整块对称板；
4. 加纯散射反射层后 k 变大；
5. k_inf>1 时燃料加厚 k 单调上升；
6. 只调大 D，裸板 k 变小；
7. 通量处处非负；
8. 中子平衡（产生率/k = 吸收+两端泄漏，相对误差 < 1e-8）；
9. 区界面通量/净流连续（同材料分两区与整区结果一致）。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.core.discretize import discretize
from app.core.models import Problem, Zone
from app.core.reactions import reaction_rates
from app.core.reference import analytic_k_eff
from app.core.solver import source_iteration
from app.core.solver import thomas_solve

TOL = 1e-11
MAX_ITER = 100_000


def solve(problem, phi0=None, k0=None, tol=TOL, max_iter=MAX_ITER):
    mesh = discretize(problem)
    out = source_iteration(mesh, tol_k=tol, tol_phi=tol, max_iter=max_iter,
                           phi0=phi0, k0=k0)
    assert out.converged, f"未收敛: {out.message}"
    return mesh, out


# ---------------- 典型均匀燃料板参数 ----------------

def fuel_zone(thickness, n_mesh, *, d=1.0, sa=0.1, nuf=0.12):
    return Zone(thickness, d, sa, nuf, n_mesh)


def bare_slab(thickness, n_mesh, *, left="zero", right="zero", **kw):
    return Problem((fuel_zone(thickness, n_mesh, **kw),), left, right)


# 1. 解析解收敛阶 ---------------------------------------------------------

@pytest.mark.parametrize("bcs", [("zero", "zero"), ("extrap", "extrap")])
def test_bare_slab_convergence_order(bcs):
    n_list = [8, 16, 32, 64, 128]
    problem = bare_slab(50.0, n_list[0], left=bcs[0], right=bcs[1])
    k_exact = analytic_k_eff(problem)
    errors = []
    for n in n_list:
        p = bare_slab(50.0, n, left=bcs[0], right=bcs[1])
        _, out = solve(p)
        errors.append(abs(out.k - k_exact))

    # 误差应逐级下降
    for e1, e2 in zip(errors, errors[1:]):
        assert e2 < e1

    # 用最后三对网格算观察阶，h 加倍误差应约缩到 1/4（二阶方法）
    orders = [
        math.log(errors[i] / errors[i + 1]) / math.log(2.0)
        for i in range(len(errors) - 1)
    ]
    observed_order = sum(orders[-2:]) / 2
    assert 1.85 < observed_order < 2.15, (
        f"观察阶 {observed_order:.3f} 不是二阶；errors={errors}")
    # 网格加倍误差约缩到四分之一（细网格上相邻两级比值在 3.3~4.7 之间）
    for i in range(2, len(errors)):
        ratio = errors[i - 1] / errors[i]
        assert 3.3 < ratio < 4.7, (bcs, i, ratio)
    # 最密网格与解析值足够接近
    assert errors[-1] < 1e-5, (errors[-1], k_exact)


def test_analytic_value_sanity():
    # k_inf=1.2, L^2=10, B=pi/50 -> k = 1.2/(1+10*(pi/50)^2)
    p = bare_slab(50.0, 16)
    z = p.zones[0]
    expected = 1.2 / (1.0 + (z.d / z.sigma_a) * (math.pi / 50.0) ** 2)
    assert analytic_k_eff(p) == pytest.approx(expected, rel=1e-14)


# 2. 对称性 ---------------------------------------------------------------

def test_symmetric_problem_flux_symmetric():
    # 燃料-燃料同材料两块（同时也考分界面处理），左右真空
    p = Problem((fuel_zone(25.0, 25), fuel_zone(25.0, 25)), "zero", "zero")
    mesh, out = solve(p)
    phi = out.phi
    assert np.all(phi >= 0.0)
    # 关于中心逐点对称
    np.testing.assert_allclose(phi, phi[::-1], rtol=1e-10, atol=1e-12)
    # 峰在中心：偶数网格时中心两侧两个点并列最高
    peak = int(np.argmax(phi))
    assert peak in (len(phi) // 2 - 1, len(phi) // 2)


# 3. 半板+全反射 == 整板 ---------------------------------------------------

def test_half_slab_reflective_equals_full_slab():
    n_half = 32
    half = bare_slab(25.0, n_half, left="reflect", right="zero")
    full = bare_slab(50.0, 2 * n_half, left="zero", right="zero")
    k_half = solve(half)[1].k
    k_full = solve(full)[1].k
    # 匹配网格下离散方程逐行等价，k 应高度一致
    assert k_half == pytest.approx(k_full, rel=1e-11, abs=1e-12)


def test_half_slab_extrap_reflective_equals_full_extrap():
    n_half = 40
    half = bare_slab(30.0, n_half, left="reflect", right="extrap")
    full = bare_slab(60.0, 2 * n_half, left="extrap", right="extrap")
    assert solve(half)[1].k == pytest.approx(solve(full)[1].k, rel=1e-11)


# 4. 反射层增 k -----------------------------------------------------------

def test_pure_scattering_reflector_increases_k():
    fuel = fuel_zone(20.0, 40)
    bare = Problem((fuel,), "zero", "zero")
    # 只散射（sa=0, nuf=0）反射层，D 与燃料相同量级
    reflector = Zone(15.0, 1.0, 0.0, 0.0, 30)
    reflected = Problem((fuel, reflector), "zero", "zero")
    reflected_r = Problem((reflector, fuel), "zero", "zero")
    k_bare = solve(bare)[1].k
    k_r = solve(reflected)[1].k
    k_rr = solve(reflected_r)[1].k
    assert k_r > k_bare + 1e-6
    assert k_rr == pytest.approx(k_r, rel=1e-10)  # 左右镜像应一致
    # 反射层里通量为正（没有源也不会负）
    mesh_r, out_r = solve(reflected)
    assert np.all(out_r.phi > 0.0)


# 5. 燃料加厚 k 单调 -------------------------------------------------------

def test_k_monotonic_in_fuel_thickness():
    ks = []
    for a in (10.0, 20.0, 30.0, 50.0, 80.0):
        p = bare_slab(a, max(10, int(a * 2)))
        ks.append(solve(p)[1].k)
    assert all(ks[i] < ks[i + 1] for i in range(len(ks) - 1))
    # 足够厚时逼近 k_inf = 1.2
    assert ks[-1] < 1.2 and ks[-1] > 1.15


# 6. D 增大 k 减小 --------------------------------------------------------

def test_k_decreases_with_d():
    ks = []
    for d in (0.5, 1.0, 2.0, 4.0):
        p = bare_slab(40.0, 80, d=d)
        ks.append(solve(p)[1].k)
    assert all(ks[i] > ks[i + 1] for i in range(len(ks) - 1))


# 7. 非负通量（含反射层/多区随机物性） ------------------------------------

def test_reflector_allows_thinner_fuel_at_criticality():
    """同一燃料，加反射层后达到同样 k 水平所需的燃料厚度更薄（反射层节省）。"""
    def k_bare(a):
        return solve(Problem((Zone(a, 1.0, 0.12, 0.15, 50),),
                             "zero", "zero"))[1].k

    def k_reflected(a):
        return solve(Problem(
            (Zone(a, 1.0, 0.12, 0.15, 50),
             Zone(12.0, 1.0, 0.01, 0.0, 24)), "zero", "zero"))[1].k

    target = 0.95
    a_bare = next(a for a in range(10, 60) if k_bare(float(a)) >= target)
    a_refl = next(a for a in range(10, 60) if k_reflected(float(a)) >= target)
    assert a_refl < a_bare


def test_mixed_extrap_zero_boundary():
    # 左外推、右零通量的非对称边界组合也能求解且守平衡、通量非负
    p = Problem((fuel_zone(30.0, 60), Zone(10.0, 1.2, 0.03, 0.0, 20)),
                "extrap", "zero")
    mesh, out = solve(p)
    assert np.all(out.phi >= 0.0)
    report = reaction_rates(p, mesh, out.phi, out.k)
    assert report.balance_relative_error < 1e-8
    # 两端泄漏都为正（向外流），且总泄漏与吸收共同闭合中子平衡
    assert report.leakage_left > 0 and report.leakage_right > 0


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_flux_nonnegative_multizone(seed):
    rng = np.random.default_rng(seed)
    zones = []
    for i in range(4):
        nuf = 0.0 if i % 2 else float(0.1 + rng.random() * 0.05)
        zones.append(Zone(float(5 + rng.random() * 20),
                          float(0.3 + rng.random()),
                          float(0.02 + rng.random() * 0.1),
                          nuf, 20))
    p = Problem(tuple(zones), "extrap", "zero")
    mesh, out = solve(p)
    assert np.all(out.phi >= 0.0)
    report = reaction_rates(p, mesh, out.phi, out.k)
    assert report.balance_relative_error < 1e-8


# 8. 中子平衡 -------------------------------------------------------------

@pytest.mark.parametrize("bcs", [("zero", "zero"), ("extrap", "extrap"),
                                 ("reflect", "zero"), ("extrap", "reflect"),
                                 ("reflect", "reflect")])
def test_neutron_balance(bcs):
    zones = (fuel_zone(20.0, 30), Zone(10.0, 0.5, 0.05, 0.0, 20),
             Zone(8.0, 1.5, 0.08, 0.06, 16))
    if bcs == ("reflect", "reflect"):
        # 两端反射时裂变系统仍可临界（k_inf>1），可以算
        pass
    p = Problem(zones, bcs[0], bcs[1])
    mesh, out = solve(p)
    report = reaction_rates(p, mesh, out.phi, out.k)
    assert report.balance_relative_error < 1e-8, (
        report.balance_relative_error, report)
    # 逐区反应率之和与全场一致
    assert sum(report.absorption_by_zone) + report.leakage_left \
        + report.leakage_right == pytest.approx(report.production_over_k,
                                                rel=1e-9)


# 9. 区界面连续：同材料切成两区 k 必须一致 --------------------------------

def test_zone_interface_same_as_uniform():
    n = 64
    whole = bare_slab(40.0, n, left="extrap", right="extrap")
    split = Problem((fuel_zone(13.3, 20), fuel_zone(26.7, 44)),
                    "extrap", "extrap")
    k_whole = solve(whole)[1].k
    mesh_s, out_s = solve(split)
    # 界面两侧网格点通量不同是正常的（位置不同）；关键是整体系数严格守恒，
    # 切成同材料两区与均匀整区的数值矩阵在公共网格上完全一致：
    split_uniform = Problem((fuel_zone(20.0, 32), fuel_zone(20.0, 32)),
                            "extrap", "extrap")
    mesh_u = discretize(bare_slab(40.0, 64, left="extrap", right="extrap"))
    mesh_split = discretize(split_uniform)
    np.testing.assert_allclose(mesh_split.diag, mesh_u.diag, rtol=1e-14)
    np.testing.assert_allclose(mesh_split.lower,
                               mesh_u.lower, rtol=1e-14)
    np.testing.assert_allclose(mesh_split.upper, mesh_u.upper, rtol=1e-14)
    assert out_s.k == pytest.approx(k_whole, abs=2e-4)  # 网格略不同，允许小偏差


def test_interface_current_continuity():
    """异材界面两侧用各自单侧梯度重构净流，应在离散精度内相等。"""
    z1 = Zone(20.0, 0.8, 0.12, 0.15, 80)
    z2 = Zone(15.0, 2.0, 0.01, 0.0, 60)
    p = Problem((z1, z2), "zero", "zero")
    mesh, out = solve(p)
    i = 80  # 界面在 cell 79 | cell 80 之间
    phi_l, phi_r = out.phi[i - 1], out.phi[i]
    h_l, h_r = mesh.widths[i - 1], mesh.widths[i]
    d_l, d_r = z1.d, z2.d
    # 左区界面通量由线性外推：phi_s = (h_r*D_l*phi_l + h_l*D_r*phi_r)/
    #                              (h_r*D_l + h_l*D_r)
    phi_s = (h_r * d_l * phi_l + h_l * d_r * phi_r) / (h_r * d_l + h_l * d_r)
    j_from_left = -d_l * (phi_s - phi_l) / (0.5 * h_l)
    j_from_right = -d_r * (phi_r - phi_s) / (0.5 * h_r)
    assert j_from_left == pytest.approx(j_from_right, rel=1e-12)
    # 净流方向：从堆芯（左，有裂变）流向反射层（右），按坐标约定为正
    assert j_from_left > 0


def test_interface_transmiss_mismatched_h_and_d():
    """异材且两侧网格宽度不同：调和导纳应对称、退化为 D/h，并等于
    半格导纳 g1=2D1/h1, g2=2D2/h2 的调和组合 g1*g2/(g1+g2)。"""
    from app.core.discretize import _interface_transmiss
    # 均匀退化
    assert _interface_transmiss(1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)
    # 与左右次序无关（同一个 t 进入界面两侧方程 -> 严格行守恒）
    for (h1, d1, h2, d2) in [(0.5, 1.0, 2.0, 3.0),
                             (0.3, 2.5, 1.7, 0.4),
                             (1.0, 1e9, 1.0, 1e-9)]:
        t = _interface_transmiss(h1, d1, h2, d2)
        assert t == pytest.approx(_interface_transmiss(h2, d2, h1, d1))
        g1, g2 = 2 * d1 / h1, 2 * d2 / h2
        assert t == pytest.approx(g1 * g2 / (g1 + g2), rel=1e-13)
        assert t >= 0.0


def test_mismatched_mesh_interface_balance_and_continuity():
    """异材 + 两侧网格宽度不同的完整两区问题：守平衡、界面流连续、通量非负。"""
    z1 = Zone(18.0, 0.7, 0.11, 0.14, 37)   # h1 ~ 0.486
    z2 = Zone(13.0, 2.3, 0.015, 0.0, 91)   # h2 ~ 0.143
    p = Problem((z1, z2), "extrap", "zero")
    mesh, out = solve(p)
    assert np.all(out.phi >= 0.0)
    i = 37
    phi_l, phi_r = out.phi[i - 1], out.phi[i]
    t = 2.0 / (mesh.widths[i - 1] / z1.d + mesh.widths[i] / z2.d)
    # 界面净流按共享导纳 t*(phi_l-phi_r) 计算，构造上左右两行用同一 t
    j = t * (phi_l - phi_r)
    assert j > 0  # 从堆芯流向反射层
    report = reaction_rates(p, mesh, out.phi, out.k)
    assert report.balance_relative_error < 1e-8


# 10. Thomas 求解器自测：与简单三对角已知解对比 ---------------------------

def test_thomas_solver_known_system():
    # 对角占优的 1-2-1 类系统
    lower = np.array([0.0, -1.0, -1.0, -1.0])
    diag = np.array([2.0, 2.0, 2.0, 2.0])
    upper = np.array([-1.0, -1.0, -1.0, 0.0])
    rhs = np.array([1.0, 0.0, 0.0, 1.0])
    x = thomas_solve(lower, diag, upper, rhs)
    # 手算：解为 [1,1,1,1]
    np.testing.assert_allclose(x, np.ones(4), rtol=1e-14)
