"""M1 - quadcopter MDO (models on pp.9-10 of the brief).

Design vector x = [D (m), E (Wh), d (m), t (m), V (m/s)].
All model code is written so it runs with complex numbers: gradients are taken by
complex step (exact to round-off) and *checked* against forward finite differences.
"""
from __future__ import annotations
import math, json, os
import numpy as np
from scipy.optimize import minimize
from . import data

RHO, G, N = 1.225, 9.81, 4
M_FIXED, EB, ETA, PAV, CDA, FM0, VTIP = 0.60, 150.0, 0.85, 30.0, 0.08, 0.65, 110.0
E_CF, RHO_CF, SIG_ALLOW = 70e9, 1550.0, 300e6 / 1.5
USABLE = 0.80 * (1 - 0.15)
LB = np.array([0.35, 200.0, 0.015, 0.0005, 8.0])
UB = np.array([0.80, 2000.0, 0.040, 0.0025, 22.0])
NAMES = ["D_m", "E_Wh", "d_m", "t_m", "V_ms"]


class Vehicle:
    def __init__(self, payload=2.5, m_plate=0.25, fm=None, eb=EB):
        self.p, self.m_plate, self.eb = payload, m_plate, eb
        self.fm = fm if fm is not None else (lambda D, T: FM0 + 0 * D * T)

    # ---- disciplines -------------------------------------------------
    def geometry(self, x):
        D, E, d, t, V = x
        L = 0.78 * D - 0.08
        A_t = math.pi / 4 * (d ** 2 - (d - 2 * t) ** 2)
        I = math.pi / 64 * (d ** 4 - (d - 2 * t) ** 4)
        return L, A_t, I

    def mass_fn(self, x, m):
        """Structures+propulsion mass model: returns m_new(x, m). Coupling through Tmax=2mg/n."""
        D, E, d, t, V = x
        L, A_t, _ = self.geometry(x)
        Tmax = 2 * m * G / N
        m_mp = 0.12 + 0.004 * Tmax + 0.05 * (D / 0.6) ** 2.5
        return self.p + M_FIXED + self.m_plate + E / self.eb + N * RHO_CF * A_t * L + N * m_mp

    def mda(self, x, m0=10.0, tol=1e-13, maxit=200, hist=False):
        m, h = m0, [m0]
        for k in range(maxit):
            mn = self.mass_fn(x, m)
            h.append(mn)
            if abs((mn - m).real) < tol * abs(m.real if hasattr(m, 'real') else m):
                m = mn; break
            m = mn
        return (m, h) if hist else m

    def power(self, x, W):
        D, E, d, t, V = x
        A = math.pi * D ** 2 / 4
        T = W / N
        FM = self.fm(D, T)
        vh = np.sqrt(W / (2 * RHO * N * A))
        Ph = W * vh / (FM * ETA) + PAV
        vi = np.sqrt(-V ** 2 / 2 + np.sqrt(V ** 4 / 4 + vh ** 4))
        mu = V / VTIP
        Pc = (W * vi + W * vh * (1 / FM - 1) * (1 + 4.65 * mu ** 2) + 0.5 * RHO * CDA * V ** 3) / ETA + PAV
        return Ph, Pc, FM

    def performance(self, x, m):
        """Given x and the (coupling) mass m: radius (km), powers, structure constraint values."""
        D, E, d, t, V = x
        Wl, We = m * G, (m - self.p) * G
        Phl, Pcl, FMl = self.power(x, Wl)
        Phe, Pce, FMe = self.power(x, We)
        Euse = 3600 * USABLE * E
        R = V * (Euse - 60 * (Phl + Phe)) / (Pcl + Pce) / 1000.0
        L, A_t, I = self.geometry(x)
        Tmax = 2 * m * G / N
        m_tip = 0.12 + 0.004 * Tmax + 0.05 * (D / 0.6) ** 2.5
        m_arm = RHO_CF * A_t * L
        sigma = Tmax * L * (d / 2) / I
        defl = Tmax * L ** 3 / (3 * E_CF * I)
        f1 = 1 / (2 * math.pi) * np.sqrt(3 * E_CF * I / (L ** 3 * (m_tip + 0.24 * m_arm)))
        f_rot = VTIP / (math.pi * D)
        # g <= 0, normalised
        g = np.array([sigma / SIG_ALLOW - 1,
                      defl / (0.01 * L) - 1,
                      1 - f1 / (1.15 * f_rot),
                      (d / t) / 40 - 1])
        return dict(R=R, m=m, Phl=Phl, Pcl=Pcl, Phe=Phe, Pce=Pce, FMl=FMl, FMe=FMe, g=g, Tmax=Tmax,
                    sigma=sigma, defl_ratio=defl / L, f1=f1, f_rot=f_rot, Euse=Euse, L=L, m_tip=m_tip, m_arm=m_arm)

    def evaluate(self, x):
        m = self.mda(x)
        return self.performance(x, m)


# ------------------------------------------------------------------ helpers
def cs_grad(f, x, h=1e-30):
    """complex-step gradient of scalar or vector f"""
    x = np.asarray(x, float)
    f0 = np.asarray(f(x.astype(complex)))
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        xc = x.astype(complex); xc[i] += 1j * h
        J[:, i] = np.imag(np.asarray(f(xc))).ravel() / h
    return J.squeeze() if f0.ndim == 0 else J


def fd_grad(f, x, rel):
    x = np.asarray(x, float); f0 = np.asarray(f(x)).real
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        h = rel * max(abs(x[i]), 1e-8); xp = x.copy(); xp[i] += h
        J[:, i] = ((np.asarray(f(xp)).real - f0) / h).ravel()
    return J.squeeze() if f0.ndim == 0 else J


def to_s(x):   # physical -> [0,1]
    return (np.asarray(x) - LB) / (UB - LB)


def to_x(s):
    return LB + np.asarray(s) * (UB - LB)


# ------------------------------------------------------------------ MDF
def _gvec(o, m_cap):
    g = o["g"]
    return g if m_cap is None else np.r_[g, o["m"] / m_cap - 1]


STARTS = [[0.9, 0.5, 0.6, 0.3, 0.4], [0.6, 0.2, 0.8, 0.5, 0.5], [0.95, 0.8, 0.5, 0.2, 0.3], [0.5, 0.4, 0.9, 0.6, 0.6]]


def solve_mdf(veh, m_cap=12.0, x0=None):
    """MDF: max R(x) s.t. structural g(x)<=0, MTOW(x)<=m_cap; the MDA (fixed point on mass)
    runs inside every function call so every iterate is multidisciplinary-consistent."""
    def ev(s): return veh.evaluate(to_x(s))
    fobj = lambda s: -ev(s)["R"] / 50
    gfun = lambda s: _gvec(ev(s), m_cap)
    cons = [{"type": "ineq", "fun": lambda s: -gfun(s), "jac": lambda s: -cs_grad(gfun, s)}]
    best = None
    starts = ([np.clip(to_s(x0), 0, 1)] if x0 is not None else []) + [np.array(v) for v in STARTS]
    for s0 in starts:
        r = minimize(fobj, s0, jac=lambda s: cs_grad(fobj, s), bounds=[(0, 1)] * 5, constraints=cons,
                     method="SLSQP", options=dict(ftol=1e-13, maxiter=500))
        if r.success and np.all(gfun(r.x) < 1e-7) and (best is None or r.fun < best.fun - 1e-12):
            best = r
    x = to_x(best.x)
    return x, veh.evaluate(x), best


# ------------------------------------------------------------------ IDF
def solve_idf(veh, m_cap=12.0, x0=None):
    """IDF: total mass m_c is an optimiser variable (coupling variable); the consistency
    m_c - mass_fn(x, m_c) = 0 is an equality constraint. No MDA loop is run, so intermediate
    iterates are NOT consistent - only the converged point is.
    (SLSQP stalled on this formulation - see decision log - so trust-constr is used.)"""
    from scipy.optimize import NonlinearConstraint, Bounds
    mlo, mhi = 4.0, 30.0
    def split(z): return to_x(z[:5]), mlo + z[5] * (mhi - mlo)
    fobj = lambda z: -veh.performance(*split(z))["R"] / 50
    hcon = lambda z: np.atleast_1d((split(z)[1] - veh.mass_fn(*split(z))) / 10)
    def gcon(z):
        x, m = split(z); return _gvec(veh.performance(x, m), m_cap)
    best = None
    starts = [np.array(v) for v in STARTS] if x0 is None else [np.clip(to_s(x0), 0, 1)]
    for s0 in starts:
        z0 = np.r_[s0, (veh.mda(to_x(s0)) - mlo) / (mhi - mlo)]   # consistent start, then let it go
        r = minimize(fobj, z0, jac=lambda z: cs_grad(fobj, z), method="trust-constr", bounds=Bounds(0, 1),
                     constraints=[NonlinearConstraint(hcon, 0, 0, jac=lambda z: np.atleast_2d(cs_grad(hcon, z))),
                                  NonlinearConstraint(gcon, -np.inf, 0, jac=lambda z: cs_grad(gcon, z))],
                     options=dict(xtol=1e-12, gtol=1e-10, maxiter=3000))
        ok = abs(hcon(r.x)[0]) < 1e-7 and gcon(r.x).max() < 1e-6
        if ok and (best is None or r.fun < best.fun - 1e-12):
            best = r
    x, m = split(best.x)
    return x, m, veh.performance(x, m), best


# ------------------------------------------------------------------ KKT / multipliers
def kkt_multipliers(veh, x, m_cap=12.0):
    """Recover Lagrange multipliers at the MDF optimum by least squares on the stationarity
    condition grad(R) = sum_k lam_k grad(g_k) over active constraints (incl. bounds), then the
    multiplier of the payload equality p = 2.5 from the envelope theorem."""
    s = to_s(x)
    def ev(ss, p=None):
        v = Vehicle(payload=veh.p if p is None else p, m_plate=veh.m_plate, fm=veh.fm, eb=veh.eb)
        return v.evaluate(to_x(ss))
    out = ev(s)
    gR = cs_grad(lambda q: ev(q)["R"], s)
    Jg = cs_grad(lambda q: _gvec(ev(q), m_cap), s)
    out = dict(out); out["g"] = _gvec(out, m_cap)
    active, A = [], []
    names = ["bending stress", "tip deflection", "natural frequency", "d/t slenderness", "MTOW cap"]
    for k in range(len(out["g"])):
        if out["g"][k] > -1e-4:
            active.append(names[k]); A.append(Jg[k])
    for i in range(5):
        if s[i] < 1e-6: active.append(f"{NAMES[i]} lower bound"); A.append(-np.eye(5)[i])
        if s[i] > 1 - 1e-6: active.append(f"{NAMES[i]} upper bound"); A.append(np.eye(5)[i])
    A = np.array(A)
    # max R  <=> grad R = sum lam * grad(c), lam >= 0 for c <= 0
    lam, *_ = np.linalg.lstsq(A.T, gR, rcond=None)
    resid = np.linalg.norm(A.T @ lam - gR) / np.linalg.norm(gR)
    # envelope theorem: dR*/dp = dR/dp - sum lam dc/dp   (p enters model only, not x)
    dp = 1e-30
    def Rp(p): return ev(s.astype(complex), p)["R"]
    def gp(p): return _gvec(ev(s.astype(complex), p), m_cap)
    dRdp = np.imag(Rp(veh.p + 1j * dp)) / dp
    dgdp = np.imag(gp(veh.p + 1j * dp)) / dp
    dg_act = [dgdp[names.index(a)] if a in names else 0.0 for a in active]
    lam_p = dRdp - float(np.dot(lam, dg_act))
    return dict(active=active, lam=dict(zip(active, lam.round(4))), stationarity_resid=resid,
                dRstar_dp_km_per_kg=lam_p, km_lost_per_100g=-0.1 * lam_p)


# ------------------------------------------------------------------ catalogue snap
def snap(veh, x):
    comp = data.components()
    D, E, d, t, V = x
    out = veh.evaluate(x)
    bats = comp["battery"]
    best = None
    for _, b in bats.iterrows():
        for k in (1, 2, 3):
            Ek = k * b.wh
            if Ek < 0.9 * E or Ek > 1.25 * E: continue
            cand = (abs(Ek - E), b.part_id, k, Ek)
            best = cand if best is None or cand < best else best
    if best is None:   # nearest available
        b = bats.iloc[(bats.wh - E).abs().argmin()]; best = (0, b.part_id, 1, b.wh)
    tubes = comp["tube"].sort_values("mass_kg")
    Dn = round(D / 0.0254) * 0.0254   # whole-inch prop
    feas = None
    for _, tb in tubes.iterrows():
        xs = np.array([Dn, best[3], tb.outer_dia_mm / 1000, tb.wall_mm / 1000, V])
        o = veh.evaluate(xs)
        if np.all(o["g"] <= 1e-9):
            feas = (tb.part_id, xs, o); break
    motors = comp["motor"].sort_values("max_thrust_n")
    mot = motors[motors.max_thrust_n >= feas[2]["Tmax"]] if feas else motors.iloc[[-1]]
    mot_id = mot.iloc[0].part_id if len(mot) else "NONE (Tmax exceeds catalogue)"
    return dict(prop_in=round(Dn / 0.0254), battery=f"{best[2]}x {best[1]}", E_Wh=best[3], tube=feas[0] if feas else None,
                motor=mot_id, x=feas[1] if feas else None, perf=feas[2] if feas else None)


# ------------------------------------------------------------------ driver
def summarize(x, o):
    return dict(D_m=x[0], E_Wh=x[1], d_mm=x[2] * 1e3, t_mm=x[3] * 1e3, V_ms=x[4], MTOW_kg=o["m"], R_km=o["R"],
                Tmax_N=o["Tmax"], P_hover_loaded_W=o["Phl"], P_cruise_loaded_W=o["Pcl"], P_cruise_empty_W=o["Pce"],
                FM_loaded=o["FMl"], FM_empty=o["FMe"])


def run(m_plate=0.25, fm=None, tag="m1", make_figs=True, eb=EB, m_cap=12.0, caps=(7.0, 8.5, 10.0, 12.0, 14.0), quick=False):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import pandas as pd
    figd = os.path.join(data.OUT, "figs"); tabd = os.path.join(data.OUT, "tables")
    veh = Vehicle(m_plate=m_plate, fm=fm, eb=eb)
    res = dict(m_cap=m_cap, m_plate=m_plate)
    x_mdf, o_mdf, r_mdf = solve_mdf(veh, m_cap)
    res["mdf"] = summarize(x_mdf, o_mdf)
    if not quick:
        x_u, o_u, _ = solve_mdf(veh, None)
        res["uncapped_maxR"] = summarize(x_u, o_u)
        x_idf, m_idf, o_idf, r_idf = solve_idf(veh, m_cap, x0=None)
        res["idf"] = summarize(x_idf, o_idf)
        res["idf_consistency_residual_kg"] = float(m_idf - veh.mass_fn(x_idf, m_idf))
        res["mdf_idf_R_rel_diff"] = abs(o_mdf["R"] - o_idf["R"]) / o_mdf["R"]
        res["mdf_idf_x_rel_diff"] = float((np.abs(x_mdf - x_idf) / np.abs(x_mdf)).max())
        res["mdf_nfev"], res["idf_nfev"] = int(r_mdf.nfev), int(r_idf.nfev)
        res["kkt"] = kkt_multipliers(veh, x_mdf, m_cap)
        v2 = Vehicle(payload=veh.p + 0.1, m_plate=m_plate, fm=fm, eb=eb)
        x2, o2, _ = solve_mdf(v2, m_cap, x0=x_mdf)
        res["reopt_km_lost_per_100g"] = o_mdf["R"] - o2["R"]
        gc = []
        for label, xx in [("optimum", x_mdf), ("interior", np.array([0.6, 700, 0.025, 0.0012, 14.0]))]:
            s = to_s(xx)
            f = lambda q: (lambda o: np.r_[o["R"], o["m"], o["g"]])(veh.evaluate(to_x(q)))
            Jc = cs_grad(f, s)
            for h in (1e-4, 1e-7):
                Jf = fd_grad(f, s, h)
                gc.append(dict(point=label, fd_rel_step=h, max_abs_err=np.abs(Jf - Jc).max(),
                               max_rel_err=float(np.max(np.abs(Jf - Jc) / (np.abs(Jc) + 1e-3 * np.abs(Jc).max())))))
        res["gradcheck"] = gc
        pd.DataFrame(gc).to_csv(os.path.join(tabd, f"{tag}_gradcheck.csv"), index=False)
    designs = []
    for c in caps:
        x, o, _ = solve_mdf(veh, c, x0=x_mdf)
        sn = snap(veh, x)
        dd = summarize(x, o); dd.update(m_cap=c, snap_prop_in=sn["prop_in"], snap_battery=sn["battery"],
                                        snap_tube=sn["tube"], snap_motor=sn["motor"],
                                        snap_R_km=sn["perf"]["R"] if sn["perf"] else np.nan,
                                        snap_MTOW_kg=sn["perf"]["m"] if sn["perf"] else np.nan,
                                        snap_x=json.dumps(list(map(float, sn["x"]))) if sn["x"] is not None else None)
        designs.append(dd)
    dfd = pd.DataFrame(designs); dfd.index = [f"V{i+1}" for i in range(len(dfd))]
    dfd.to_csv(os.path.join(tabd, f"{tag}_designs.csv"))
    json.dump(res, open(os.path.join(tabd, f"{tag}_summary.json"), "w"), indent=1, default=float)
    if quick or not make_figs:
        return res, dfd, None
    pr = []
    for p in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        vp = Vehicle(payload=p, m_plate=m_plate, fm=fm, eb=eb)
        xp, op, _ = solve_mdf(vp, m_cap, x0=x_mdf)
        pr.append(dict(payload=p, Rmax_km=op["R"], E_Wh=xp[1]))
    pr = pd.DataFrame(pr); pr.to_csv(os.path.join(tabd, f"{tag}_payload_radius.csv"), index=False)
    m_end, hist = veh.mda(x_mdf, m0=3.0, hist=True)
    _, hist2 = veh.mda(x_mdf, m0=40.0, hist=True)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    ax[0].semilogy(np.abs(np.diff(hist)) + 1e-16, "o-", label="start 3 kg")
    ax[0].semilogy(np.abs(np.diff(hist2)) + 1e-16, "s-", label="start 40 kg")
    ax[0].set_xlabel("iteration k"); ax[0].set_ylabel("|m(k+1) - m(k)|  (kg)")
    ax[0].set_title(f"MDA fixed point: contraction 0.008 g = {0.008*G:.4f}"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(pr.payload, pr.Rmax_km, "o-k", lw=2, label=f"optimum per payload (MTOW<={m_cap:g} kg)")
    for k, (_, d) in enumerate(dfd.iterrows()):
        xx = np.array([d.D_m, d.E_Wh, d.d_mm / 1e3, d.t_mm / 1e3, d.V_ms])
        ps = np.linspace(0.0, 3.5, 15)
        ax[1].plot(ps, [Vehicle(payload=p, m_plate=m_plate, fm=fm, eb=eb).evaluate(xx)["R"] for p in ps], "--", alpha=.7, label=f"{dfd.index[k]} flown off-design")
    ax[1].axvline(2.5, color="grey", lw=.8); ax[1].set_xlabel("payload (kg)"); ax[1].set_ylabel("radius (km)")
    ax[1].set_title("Payload-radius trade-off"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    ax[2].plot(dfd.MTOW_kg, dfd.R_km, "o-", label="continuous optimum"); ax[2].plot(dfd.snap_MTOW_kg, dfd.snap_R_km, "x--", label="snapped to catalogue")
    [ax[2].annotate(i, (r.MTOW_kg, r.R_km), textcoords="offset points", xytext=(4, -10)) for i, r in dfd.iterrows()]
    ax[2].set_xlabel("MTOW (kg)"); ax[2].set_ylabel("radius at 2.5 kg (km)"); ax[2].set_title("Designs passed to M4/M6"); ax[2].legend(); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(figd, f"{tag}_mda_tradeoff.png"), dpi=130); plt.close()
    return res, dfd, pr


if __name__ == "__main__":
    res, dfd, pr = run()
    print(json.dumps({k: v for k, v in res.items()}, indent=1, default=float))
    print(dfd.round(3).T)
    print(pr.round(2))
