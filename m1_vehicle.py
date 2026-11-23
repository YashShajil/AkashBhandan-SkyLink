"""M1 - quadcopter MDO (models on pp.9-10 of the brief).

Design vector x = [D (m), E (Wh), d (m), t (m), V (m/s)].
All model code is written so it runs with complex numbers: gradients are taken by
complex step (exact to round-off) and *checked* against forward finite differences.
"""
from __future__ import annotations
import sys, os
if __package__ in (None, ""):          # allow running this file directly (PyCharm Run button)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "skylink"
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
R_REQ = 22.0          # required mission radius at full payload (km) - the M1 requirement


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


# ------------------------------------------------------------------ formulation
# Objective (brief option 2): MINIMISE take-off mass (MTOW) at full payload,
# subject to   structure g(x) <= 0   and   radius R(x) >= R_req.
GNAMES = ["bending stress", "tip deflection", "natural frequency", "d/t slenderness", "radius requirement"]
STARTS = [[0.9, 0.5, 0.6, 0.3, 0.4], [0.6, 0.2, 0.8, 0.5, 0.5], [0.95, 0.8, 0.5, 0.2, 0.3], [0.5, 0.4, 0.9, 0.6, 0.6]]


def _gvec(o, R_req):
    """all inequality constraints as c <= 0: 4 structural + radius requirement"""
    return np.r_[o["g"], 1 - o["R"] / R_req]


# ------------------------------------------------------------------ MDF
def solve_mdf(veh, R_req=R_REQ, x0=None):
    """MDF: min MTOW(x) s.t. g(x) <= 0, R(x) >= R_req. The MDA (fixed point on mass) runs
    inside every function call, so every iterate is multidisciplinary-consistent."""
    def ev(s): return veh.evaluate(to_x(s))
    fobj = lambda s: ev(s)["m"] / 10
    gfun = lambda s: _gvec(ev(s), R_req)
    cons = [{"type": "ineq", "fun": lambda s: -gfun(s), "jac": lambda s: -cs_grad(gfun, s)}]
    best = None
    starts = ([np.clip(to_s(x0), 0, 1)] if x0 is not None else []) + [np.array(v) for v in STARTS]
    for s0 in starts:
        r = minimize(fobj, s0, jac=lambda s: cs_grad(fobj, s), bounds=[(0, 1)] * 5, constraints=cons,
                     method="SLSQP", options=dict(ftol=1e-13, maxiter=500))
        if r.success and np.all(gfun(r.x) < 1e-7) and (best is None or r.fun < best.fun - 1e-12):
            best = r
    if best is None:
        raise ValueError(f"no feasible design reaches R = {R_req} km within the bounds")
    x = to_x(best.x)
    return x, veh.evaluate(x), best


def solve_maxR(veh):
    """max R with no mass limit: used only to show why max-R is a poor objective (all bounds active)"""
    def ev(s): return veh.evaluate(to_x(s))
    fobj = lambda s: -ev(s)["R"] / 50
    gfun = lambda s: ev(s)["g"]
    best = None
    for s0 in [np.array(v) for v in STARTS]:
        r = minimize(fobj, s0, jac=lambda s: cs_grad(fobj, s), bounds=[(0, 1)] * 5, method="SLSQP",
                     constraints=[{"type": "ineq", "fun": lambda s: -gfun(s), "jac": lambda s: -cs_grad(gfun, s)}],
                     options=dict(ftol=1e-13, maxiter=500))
        if r.success and (best is None or r.fun < best.fun): best = r
    x = to_x(best.x)
    return x, veh.evaluate(x)


# ------------------------------------------------------------------ IDF
def solve_idf(veh, R_req=R_REQ, x0=None):
    """IDF: the total mass m_c is an optimiser variable (coupling variable) and IS the objective;
    the consistency m_c - mass_fn(x, m_c) = 0 is an equality constraint. No MDA loop is run, so
    intermediate iterates are NOT consistent - only the converged point is.
    (SLSQP stalled on this formulation - see decision log - so trust-constr is used.)"""
    from scipy.optimize import NonlinearConstraint, Bounds
    mlo, mhi = 4.0, 30.0
    def split(z): return to_x(z[:5]), mlo + z[5] * (mhi - mlo)
    fobj = lambda z: split(z)[1] / 10
    fjac = lambda z: np.r_[np.zeros(5), (mhi - mlo) / 10]
    hcon = lambda z: np.atleast_1d((split(z)[1] - veh.mass_fn(*split(z))) / 10)
    def gcon(z):
        x, m = split(z); return _gvec(veh.performance(x, m), R_req)
    best = None
    starts = [np.array(v) for v in STARTS] if x0 is None else [np.clip(to_s(x0), 0, 1)]
    import warnings
    for s0 in starts:
        z0 = np.r_[s0, (veh.mda(to_x(s0)) - mlo) / (mhi - mlo)]   # consistent start, then let it go
        # Known, harmless solver messages: the objective (m_c) and the consistency constraint are
        # (nearly) linear, so trust-constr's quasi-Newton update sees delta_grad == 0; rejected trial
        # steps can briefly give m_c < payload (sqrt of a negative weight). The converged point is
        # checked below (consistency residual) and against MDF, so these are silenced here only.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="delta_grad == 0.0")
            warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
            r = minimize(fobj, z0, jac=fjac, hess=lambda z: np.zeros((6, 6)), method="trust-constr", bounds=Bounds(0, 1),
                         constraints=[NonlinearConstraint(hcon, 0, 0, jac=lambda z: np.atleast_2d(cs_grad(hcon, z))),
                                      NonlinearConstraint(gcon, -np.inf, 0, jac=lambda z: cs_grad(gcon, z))],
                         options=dict(xtol=1e-12, gtol=1e-10, maxiter=3000))
        ok = abs(hcon(r.x)[0]) < 1e-7 and gcon(r.x).max() < 1e-6
        if ok and (best is None or r.fun < best.fun - 1e-12):
            best = r
    x, m = split(best.x)
    return x, m, veh.performance(x, m), best


# ------------------------------------------------------------------ KKT / multipliers
def kkt_multipliers(veh, x, R_req=R_REQ):
    """Lagrange multipliers at the MDF optimum from the stationarity condition
         grad(m) + sum_k lam_k grad(c_k) = 0,  lam_k >= 0, over active constraints (incl. bounds),
    solved by least squares. Then, by the envelope theorem:
         dm*/dp     = dm/dp + sum lam_k dc_k/dp          (payload multiplier, kg MTOW per kg payload)
         dm*/dR_req = lam_R * R/R_req^2                   (mass cost of 1 km more radius)
    and radius lost per extra 100 g at FIXED MTOW = 0.1 * (dm*/dp) / (dm*/dR_req)."""
    s = to_s(x)
    def ev(ss, p=None):
        v = Vehicle(payload=veh.p if p is None else p, m_plate=veh.m_plate, fm=veh.fm, eb=veh.eb)
        return v.evaluate(to_x(ss))
    out = ev(s)
    c = _gvec(out, R_req)
    gm = cs_grad(lambda q: ev(q)["m"], s)
    Jc = cs_grad(lambda q: _gvec(ev(q), R_req), s)
    active, A, kind = [], [], []
    for k in range(len(c)):
        if c[k] > -1e-4: active.append(GNAMES[k]); A.append(Jc[k]); kind.append(k)
    for i in range(5):
        if s[i] < 1e-6: active.append(f"{NAMES[i]} lower bound"); A.append(-np.eye(5)[i]); kind.append(None)
        if s[i] > 1 - 1e-6: active.append(f"{NAMES[i]} upper bound"); A.append(np.eye(5)[i]); kind.append(None)
    A = np.array(A)
    lam, *_ = np.linalg.lstsq(A.T, -gm, rcond=None)
    resid = np.linalg.norm(A.T @ lam + gm) / np.linalg.norm(gm)
    h = 1e-30
    dmdp = np.imag(ev(s.astype(complex), veh.p + 1j * h)["m"]) / h
    dcdp = np.imag(_gvec(ev(s.astype(complex), veh.p + 1j * h), R_req)) / h
    dm_star_dp = dmdp + sum(l * dcdp[k] for l, k in zip(lam, kind) if k is not None)
    lamR = dict(zip(active, lam)).get("radius requirement", 0.0)
    dm_star_dR = lamR * out["R"] / R_req ** 2
    km_per_100g = 0.1 * dm_star_dp / dm_star_dR if dm_star_dR > 0 else np.nan
    return dict(active=active, lam=dict(zip(active, np.round(lam, 5))), stationarity_resid=resid,
                dMTOW_dpayload=dm_star_dp, kg_MTOW_per_100g_payload=0.1 * dm_star_dp,
                dMTOW_dR_kg_per_km=dm_star_dR, km_lost_per_100g=km_per_100g)


# ------------------------------------------------------------------ catalogue snap
def snap(veh, x, R_req=None):
    """Snap a continuous optimum to buildable parts: whole-inch prop, 1-3 identical catalogue packs,
    a catalogue tube, the smallest motor rated >= Tmax. Among all pack x tube combinations that pass
    every structural check (and reach R_req, if given) pick the one with the lowest MTOW."""
    comp = data.components()
    D, E, d, t, V = x
    Dn = round(D / 0.0254) * 0.0254
    best = None
    for _, b in comp["battery"].iterrows():
        for k in (1, 2, 3):
            Ek = k * b.wh
            if Ek < 0.85 * E or Ek > 1.6 * E: continue
            for _, tb in comp["tube"].iterrows():
                xs = np.array([Dn, Ek, tb.outer_dia_mm / 1000, tb.wall_mm / 1000, V])
                o = veh.evaluate(xs)
                if np.any(o["g"] > 1e-9): continue
                if R_req is not None and o["R"] < R_req - 1e-6: continue
                if best is None or o["m"] < best[2]["m"]:
                    best = (f"{k}x {b.part_id}", tb.part_id, o, xs)
    if best is None:
        return dict(prop_in=round(Dn / 0.0254), battery=None, E_Wh=np.nan, tube=None, motor=None, x=None, perf=None)
    motors = comp["motor"].sort_values("max_thrust_n")
    mot = motors[motors.max_thrust_n >= best[2]["Tmax"]]
    mot_id = mot.iloc[0].part_id if len(mot) else "NONE (Tmax exceeds catalogue)"
    return dict(prop_in=round(Dn / 0.0254), battery=best[0], E_Wh=best[3][1], tube=best[1], motor=mot_id, x=best[3], perf=best[2])


# ------------------------------------------------------------------ driver
def summarize(x, o):
    return dict(D_m=x[0], E_Wh=x[1], d_mm=x[2] * 1e3, t_mm=x[3] * 1e3, V_ms=x[4], MTOW_kg=o["m"], R_km=o["R"],
                Tmax_N=o["Tmax"], P_hover_loaded_W=o["Phl"], P_cruise_loaded_W=o["Pcl"], P_cruise_empty_W=o["Pce"],
                FM_loaded=o["FMl"], FM_empty=o["FMe"])


def run(m_plate=0.25, fm=None, tag="m1", make_figs=True, eb=EB, R_req=R_REQ, quick=False):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import pandas as pd
    figd = os.path.join(data.OUT, "figs"); tabd = os.path.join(data.OUT, "tables")
    veh = Vehicle(m_plate=m_plate, fm=fm, eb=eb)
    res = dict(objective="min MTOW s.t. R >= R_req", R_req=R_req, m_plate=m_plate)
    x_mdf, o_mdf, r_mdf = solve_mdf(veh, R_req)
    res["mdf"] = summarize(x_mdf, o_mdf)
    if not quick:
        x_u, o_u = solve_maxR(veh)
        res["uncapped_maxR"] = summarize(x_u, o_u)
        x_idf, m_idf, o_idf, r_idf = solve_idf(veh, R_req)
        res["idf"] = summarize(x_idf, o_idf)
        res["idf_consistency_residual_kg"] = float(m_idf - veh.mass_fn(x_idf, m_idf))
        res["mdf_idf_rel_diff"] = abs(o_mdf["m"] - o_idf["m"]) / o_mdf["m"]
        res["mdf_idf_x_rel_diff"] = float((np.abs(x_mdf - x_idf) / np.abs(x_mdf)).max())
        res["mdf_nfev"], res["idf_nfev"] = int(r_mdf.nfev), int(r_idf.nfev)
        res["kkt"] = kkt_multipliers(veh, x_mdf, R_req)
        # check the multipliers by re-optimising (finite differences of the optimum)
        v2 = Vehicle(payload=veh.p + 0.1, m_plate=m_plate, fm=fm, eb=eb)
        _, o2, _ = solve_mdf(v2, R_req, x0=x_mdf)
        _, o3, _ = solve_mdf(veh, R_req + 0.1, x0=x_mdf)
        dm_p = o2["m"] - o_mdf["m"]; dm_R = (o3["m"] - o_mdf["m"]) / 0.1
        res["reopt_kg_MTOW_per_100g_payload"] = dm_p
        res["reopt_dMTOW_dR_kg_per_km"] = dm_R
        res["reopt_km_lost_per_100g"] = dm_p / dm_R
        # the two objective options of the brief give the same design: max R s.t. MTOW <= m*
        from scipy.optimize import minimize as _min
        def ev(s): return veh.evaluate(to_x(s))
        mstar = o_mdf["m"]
        r = _min(lambda s: -ev(s)["R"] / 50, to_s(x_mdf) * 0.97 + 0.01, jac=lambda s: cs_grad(lambda q: -ev(q)["R"] / 50, s),
                 bounds=[(0, 1)] * 5, method="SLSQP", options=dict(ftol=1e-13, maxiter=500),
                 constraints=[{"type": "ineq", "fun": lambda s: -np.r_[ev(s)["g"], ev(s)["m"] / mstar - 1],
                               "jac": lambda s: -cs_grad(lambda q: np.r_[ev(q)["g"], ev(q)["m"] / mstar - 1], s)}])
        res["equivalence_maxR_at_mstar_km"] = float(ev(r.x)["R"])
        res["equivalence_x_rel_diff"] = float(np.max(np.abs(to_x(r.x) - x_mdf) / np.abs(x_mdf)))
        gc = []
        for label, xx in [("optimum", x_mdf), ("interior", np.array([0.6, 700, 0.025, 0.0012, 14.0]))]:
            s = to_s(xx)
            f = lambda q: (lambda o: np.r_[o["m"], o["R"], o["g"]])(veh.evaluate(to_x(q)))
            Jc = cs_grad(f, s)
            for hh in (1e-4, 1e-7):
                Jf = fd_grad(f, s, hh)
                gc.append(dict(point=label, fd_rel_step=hh, max_abs_err=np.abs(Jf - Jc).max(),
                               max_rel_err=float(np.max(np.abs(Jf - Jc) / (np.abs(Jc) + 1e-3 * np.abs(Jc).max())))))
        res["gradcheck"] = gc
        pd.DataFrame(gc).to_csv(os.path.join(tabd, f"{tag}_gradcheck.csv"), index=False)
    designs = []
    sn = snap(veh, x_mdf, R_req)
    dd = summarize(x_mdf, o_mdf); dd.update(R_req_km=R_req, snap_prop_in=sn["prop_in"], snap_battery=sn["battery"],
                                            snap_tube=sn["tube"], snap_motor=sn["motor"],
                                            snap_R_km=sn["perf"]["R"] if sn["perf"] else np.nan,
                                            snap_MTOW_kg=sn["perf"]["m"] if sn["perf"] else np.nan,
                                            snap_x=json.dumps(list(map(float, sn["x"]))) if sn["x"] is not None else None)
    designs.append(dd)
    dfd = pd.DataFrame(designs); dfd.index = [f"V{i+1}" for i in range(len(dfd))]
    dfd.to_csv(os.path.join(tabd, f"{tag}_designs.csv"))
    json.dump(res, open(os.path.join(tabd, f"{tag}_summary.json"), "w", encoding="utf-8"), indent=1, default=float)
    if quick or not make_figs:
        return res, dfd, None
    # payload-radius trade-off: minimum MTOW to carry payload p over R_req, and each design flown off-design
    pr = []
    for p in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        vp = Vehicle(payload=p, m_plate=m_plate, fm=fm, eb=eb)
        xp, op, _ = solve_mdf(vp, R_req, x0=x_mdf)
        pr.append(dict(payload=p, minMTOW_kg=op["m"], E_Wh=xp[1]))
    pr = pd.DataFrame(pr); pr.to_csv(os.path.join(tabd, f"{tag}_payload_radius.csv"), index=False)
    m_end, hist = veh.mda(x_mdf, m0=3.0, hist=True)
    _, hist2 = veh.mda(x_mdf, m0=40.0, hist=True)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    ax[0].semilogy(np.abs(np.diff(hist)) + 1e-16, "o-", label="start 3 kg")
    ax[0].semilogy(np.abs(np.diff(hist2)) + 1e-16, "s-", label="start 40 kg")
    ax[0].set_xlabel("iteration k"); ax[0].set_ylabel("|m(k+1) - m(k)|  (kg)")
    ax[0].set_title(f"MDA fixed point: contraction 0.008 g = {0.008*G:.4f}"); ax[0].legend(); ax[0].grid(alpha=.3)
    for k, (_, d) in enumerate(dfd.iterrows()):
        xx = np.array([d.D_m, d.E_Wh, d.d_mm / 1e3, d.t_mm / 1e3, d.V_ms])
        ps = np.linspace(0.0, 3.5, 15)
        ax[1].plot(ps, [Vehicle(payload=p, m_plate=m_plate, fm=fm, eb=eb).evaluate(xx)["R"] for p in ps], "-", alpha=.8,
                   label=f"optimum (designed for {d.R_req_km:g} km)")
    ax[1].axvline(2.5, color="grey", lw=.8); ax[1].set_xlabel("payload carried (kg)"); ax[1].set_ylabel("radius (km)")
    ax[1].set_title("Payload-radius diagram of the optimum"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    ax[2].plot(pr.payload, pr.minMTOW_kg, "o-", label=f"min MTOW for R >= {R_req:g} km")
    ax[2].plot([2.5], dfd.snap_MTOW_kg.values[:1], "x", ms=10, label="optimum snapped to catalogue")
    ax[2].set_xlabel("payload (kg)"); ax[2].set_ylabel("minimum MTOW (kg)"); ax[2].set_title("Payload-mass trade-off"); ax[2].legend(); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(figd, f"{tag}_mda_tradeoff.png"), dpi=130); plt.close()
    return res, dfd, pr


if __name__ == "__main__":
    # Baseline M1 (FM = 0.65, plate 0.25 kg). Saves to outputs/*/m1_base_* so the final
    # M1 files used by M4 (m1_designs.csv, m1_summary.json) are NOT overwritten.
    import pandas as pd
    res, dfd, pr = run(m_plate=0.25, tag="m1_base")
    m, i, k = res["mdf"], res["idf"], res["kkt"]
    print(f"\n=== M1 baseline: MINIMISE take-off mass, 2.5 kg payload, radius >= {res['R_req']:g} km ===")
    print(f"MDF : MTOW = {m['MTOW_kg']:.4f} kg | R = {m['R_km']:.2f} km | D = {m['D_m']:.3f} m, E = {m['E_Wh']:.1f} Wh, d = {m['d_mm']:.1f} mm, t = {m['t_mm']:.3f} mm, V = {m['V_ms']:.2f} m/s")
    print(f"IDF : MTOW = {i['MTOW_kg']:.4f} kg   (relative difference {res['mdf_idf_rel_diff']:.1e}, consistency residual {res['idf_consistency_residual_kg']:.1e} kg)")
    print(f"Active constraints: {', '.join(k['active'])}")
    print(f"Payload multiplier: +{k['kg_MTOW_per_100g_payload']:.3f} kg MTOW per extra 100 g payload (re-optimised: {res['reopt_kg_MTOW_per_100g_payload']:.3f})")
    print(f"Radius multiplier : +{k['dMTOW_dR_kg_per_km']:.3f} kg MTOW per extra km (re-optimised: {res['reopt_dMTOW_dR_kg_per_km']:.3f})")
    print(f"=> at fixed MTOW, {k['km_lost_per_100g']:.3f} km of radius lost per extra 100 g (re-optimised: {res['reopt_km_lost_per_100g']:.3f})")
    print(f"Same design from brief's other option (max R s.t. MTOW <= {m['MTOW_kg']:.2f} kg): R = {res['equivalence_maxR_at_mstar_km']:.3f} km, x differs by {res['equivalence_x_rel_diff']:.1e}")
    u = res["uncapped_maxR"]
    print(f"Why not plain max-R: R = {u['R_km']:.2f} km at {u['MTOW_kg']:.1f} kg, E = {u['E_Wh']:.0f} Wh, V = {u['V_ms']:.1f} m/s (all at bounds)")
    print("\nGradient check (complex step vs forward FD):")
    print(pd.DataFrame(res["gradcheck"])[["point", "fd_rel_step", "max_abs_err"]].to_string(index=False))
    print("\nOptimum snapped to catalogue parts:")
    print(dfd[["R_req_km", "MTOW_kg", "R_km", "D_m", "E_Wh", "d_mm", "t_mm", "V_ms", "snap_prop_in", "snap_battery", "snap_tube", "snap_motor", "snap_MTOW_kg", "snap_R_km"]].round(2).to_string())
    print("\nMinimum MTOW vs payload (R >= %g km):" % res["R_req"]); print(pr.round(2).to_string(index=False))
    print("\nSaved: outputs/figs/m1_base_mda_tradeoff.png, outputs/tables/m1_base_*.csv/json")
