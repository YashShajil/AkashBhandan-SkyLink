"""M3 - figure-of-merit surrogate from <= 30 oracle evaluations in <= 3 requests.

Workflow (re-run after each TA reply):
  1. `python -m skylink.m3_surrogate request 1` -> oracle/request_1.csv (15 points)
  2. save TA reply as oracle/response_1.csv (columns D_m,T_N,FM)
  3. `python -m skylink.m3_surrogate request 2` -> adaptive points from the fitted model
  ... same for request 3; then `python -m skylink.m3_surrogate fit`.
If a response file is missing, the MOCK oracle fills it and every output is stamped MOCK.
"""
from __future__ import annotations
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import qmc
from scipy.interpolate import RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as SKRBF, ConstantKernel, WhiteKernel
from . import data, mock_oracle

ORA = os.path.join(data.ROOT, "oracle")
DLO, DHI, TLO, THI = 0.35, 0.80, 8.0, 40.0
ROLL = os.environ.get("ROLL_NO", "<ROLL_NO>")
BUDGET = {1: 15, 2: 8, 3: 7}


def scale(D, T):
    return np.c_[(np.asarray(D) - DLO) / (DHI - DLO), (np.asarray(T) - TLO) / (THI - TLO)]


# ------------------------------------------------------------------ requests
def request_1(seed=7):
    """13-point maximin Latin hypercube over the whole oracle box + the centre of our region of
    interest (D=0.75 m, T=24 N) sent TWICE.
    Why: no prior knowledge of FM(D,T) shape, so a space-filling design gives the global trend
    and can reveal non-smooth regions; LHS keeps every D and T level distinct (1-D trends are
    visible); maximin spreads points so kriging length-scales are identifiable. The duplicated
    point measures oracle noise directly (costs one point; tells us whether to interpolate or smooth)."""
    best, bd = None, -1
    for s in range(300):
        X = qmc.LatinHypercube(d=2, seed=seed * 1000 + s).random(13)
        d = min(np.linalg.norm(X[i] - X[j]) for i in range(13) for j in range(i + 1, 13))
        if d > bd: best, bd = X, d
    D = np.r_[DLO + best[:, 0] * (DHI - DLO), 0.75, 0.75]; T = np.r_[TLO + best[:, 1] * (THI - TLO), 24.0, 24.0]
    return pd.DataFrame(dict(student_id=ROLL, D_m=D.round(4), T_N=T.round(2)))


def load_responses(upto=3):
    frames, mock = [], False
    for k in range(1, upto + 1):
        req = os.path.join(ORA, f"request_{k}.csv"); resp = os.path.join(ORA, f"response_{k}.csv")
        if not os.path.exists(req):
            break
        q = pd.read_csv(req)
        if os.path.exists(resp):
            r = pd.read_csv(resp); r["source"] = "oracle"
        else:
            r = q[["D_m", "T_N"]].copy(); r["FM"] = mock_oracle.query(r.D_m.values, r.T_N.values, seed=k); r["source"] = "MOCK"
            mock = True
        r["request"] = k; frames.append(r)
    df = pd.concat(frames, ignore_index=True)
    # data checks on the oracle reply itself
    bad = (df.FM <= 0) | (df.FM >= 1) | df.FM.isna()
    if bad.any():
        data.log("oracle", f"{int(bad.sum())} FM values outside (0,1) or missing", "excluded from fit")
        df = df[~bad]
    return df, mock


# ------------------------------------------------------------------ models
class Quad:
    name = "quadratic RS"
    def fit(self, X, y):
        self.c, *_ = np.linalg.lstsq(self._P(X), y, rcond=None); return self
    def _P(self, X):
        x1, x2 = X[:, 0], X[:, 1]; return np.c_[np.ones_like(x1), x1, x2, x1 * x1, x1 * x2, x2 * x2]
    def predict(self, X, std=False):
        m = self._P(X) @ self.c; return (m, np.zeros(len(X))) if std else m


class RBFm:
    name = "RBF (thin-plate, smoothed)"
    def __init__(self, smoothing=1e-4): self.s = smoothing
    def fit(self, X, y):
        self.f = RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=self.s, degree=1); return self
    def predict(self, X, std=False):
        m = self.f(X); return (m, np.zeros(len(X))) if std else m


class Krig:
    name = "kriging + nugget"
    def fit(self, X, y):
        k = ConstantKernel(1.0, (1e-3, 1e2)) * SKRBF([0.3, 0.3], (0.02, 10.0)) + WhiteKernel(1e-4, (1e-8, 1e-1))
        self.g = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=8, random_state=0).fit(X, y)
        p = self.g.kernel_.get_params()
        self.c = p["k1__k1__constant_value"]; self.ell = np.asarray(p["k1__k2__length_scale"]); self.nug = p["k2__noise_level"]
        self.X = X; self.alpha = self.g.alpha_; self.ym, self.ys = self.g._y_train_mean, self.g._y_train_std
        return self
    def predict(self, X, std=False):
        m, s = self.g.predict(X, return_std=True)
        return (m, s) if std else m
    def predict_cs(self, X):
        """complex-step-safe predictive mean (for M1 gradients)"""
        d2 = (((X[:, None, :] - self.X[None, :, :]) / self.ell) ** 2).sum(-1)
        return self.ym + self.ys * (self.c * np.exp(-0.5 * d2)) @ self.alpha
    def noise_sd(self):
        return float(np.sqrt(self.nug) * self.ys)


def loocv(model_factory, X, y):
    e = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        e[i] = model_factory().fit(X[m], y[m]).predict(X[[i]])[0] - y[i]
    return e


# ------------------------------------------------------------------ adaptive sampling
def roi_weight(D, T):
    """region of interest: rotors and thrusts our M1 designs actually use (D>=0.6 m, 9-36 N)"""
    return np.where((D >= 0.6) & (T >= 9) & (T <= 36), 1.0, 0.35)


def adaptive(df, n, extra_targets=()):
    X = scale(df.D_m, df.T_N); y = df.FM.values
    g = np.linspace(0, 1, 61); GX = np.array([[a, b] for a in g for b in g])
    Dg = DLO + GX[:, 0] * (DHI - DLO); Tg = TLO + GX[:, 1] * (THI - TLO)
    pts = [scale([d], [t])[0] for d, t in extra_targets]
    Xa, ya = X.copy(), y.copy()
    for p in pts:   # kriging believer: pretend we observed the mean
        k = Krig().fit(Xa, ya); Xa = np.vstack([Xa, p]); ya = np.r_[ya, k.predict(p[None])[0]]
    # gradient magnitude of current fit -> flags steep / non-smooth zones
    while len(pts) < n:
        k = Krig().fit(Xa, ya)
        m, s = k.predict(GX, std=True)
        gm = np.hypot(*np.gradient(m.reshape(61, 61)))
        crit = s * roi_weight(Dg, Tg) * (1 + 3 * gm.ravel() / (gm.max() + 1e-12))
        dmin = np.min(np.linalg.norm(GX[:, None] - Xa[None], axis=2), axis=1)
        crit[dmin < 0.06] = 0
        j = int(np.argmax(crit)); pts.append(GX[j]); Xa = np.vstack([Xa, GX[j]]); ya = np.r_[ya, m[j]]
    P = np.array(pts)
    return pd.DataFrame(dict(student_id=ROLL, D_m=(DLO + P[:, 0] * (DHI - DLO)).round(4), T_N=(TLO + P[:, 1] * (THI - TLO)).round(2)))


def design_targets():
    p = os.path.join(data.OUT, "tables", "m1_designs.csv")
    if not os.path.exists(p): return []
    d = pd.read_csv(p, index_col=0); out = []
    for _, r in d.iterrows():
        out += [(min(r.D_m, DHI), r.MTOW_kg * 9.81 / 4), (min(r.D_m, DHI), (r.MTOW_kg - 2.5) * 9.81 / 4)]
    return out


# ------------------------------------------------------------------ fit + report
def fit_all(make_figs=True):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    df, mock = load_responses()
    stamp = " [MOCK ORACLE]" if mock else ""
    X = scale(df.D_m, df.T_N); y = df.FM.values
    # noise from replicates
    rep = df.groupby(["D_m", "T_N"]).FM.agg(["count", "std"]).dropna()
    rep_sd = float(np.sqrt((rep["std"] ** 2).mean())) if len(rep) else np.nan
    # stall detection: residual of a smooth fit vs disk loading
    DL = df.T_N / (np.pi * df.D_m ** 2 / 4)
    facs = {"kriging + nugget": Krig, "RBF (thin-plate, smoothed)": lambda: RBFm(1e-4), "quadratic RS": Quad}
    rows, errs = [], {}
    for nm, f in facs.items():
        e = loocv(f, X, y); errs[nm] = e
        rows.append(dict(model=nm, LOOCV_RMSE=float(np.sqrt((e ** 2).mean())), LOOCV_max=float(np.abs(e).max()),
                         LOOCV_RMSE_ROI=float(np.sqrt((e[roi_weight(df.D_m.values, df.T_N.values) == 1] ** 2).mean()))))
    tab = pd.DataFrame(rows).sort_values("LOOCV_RMSE")
    # handling the non-smooth region: points with high disk loading
    k_all = Krig().fit(X, y)
    e_k = errs["kriging + nugget"]; DLv = DL.values
    sig = max(1.4826 * np.median(np.abs(e_k - np.median(e_k))), k_all.noise_sd(), 1e-3)
    big = np.abs(e_k) > 4 * sig
    hi = big & (DLv > np.median(DLv))        # failures that cluster at high disk loading
    DL_thr = float(DLv[hi].min() * 0.85) if hi.any() else np.inf
    # trusted model = fit only on points below the threshold (smooth regime); M1 gets a constraint DL <= DL_thr
    keep = DLv < DL_thr if np.isfinite(DL_thr) else np.ones(len(df), bool)
    if keep.sum() < 10:
        keep = np.ones(len(df), bool); DL_thr = np.inf
    best_name = tab.model.iloc[0]
    k_trust = Krig().fit(X[keep], y[keep])
    e_trust = loocv(Krig, X[keep], y[keep])
    tab = pd.concat([tab, pd.DataFrame([dict(model="kriging + nugget, smooth regime only (DL < %.0f N/m2)" % DL_thr if np.isfinite(DL_thr) else "kriging (all)",
                                             LOOCV_RMSE=float(np.sqrt((e_trust ** 2).mean())), LOOCV_max=float(np.abs(e_trust).max()), LOOCV_RMSE_ROI=np.nan)])])
    tab["data"] = "MOCK" if mock else "oracle"
    tab.to_csv(os.path.join(data.OUT, "tables", "m3_loocv.csv"), index=False)
    if make_figs:
        g = np.linspace(0, 1, 81); GX = np.array([[a, b] for a in g for b in g])
        m, s = k_trust.predict(GX, std=True)
        Dg = DLO + GX[:, 0] * (DHI - DLO); Tg = TLO + GX[:, 1] * (THI - TLO)
        fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
        c0 = ax[0].tricontourf(Dg, Tg, m, 20, cmap="viridis"); plt.colorbar(c0, ax=ax[0])
        sc = ax[0].scatter(df.D_m, df.T_N, c=df.request, cmap="autumn", edgecolor="k", s=30)
        if np.isfinite(DL_thr):
            dd = np.linspace(DLO, DHI, 50); ax[0].plot(dd, DL_thr * np.pi * dd ** 2 / 4, "w--", lw=2, label=f"trust limit DL={DL_thr:.0f}")
            ax[0].legend(fontsize=7, loc="upper left")
        ax[0].set_ylim(TLO, THI); ax[0].set_xlabel("D (m)"); ax[0].set_ylabel("T (N)"); ax[0].set_title("FM kriging mean" + stamp, fontsize=9)
        c1 = ax[1].tricontourf(Dg, Tg, s, 20, cmap="magma"); plt.colorbar(c1, ax=ax[1]); ax[1].scatter(df.D_m, df.T_N, c="w", s=8)
        ax[1].set_title("kriging std (uncertainty)"); ax[1].set_xlabel("D (m)")
        for nm, e in errs.items():
            ax[2].scatter(y, y + e, s=18, label=nm)
        ax[2].plot([y.min(), y.max()], [y.min(), y.max()], "k-"); ax[2].set_xlabel("oracle FM"); ax[2].set_ylabel("LOOCV prediction"); ax[2].legend(fontsize=7); ax[2].set_title("Parity (leave-one-out)")
        ax[3].scatter(DL, errs["kriging + nugget"], c=big, cmap="coolwarm"); ax[3].axhline(0, color="k")
        ax[3].set_xlabel("disk loading T/A (N/m2)"); ax[3].set_ylabel("LOOCV error (kriging)"); ax[3].set_title("Where a smooth model fails")
        if np.isfinite(DL_thr): ax[3].axvline(DL_thr, color="r", ls="--")
        plt.tight_layout(); plt.savefig(os.path.join(data.OUT, "figs", "m3_surrogate.png"), dpi=120); plt.close()
    summ = dict(mock=mock, n_points=len(df), replicate_noise_sd=rep_sd, kriging_nugget_sd=k_all.noise_sd(),
                best_by_loocv=best_name, DL_trust_limit=DL_thr, n_excluded_nonsmooth=int((~keep).sum()))
    json.dump(summ, open(os.path.join(data.OUT, "tables", "m3_summary.json"), "w"), indent=1, default=float)
    return k_trust, summ, tab


def fm_function(k):
    """FM(D,T) callable for M1 (complex-step safe). Clamped to the oracle box by smooth extension."""
    def fm(D, T):
        X = np.array([[(D - DLO) / (DHI - DLO), (T - TLO) / (THI - TLO)]], dtype=complex if np.iscomplexobj(D) or np.iscomplexobj(T) else float)
        return k.predict_cs(X)[0]
    return fm


if __name__ == "__main__":
    os.makedirs(ORA, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if cmd == "request":
        k = int(sys.argv[2])
        if k == 1:
            df = request_1()
        else:
            dfr, mock = load_responses(k - 1)
            df = adaptive(dfr, BUDGET[k], extra_targets=design_targets()[:3] if k == 3 else ())
        df.to_csv(os.path.join(ORA, f"request_{k}.csv"), index=False); print(df)
    else:
        k, s, t = fit_all(); print(s); print(t)
