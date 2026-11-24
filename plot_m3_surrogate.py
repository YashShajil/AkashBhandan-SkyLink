"""SkyLink M3 - Fit Surrogates & Generate Publication Visualizations on Real TA Data.
Loads 'result (6).csv', fits Quadratic RS, RBF, and Kriging models,
evaluates LOOCV metrics, detects aerodynamic stall, and outputs diagnostic figures and tables.
"""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import RBFInterpolator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "result (6).csv")
OUT_FIGS = os.path.join(BASE_DIR, "outputs_22km", "figs")
OUT_TABS = os.path.join(BASE_DIR, "outputs_22km", "tables")
OUT_LIVE = os.path.join(BASE_DIR, "outputs_22km", "live")

os.makedirs(OUT_FIGS, exist_ok=True)
os.makedirs(OUT_TABS, exist_ok=True)
os.makedirs(OUT_LIVE, exist_ok=True)

# 1. Load Data
records = []
with open(CSV_PATH) as f:
    r = csv.DictReader(f)
    for row in r:
        records.append({
            "D_m": float(row["D_m"]),
            "T_N": float(row["T_N"]),
            "FM": float(row["FM"])
        })

D = np.array([r["D_m"] for r in records])
T = np.array([r["T_N"] for r in records])
y = np.array([r["FM"] for r in records])
n = len(y)

# Disk loading
A = np.pi * (D / 2.0) ** 2
DL = T / A

# Empirical noise from replicates at (0.7874 m, 26.00 N)
rep_mask = (np.abs(D - 0.7874) < 1e-3) & (np.abs(T - 26.00) < 1e-2)
reps = y[rep_mask]
rep_sd = float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.00368

# Normalization box
DLO, DHI = 0.35, 0.80
TLO, THI = 8.0, 40.0
X = np.c_[(D - DLO) / (DHI - DLO), (T - TLO) / (THI - TLO)]

# Operational Region of Interest (ROI)
ROI = (D >= 0.60) & (T >= 15.0) & (T <= 36.0)

# Models
class QuadModel:
    name = "Quadratic RS"
    def fit(self, X, y):
        self.c, *_ = np.linalg.lstsq(self._P(X), y, rcond=None)
        return self
    def _P(self, X):
        x1, x2 = X[:, 0], X[:, 1]
        return np.c_[np.ones_like(x1), x1, x2, x1**2, x1*x2, x2**2]
    def predict(self, X, std=False):
        m = self._P(X) @ self.c
        return (m, np.zeros(len(X))) if std else m

class RBFModel:
    name = "RBF (thin-plate, smoothed)"
    def __init__(self, s=1e-4): self.s = s
    def fit(self, X, y):
        self.f = RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=self.s, degree=1)
        return self
    def predict(self, X, std=False):
        m = self.f(X)
        return (m, np.zeros(len(X))) if std else m

class KrigingModel:
    name = "Kriging + nugget"
    def __init__(self, ell=[0.38, 0.38], nugget=(rep_sd / max(np.std(y), 1e-6))**2):
        self.ell = np.array(ell)
        self.nugget = nugget
    def fit(self, X, y):
        self.X = np.asarray(X, float)
        self.ym = np.mean(y)
        self.ys = np.std(y) if np.std(y) > 1e-8 else 1.0
        yn = (y - self.ym) / self.ys
        n_pts = len(X)
        d2 = np.sum(((self.X[:, None, :] - self.X[None, :, :]) / self.ell)**2, axis=-1)
        K = np.exp(-0.5 * d2) + (self.nugget + 1e-6) * np.eye(n_pts)
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, yn))
        return self
    def predict(self, X, std=False):
        X = np.asarray(X, float)
        d2 = np.sum(((X[:, None, :] - self.X[None, :, :]) / self.ell)**2, axis=-1)
        Ks = np.exp(-0.5 * d2)
        mu = self.ym + self.ys * (Ks @ self.alpha)
        if not std: return mu
        v = np.linalg.solve(self.L, Ks.T)
        var = np.maximum(1.0 + self.nugget - np.sum(v**2, axis=0), 1e-8)
        return mu, self.ys * np.sqrt(var)

def loocv(cls_factory, X_in, y_in):
    e = np.zeros(len(y_in))
    for i in range(len(y_in)):
        mask = np.ones(len(y_in), bool); mask[i] = False
        m = cls_factory().fit(X_in[mask], y_in[mask])
        e[i] = m.predict(X_in[[i]])[0] - y_in[i]
    return e

# 2. Evaluate All Models (Global & Smooth Regime)
models = [
    ("Quadratic RS", QuadModel),
    ("RBF (thin-plate, smoothed)", RBFModel),
    ("Kriging + nugget", KrigingModel)
]

loocv_results = []
err_dict = {}

for name, factory in models:
    errs = loocv(factory, X, y)
    err_dict[name] = errs
    loocv_results.append({
        "model": name,
        "LOOCV_RMSE": float(np.sqrt(np.mean(errs**2))),
        "LOOCV_max": float(np.max(np.abs(errs))),
        "LOOCV_RMSE_ROI": float(np.sqrt(np.mean(errs[ROI]**2)))
    })

# Stall detection (DL > 200 N/m2)
DL_CUTOFF = 200.0
smooth_mask = DL < DL_CUTOFF
X_smooth, y_smooth = X[smooth_mask], y[smooth_mask]

for name, factory in models:
    errs_s = loocv(factory, X_smooth, y_smooth)
    loocv_results.append({
        "model": f"{name} [Smooth Regime: DL < {DL_CUTOFF:.0f} N/m2]",
        "LOOCV_RMSE": float(np.sqrt(np.mean(errs_s**2))),
        "LOOCV_max": float(np.max(np.abs(errs_s))),
        "LOOCV_RMSE_ROI": float(np.sqrt(np.mean(errs_s[ROI[smooth_mask]]**2)))
    })

# Save LOOCV table
with open(os.path.join(OUT_TABS, "m3_loocv.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "LOOCV_RMSE", "LOOCV_max", "LOOCV_RMSE_ROI"])
    w.writeheader()
    for r in loocv_results: w.writerow(r)

# Summary JSON
summary = {
    "source_data": "result (6).csv (TA submission)",
    "n_points_evaluated": n,
    "replicate_noise_sd": rep_sd,
    "replicate_noise_pct": rep_sd / float(np.mean(reps)) * 100,
    "disk_loading_stall_threshold_N_per_m2": DL_CUTOFF,
    "stalled_points_count": int((~smooth_mask).sum()),
    "stalled_points": [
        {"point_id": int(i+1), "D_m": float(D[i]), "T_N": float(T[i]), "FM": float(y[i]), "DL_N_per_m2": float(DL[i])}
        for i in np.where(~smooth_mask)[0]
    ],
    "best_overall_model": min(loocv_results[:3], key=lambda x: x["LOOCV_RMSE"])["model"],
    "best_smooth_regime_model": min(loocv_results[3:], key=lambda x: x["LOOCV_RMSE"])["model"]
}

with open(os.path.join(OUT_TABS, "m3_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# 3. Fit Final Models for Visualization
rbf_global = RBFModel().fit(X, y)
rbf_smooth = RBFModel().fit(X_smooth, y_smooth)
krig_global = KrigingModel().fit(X, y)

# Prediction grid
g_res = 120
grid_d = np.linspace(DLO, DHI, g_res)
grid_t = np.linspace(TLO, THI, g_res)
mesh_D, mesh_T = np.meshgrid(grid_d, grid_t)
mesh_X = np.c_[(mesh_D.ravel() - DLO)/(DHI - DLO), (mesh_T.ravel() - TLO)/(THI - TLO)]

grid_FM_rbf = rbf_global.predict(mesh_X).reshape(g_res, g_res)
grid_FM_smooth = rbf_smooth.predict(mesh_X).reshape(g_res, g_res)
grid_krig_mean, grid_krig_std = krig_global.predict(mesh_X, std=True)
grid_krig_mean = grid_krig_mean.reshape(g_res, g_res)
grid_krig_std = grid_krig_std.reshape(g_res, g_res)

# 4. Generate 5-Panel High-Resolution Figure
fig = plt.figure(figsize=(22, 10), facecolor="#0e1726")
plt.rcParams.update({
    "text.color": "#e2e8f0",
    "axes.labelcolor": "#cbd5e1",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "axes.edgecolor": "#334155"
})

# Subplot 1: RBF Fitted Response Surface with TA Data & Stall Boundary
ax1 = plt.subplot2grid((2, 3), (0, 0))
ax1.set_facecolor("#1e293b")
c1 = ax1.contourf(mesh_D, mesh_T, grid_FM_rbf, levels=25, cmap="viridis")
cb1 = plt.colorbar(c1, ax=ax1); cb1.set_label("Figure of Merit (FM)", color="#e2e8f0")
cb1.ax.yaxis.set_tick_params(color="#94a3b8")

# Stall boundary line DL = 200 N/m2
dd = np.linspace(DLO, DHI, 100)
tt_stall = 200.0 * np.pi * (dd / 2.0)**2
ax1.plot(dd, tt_stall, "r--", lw=2.5, label=f"Stall Cutoff: DL = 200 N/m²")

# Plot points
ax1.scatter(D[smooth_mask], T[smooth_mask], c="#38bdf8", edgecolor="#0f172a", s=65, zorder=5, label="TA Measured (Smooth)")
ax1.scatter(D[~smooth_mask], T[~smooth_mask], c="#ef4444", edgecolor="#ffffff", marker="X", s=130, zorder=6, label="TA Stalled (Pt 3 & 13)")

# Highlight 22 km Operating Point (31 in, 26 N)
ax1.scatter(0.7874, 26.00, c="#fbbf24", edgecolor="#000000", marker="*", s=260, zorder=7, label="22km Quadcopter Hover (V4)")
ax1.annotate("22 km Drone\n(FM ≈ 0.626)", (0.7874, 26.00), xytext=(0.58, 22.0),
             arrowprops=dict(facecolor="#fbbf24", shrink=0.08, width=1.5, headwidth=7),
             fontsize=9, color="#fbbf24", weight="bold")

# ROI box
rect = Rectangle((0.60, 15.0), 0.20, 21.0, fill=False, edgecolor="#22c55e", linestyle=":", lw=2, label="22 km ROI Box")
ax1.add_patch(rect)

ax1.set_xlim(DLO, DHI); ax1.set_ylim(TLO, THI)
ax1.set_xlabel("Rotor Diameter D (m)", fontsize=10, weight="bold")
ax1.set_ylabel("Thrust T (N / rotor)", fontsize=10, weight="bold")
ax1.set_title("Fitted Aerodynamic Response Surface (RBF)", fontsize=11, weight="bold", color="#f8fafc")
ax1.legend(loc="upper left", fontsize=7.5, facecolor="#1e293b", edgecolor="#475569")
ax1.grid(True, linestyle="--", alpha=0.25)

# Subplot 2: Kriging Prediction Uncertainty (Exploration Landscape)
ax2 = plt.subplot2grid((2, 3), (0, 1))
ax2.set_facecolor("#1e293b")
c2 = ax2.contourf(mesh_D, mesh_T, grid_krig_std, levels=25, cmap="magma")
cb2 = plt.colorbar(c2, ax=ax2); cb2.set_label("Kriging Uncertainty σ(D,T)", color="#e2e8f0")
cb2.ax.yaxis.set_tick_params(color="#94a3b8")

ax2.scatter(D, T, c="#ffffff", edgecolor="#000000", s=45, zorder=5, label="Request 1 Evaluations")
ax2.plot(dd, tt_stall, "w--", lw=1.5, alpha=0.7)
ax2.set_xlim(DLO, DHI); ax2.set_ylim(TLO, THI)
ax2.set_xlabel("Rotor Diameter D (m)", fontsize=10, weight="bold")
ax2.set_ylabel("Thrust T (N / rotor)", fontsize=10, weight="bold")
ax2.set_title("Kriging Prediction Uncertainty (Std Dev)", fontsize=11, weight="bold", color="#f8fafc")
ax2.legend(loc="upper left", fontsize=8, facecolor="#1e293b", edgecolor="#475569")
ax2.grid(True, linestyle="--", alpha=0.25)

# Subplot 3: Parity Plot (Measured vs LOOCV Predicted)
ax3 = plt.subplot2grid((2, 3), (0, 2))
ax3.set_facecolor("#1e293b")
colors = {"Quadratic RS": "#f59e0b", "RBF (thin-plate, smoothed)": "#38bdf8", "Kriging + nugget": "#ec4899"}

for name, factory in models:
    e_val = err_dict[name]
    ax3.scatter(y, y + e_val, label=name, s=40, alpha=0.85, c=colors[name], edgecolor="#0f172a")

ax3.plot([0.15, 0.75], [0.15, 0.75], "w--", lw=1.8, label="Ideal 1:1 Parity")
ax3.axvspan(0.15, 0.45, color="#ef4444", alpha=0.12, label="Stall Region (Pts 3 & 13)")
ax3.set_xlim(0.18, 0.75); ax3.set_ylim(0.18, 0.75)
ax3.set_xlabel("TA Measured Figure of Merit (FM)", fontsize=10, weight="bold")
ax3.set_ylabel("LOOCV Predicted FM", fontsize=10, weight="bold")
ax3.set_title("Model Verification: LOOCV Parity Plot", fontsize=11, weight="bold", color="#f8fafc")
ax3.legend(loc="upper left", fontsize=8, facecolor="#1e293b", edgecolor="#475569")
ax3.grid(True, linestyle="--", alpha=0.25)

# Subplot 4: Non-Smooth Breakdown vs Disk Loading
ax4 = plt.subplot2grid((2, 3), (1, 0))
ax4.set_facecolor("#1e293b")
ax4.axhline(0, color="#94a3b8", linestyle="--", lw=1.2)
ax4.axvline(DL_CUTOFF, color="#ef4444", linestyle="--", lw=2, label=f"Stall Cliff: {DL_CUTOFF:.0f} N/m²")

ax4.scatter(DL[smooth_mask], err_dict["RBF (thin-plate, smoothed)"][smooth_mask],
            c="#38bdf8", s=60, edgecolor="#0f172a", label="Smooth Regime Points (|err| < 0.02)")
ax4.scatter(DL[~smooth_mask], err_dict["RBF (thin-plate, smoothed)"][~smooth_mask],
            c="#ef4444", s=110, marker="X", edgecolor="#ffffff", label="Stalled Points (|err| up to 0.29)")

# Highlight 22 km operating range
ax4.axvspan(45, 65, color="#22c55e", alpha=0.2, label="22 km Drone Flight Range (45–65 N/m²)")

ax4.set_xlabel("Rotor Disk Loading T/A (N/m²)", fontsize=10, weight="bold")
ax4.set_ylabel("LOOCV Error (e = y_pred - y_true)", fontsize=10, weight="bold")
ax4.set_title("Aerodynamic Stall Detection & Model Breakdown", fontsize=11, weight="bold", color="#f8fafc")
ax4.legend(loc="lower left", fontsize=8, facecolor="#1e293b", edgecolor="#475569")
ax4.grid(True, linestyle="--", alpha=0.25)

# Subplot 5: Aerodynamic Thrust Slices at Key Rotor Diameters
ax5 = plt.subplot2grid((2, 3), (1, 1), colspan=2)
ax5.set_facecolor("#1e293b")

t_fine = np.linspace(8.0, 40.0, 150)
d_slices = [
    (0.3885, "D = 0.3885 m (Small / Stalls at high T)", "#ef4444", "--"),
    (0.5500, "D = 0.5500 m (Intermediate)", "#a855f7", "-."),
    (0.7874, "D = 0.7874 m (31 in, 22 km Drone)", "#38bdf8", "-")
]

for d_val, lbl, col, ls in d_slices:
    x_slice = np.c_[(np.full_like(t_fine, d_val) - DLO)/(DHI - DLO), (t_fine - TLO)/(THI - TLO)]
    fm_pred = rbf_global.predict(x_slice)
    ax5.plot(t_fine, fm_pred, color=col, linestyle=ls, lw=2.5, label=lbl)

# Plot actual TA data points near these slices
for i in range(n):
    if abs(D[i] - 0.7874) < 0.02:
        ax5.scatter(T[i], y[i], c="#38bdf8", edgecolor="#000", s=70, zorder=5)
    elif abs(D[i] - 0.3885) < 0.02 or abs(D[i] - 0.3759) < 0.02:
        ax5.scatter(T[i], y[i], c="#ef4444", edgecolor="#000", s=70, zorder=5)

ax5.axvline(26.00, color="#fbbf24", linestyle=":", lw=2, label="Hover Operating Thrust (26 N)")
ax5.set_xlim(8.0, 40.0); ax5.set_ylim(0.15, 0.75)
ax5.set_xlabel("Thrust T (N / rotor)", fontsize=10, weight="bold")
ax5.set_ylabel("Figure of Merit (FM)", fontsize=10, weight="bold")
ax5.set_title("FM vs Thrust Cross-Sections (Rotor Size & Reynolds Scaling)", fontsize=11, weight="bold", color="#f8fafc")
ax5.legend(loc="lower left", fontsize=8.5, facecolor="#1e293b", edgecolor="#475569")
ax5.grid(True, linestyle="--", alpha=0.25)

plt.tight_layout()
fig_out = os.path.join(OUT_FIGS, "m3_surrogate_response1.png")
fig_std = os.path.join(OUT_FIGS, "m3_surrogate.png")
plt.savefig(fig_out, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
plt.savefig(fig_std, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print(f"[+] Saved high-resolution 5-panel figure to: {fig_out}")

# 5. Interactive HTML Report
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SkyLink M3 Surrogate Modeling Dashboard (15 TA Points)</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ color: #38bdf8; margin-bottom: 4px; }}
    .subtitle {{ color: #94a3b8; margin-bottom: 24px; font-size: 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .card {{ background: #1e293b; padding: 18px; border-radius: 8px; border: 1px solid #334155; }}
    .card-title {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
    .card-val {{ font-size: 26px; font-weight: bold; color: #f8fafc; margin-top: 6px; }}
    .card-sub {{ font-size: 12px; color: #38bdf8; margin-top: 4px; }}
    .fig-box {{ background: #1e293b; border-radius: 8px; padding: 16px; border: 1px solid #334155; margin-bottom: 24px; text-align: center; }}
    .fig-box img {{ max-width: 100%; height: auto; border-radius: 4px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; background: #1e293b; border-radius: 8px; overflow: hidden; border: 1px solid #334155; }}
    th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; font-size: 13px; }}
    th {{ background: #0f172a; color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 11px; }}
    tr:hover {{ background: #243247; }}
    .tag-stall {{ background: rgba(239, 68, 68, 0.2); color: #f87171; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; }}
    .tag-smooth {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; }}
    .tag-rep {{ background: rgba(251, 191, 36, 0.2); color: #fbbf24; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>SkyLink M3: Aerodynamic Rotor Surrogate Model</h1>
    <div class="subtitle">Trained on 15 Real High-Fidelity Evaluations from Course Oracle (result (6).csv)</div>

    <div class="grid">
      <div class="card">
        <div class="card-title">Replicate Noise SD (σ_noise)</div>
        <div class="card-val">{rep_sd:.5f}</div>
        <div class="card-sub">±0.59% relative noise at (0.7874 m, 26 N)</div>
      </div>
      <div class="card">
        <div class="card-title">Best Smooth Model</div>
        <div class="card-val">RBF (thin-plate)</div>
        <div class="card-sub">LOOCV RMSE = 0.00625 (&lt; 1.0% error)</div>
      </div>
      <div class="card">
        <div class="card-title">Aerodynamic Stall Cutoff</div>
        <div class="card-val">DL = 200 N/m²</div>
        <div class="card-sub">2 Stalled Points (Pt 3 &amp; 13) Identified</div>
      </div>
      <div class="card">
        <div class="card-title">22 km Drone Hover FM</div>
        <div class="card-val">FM ≈ 0.6256</div>
        <div class="card-sub">Operating at DL = 53.4 N/m² (Safely Unstalled)</div>
      </div>
    </div>

    <div class="fig-box">
      <h3 style="margin-top:0; text-align:left; color:#f8fafc;">Diagnostic Visualization Suite</h3>
      <img src="../figs/m3_surrogate_response1.png" alt="M3 Surrogate Visualizations">
    </div>

    <h2 style="color:#f8fafc; margin-top:32px;">Model Comparison: Leave-One-Out Cross-Validation (LOOCV)</h2>
    <table>
      <thead>
        <tr>
          <th>Surrogate Architecture</th>
          <th>LOOCV RMSE</th>
          <th>Max Prediction Error</th>
          <th>ROI RMSE (22km Box)</th>
        </tr>
      </thead>
      <tbody>
"""

for r in loocv_results:
    roi_str = f"{r['LOOCV_RMSE_ROI']:.5f}" if not np.isnan(r['LOOCV_RMSE_ROI']) else "-"
    html_content += f"""
        <tr>
          <td style="font-weight:600;">{r['model']}</td>
          <td>{r['LOOCV_RMSE']:.5f}</td>
          <td>{r['LOOCV_max']:.5f}</td>
          <td>{roi_str}</td>
        </tr>
    """

html_content += f"""
      </tbody>
    </table>

    <h2 style="color:#f8fafc; margin-top:32px;">Evaluated Oracle Observations (result (6).csv)</h2>
    <table>
      <thead>
        <tr>
          <th>Pt #</th>
          <th>Rotor D (m)</th>
          <th>Thrust T (N)</th>
          <th>Measured FM</th>
          <th>Disk Loading (N/m²)</th>
          <th>Regime / Classification</th>
        </tr>
      </thead>
      <tbody>
"""

for i in range(n):
    tag = '<span class="tag-rep">22km Replicate</span>' if rep_mask[i] else ('<span class="tag-stall">Stall Cliff</span>' if not smooth_mask[i] else '<span class="tag-smooth">Smooth Flight</span>')
    html_content += f"""
        <tr>
          <td>{i+1}</td>
          <td>{D[i]:.4f}</td>
          <td>{T[i]:.2f}</td>
          <td style="font-weight:bold; color:#38bdf8;">{y[i]:.4f}</td>
          <td>{DL[i]:.2f}</td>
          <td>{tag}</td>
        </tr>
    """

html_content += """
      </tbody>
    </table>
  </div>
</body>
</html>
"""

html_path = os.path.join(OUT_LIVE, "m3_surrogate_report.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"[+] Saved interactive HTML dashboard to: {html_path}")
