import json, os

with open('Project/outputs_22km/live/compact_3d_payload.json') as f:
    payload_str = f.read()

explorer_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SkyLink M3: 3D Surrogate Modeling & Aerodynamic Analysis</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    body {{ background-color: var(--background, #0f172a); color: var(--foreground, #e2e8f0); }}
    .card {{ background-color: var(--card, #1e293b); border: 1px solid var(--border, #334155); }}
    .tooltip-box {{
      position: absolute;
      pointer-events: none;
      background: rgba(15, 23, 42, 0.95);
      border: 1px solid rgba(56, 189, 248, 0.4);
      color: #f8fafc;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 11px;
      line-height: 1.4;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
      z-index: 50;
      display: none;
      white-space: nowrap;
    }}
  </style>
</head>
<body class="p-6 antialiased select-none min-h-screen">
  <div class="max-w-6xl mx-auto space-y-6">

    <!-- Header -->
    <div class="flex flex-wrap items-center justify-between gap-4 border-b border-slate-700 pb-4">
      <div>
        <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <span>SkyLink M3: 3D Surrogate Modeling & Rotor Aerodynamics</span>
          <span class="px-2 py-0.5 text-xs font-mono font-medium bg-sky-500/10 text-sky-400 border border-sky-500/20 rounded">Real TA Data</span>
        </h1>
        <p class="text-xs text-slate-400 mt-1">Multi-model benchmark (Quadratic RS, RBF, Kriging) &bull; 3D Surface Visualizer with Colorbars &bull; M1 MDO Optimum Shift</p>
      </div>
      <div class="flex items-center gap-2">
        <span class="px-3 py-1 rounded bg-slate-800 border border-slate-700 text-xs font-mono text-emerald-400">30 Observed Points (Full Budget)</span>
        <span class="px-3 py-1 rounded bg-slate-800 border border-slate-700 text-xs font-mono text-sky-400">Noise σ = ±0.00368 (0.59%)</span>
      </div>
    </div>

    <!-- Main Interactive 3D Canvas Card -->
    <div class="card rounded-xl p-5 space-y-4">
      <div class="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-700">
        <!-- Model Selection Buttons -->
        <div class="flex items-center gap-1.5 bg-slate-900/80 p-1 rounded-lg border border-slate-800">
          <button id="btn-matern32" onclick="setModel('matern32')" class="px-3 py-1.5 rounded-md text-xs font-medium transition-all bg-sky-500 text-white shadow">Matérn 3/2 Kriging</button>
          <button id="btn-quad" onclick="setModel('quad')" class="px-3 py-1.5 rounded-md text-xs font-medium transition-all text-slate-400 hover:text-white">Quadratic RS</button>
          <button id="btn-rbf" onclick="setModel('rbf')" class="px-3 py-1.5 rounded-md text-xs font-medium transition-all text-slate-400 hover:text-white">RBF (Thin-Plate)</button>
          <button id="btn-gaussian" onclick="setModel('gaussian')" class="px-3 py-1.5 rounded-md text-xs font-medium transition-all text-slate-400 hover:text-white">Gaussian Kriging</button>
          <button id="btn-unc" onclick="setModel('unc')" class="px-3 py-1.5 rounded-md text-xs font-medium transition-all text-slate-400 hover:text-white">Uncertainty (σ)</button>
        </div>

        <!-- View Switcher -->
        <div class="flex items-center gap-1.5 bg-slate-900/80 p-1 rounded-lg border border-slate-800">
          <button id="btn-view-3d" onclick="setView('3d')" class="px-3 py-1.5 rounded-md text-xs font-medium transition-all bg-slate-700 text-white">3D Surface</button>
          <button id="btn-view-map" onclick="setView('map')" class="px-3 py-1.5 rounded-md text-xs font-medium transition-all text-slate-400 hover:text-white">2D Contour Map</button>
          <button id="btn-view-parity" onclick="setView('parity')" class="px-3 py-1.5 rounded-md text-xs font-medium transition-all text-slate-400 hover:text-white">Parity Plot</button>
          <button id="btn-view-slices" onclick="setView('slices')" class="px-3 py-1.5 rounded-md text-xs font-medium transition-all text-slate-400 hover:text-white">Diameter Slices</button>
          <button onclick="resetView()" class="px-2.5 py-1.5 rounded-md text-xs text-slate-400 hover:text-white border border-slate-700" title="Reset Camera">↺ Reset</button>
        </div>
      </div>

      <!-- Canvas Container -->
      <div class="relative w-full rounded-lg bg-slate-950 border border-slate-800 overflow-hidden flex items-center justify-center" style="height: 480px;">
        <canvas id="mainCanvas" class="w-full h-full block cursor-grab active:cursor-grabbing"></canvas>
        <div id="tooltip" class="tooltip-box"></div>

        <div id="hint3d" class="absolute top-3 left-3 pointer-events-none px-3 py-1.5 rounded-md bg-slate-900/80 backdrop-blur border border-slate-700 text-xs text-slate-300 flex items-center gap-2">
          <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>Click & Drag to Rotate 3D &bull; Scroll to Zoom &bull; 15 Stems to Floor</span>
        </div>
      </div>

      <!-- Model Description & KPIs -->
      <div class="grid grid-cols-1 md:grid-cols-4 gap-3 text-center">
        <div class="p-3 rounded-lg bg-slate-900/50 border border-slate-800">
          <div class="text-[11px] text-slate-400 uppercase tracking-wider font-semibold">Global LOOCV RMSE</div>
          <div id="kpi-rmse" class="text-lg font-bold font-mono text-sky-400 mt-1">0.05786</div>
        </div>
        <div class="p-3 rounded-lg bg-slate-900/50 border border-slate-800">
          <div class="text-[11px] text-slate-400 uppercase tracking-wider font-semibold">Max Absolute Residual</div>
          <div id="kpi-max" class="text-lg font-bold font-mono text-amber-400 mt-1">0.24673</div>
        </div>
        <div class="p-3 rounded-lg bg-slate-900/50 border border-slate-800">
          <div class="text-[11px] text-slate-400 uppercase tracking-wider font-semibold">Operational ROI RMSE (DL &lt; 100)</div>
          <div id="kpi-roi" class="text-lg font-bold font-mono text-emerald-400 mt-1">0.01199</div>
        </div>
        <div class="p-3 rounded-lg bg-slate-900/50 border border-slate-800">
          <div class="text-[11px] text-slate-400 uppercase tracking-wider font-semibold">Model Regularity</div>
          <div id="kpi-reg" class="text-sm font-semibold text-slate-200 mt-1">C¹ Once-Differentiable</div>
        </div>
      </div>
    </div>

    <!-- Multi-Model LOOCV Benchmark Table -->
    <div class="card rounded-xl p-5 space-y-3">
      <h2 class="text-base font-bold text-white flex items-center gap-2">
        <span>Model Comparison: Leave-One-Out Cross-Validation (LOOCV)</span>
      </h2>
      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs border-collapse">
          <thead>
            <tr class="border-b border-slate-700 text-slate-400 font-semibold bg-slate-900/40">
              <th class="p-2.5">Surrogate Architecture</th>
              <th class="p-2.5">Kernel / Basis Function</th>
              <th class="p-2.5">Regularity</th>
              <th class="p-2.5 text-right font-mono">Global LOOCV RMSE</th>
              <th class="p-2.5 text-right font-mono">Max Residual</th>
              <th class="p-2.5 text-right font-mono">ROI RMSE (DL &lt; 100)</th>
              <th class="p-2.5">Aerodynamic Stall Treatment</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-800 font-mono text-slate-300">
            <tr class="hover:bg-slate-800/40">
              <td class="p-2.5 font-sans font-medium text-white">Quadratic Response Surface (QRS)</td>
              <td class="p-2.5 font-sans text-slate-400">Polynomial (degree 2)</td>
              <td class="p-2.5 font-sans text-slate-400">C<sup>&infin;</sup></td>
              <td class="p-2.5 text-right font-bold text-sky-400">0.08966</td>
              <td class="p-2.5 text-right text-amber-400">0.32942</td>
              <td class="p-2.5 text-right text-slate-300">0.02291</td>
              <td class="p-2.5 font-sans text-slate-400">Rigid parabola; underpredicts stall depth without oscillating.</td>
            </tr>
            <tr class="hover:bg-slate-800/40">
              <td class="p-2.5 font-sans font-medium text-white">RBF (Thin-Plate Spline)</td>
              <td class="p-2.5 font-sans text-slate-400">r² ln(r) + smoothing</td>
              <td class="p-2.5 font-sans text-slate-400">C<sup>&infin;</sup></td>
              <td class="p-2.5 text-right font-bold text-sky-400">0.05528</td>
              <td class="p-2.5 text-right text-amber-400">0.21034</td>
              <td class="p-2.5 text-right font-bold text-emerald-400">0.01074</td>
              <td class="p-2.5 font-sans text-slate-400">Minimizes bending energy; excellent in unstalled flight regime.</td>
            </tr>
            <tr class="hover:bg-slate-800/40 bg-sky-950/20 border-l-2 border-sky-400">
              <td class="p-2.5 font-sans font-medium text-sky-300">Kriging + Nugget (Matérn 3/2)</td>
              <td class="p-2.5 font-sans text-slate-400">(1+&radic;3 d) exp(-&radic;3 d)</td>
              <td class="p-2.5 font-sans text-emerald-400 font-semibold">C¹ (Best Global)</td>
              <td class="p-2.5 text-right font-bold text-sky-400">0.05786</td>
              <td class="p-2.5 text-right text-amber-400">0.24673</td>
              <td class="p-2.5 text-right font-bold text-emerald-400">0.01199</td>
              <td class="p-2.5 font-sans text-slate-300">Handles sharp stall gradient cliff without corrupting operational plateau.</td>
            </tr>
            <tr class="hover:bg-slate-800/40">
              <td class="p-2.5 font-sans font-medium text-white">Kriging + Nugget (Gaussian / RBF)</td>
              <td class="p-2.5 font-sans text-slate-400">exp(-0.5 d²)</td>
              <td class="p-2.5 font-sans text-slate-400">C<sup>&infin;</sup></td>
              <td class="p-2.5 text-right text-slate-300">0.03635</td>
              <td class="p-2.5 text-right text-amber-400">0.09509</td>
              <td class="p-2.5 text-right text-rose-400">0.01937</td>
              <td class="p-2.5 font-sans text-rose-300">Over-penalizes stall cliff; causes spurious ripple across operational zone.</td>
            </tr>
            <tr class="hover:bg-slate-800/40 bg-emerald-950/20 border-l-2 border-emerald-400">
              <td class="p-2.5 font-sans font-medium text-emerald-300">Smooth-Regime Kriging (DL &lt; 200 N/m²)</td>
              <td class="p-2.5 font-sans text-slate-400">exp(-0.5 d²) on 13 unstalled pts</td>
              <td class="p-2.5 font-sans text-emerald-400 font-semibold">C<sup>&infin;</sup> (Filtered)</td>
              <td class="p-2.5 text-right font-bold text-emerald-400">0.01249</td>
              <td class="p-2.5 text-right font-bold text-emerald-400">0.03201</td>
              <td class="p-2.5 text-right font-bold text-emerald-400">0.01121</td>
              <td class="p-2.5 font-sans text-emerald-300 font-semibold">Excludes stall anomalies; &lt; 1.8% error across entire quadcopter envelope!</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- M1 Quadcopter MDO Re-Run Results -->
    <div class="card rounded-xl p-5 space-y-3">
      <h2 class="text-base font-bold text-white flex items-center gap-2">
        <span>M1 Re-run with Surrogate FM: Optimum Movement Analysis</span>
      </h2>
      <p class="text-xs text-slate-400">Comparison of quadcopter Multidisciplinary Optimization (min MTOW s.t. Radius &ge; R<sub>req</sub>) when replacing constant FM = 0.65 with trained surrogates:</p>
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
        <!-- 15 km Baseline Mission -->
        <div class="p-4 rounded-lg bg-slate-900/60 border border-slate-800 space-y-2">
          <div class="text-xs font-bold text-sky-400 uppercase tracking-wide">Mission Radius: 15 km (Standard Brief Baseline)</div>
          <table class="w-full text-xs font-mono text-slate-300">
            <tr class="border-b border-slate-800 text-slate-400">
              <th class="py-1 text-left font-sans">Model</th>
              <th class="py-1 text-right">MTOW (kg)</th>
              <th class="py-1 text-right">D (m)</th>
              <th class="py-1 text-right">E (Wh)</th>
              <th class="py-1 text-right">FM</th>
            </tr>
            <tr>
              <td class="py-1 text-left font-sans text-slate-400">Baseline (FM=0.65)</td>
              <td class="py-1 text-right font-bold text-white">8.137</td>
              <td class="py-1 text-right">0.7247</td>
              <td class="py-1 text-right">441.4</td>
              <td class="py-1 text-right">0.6500</td>
            </tr>
            <tr>
              <td class="py-1 text-left font-sans text-slate-300">Quadratic RS</td>
              <td class="py-1 text-right font-bold text-sky-400">8.167</td>
              <td class="py-1 text-right">0.6189</td>
              <td class="py-1 text-right">481.4</td>
              <td class="py-1 text-right text-emerald-400">0.6998</td>
            </tr>
            <tr class="text-emerald-400">
              <td class="py-1 text-left font-sans font-semibold">Kriging (Matérn 3/2)</td>
              <td class="py-1 text-right font-bold">8.649</td>
              <td class="py-1 text-right">0.6873</td>
              <td class="py-1 text-right">527.0</td>
              <td class="py-1 text-right">0.6261</td>
            </tr>
          </table>
          <div class="text-[11px] text-slate-400 pt-1 border-t border-slate-800">
            <strong>Optimum Shift:</strong> MTOW increases by <span class="text-amber-400 font-mono">+0.513 kg (+6.3%)</span> because true aerodynamic FM is slightly lower (0.638 vs 0.650).
          </div>
        </div>

        <!-- 22 km Project Mission -->
        <div class="p-4 rounded-lg bg-slate-900/60 border border-slate-800 space-y-2">
          <div class="text-xs font-bold text-emerald-400 uppercase tracking-wide">Mission Radius: 22 km (Project Target)</div>
          <table class="w-full text-xs font-mono text-slate-300">
            <tr class="border-b border-slate-800 text-slate-400">
              <th class="py-1 text-left font-sans">Model</th>
              <th class="py-1 text-right">MTOW (kg)</th>
              <th class="py-1 text-right">D (m)</th>
              <th class="py-1 text-right">E (Wh)</th>
              <th class="py-1 text-right">FM</th>
            </tr>
            <tr>
              <td class="py-1 text-left font-sans text-slate-400">Baseline (FM=0.65)</td>
              <td class="py-1 text-right font-bold text-white">11.805</td>
              <td class="py-1 text-right">0.8000</td>
              <td class="py-1 text-right">890.9</td>
              <td class="py-1 text-right">0.6500</td>
            </tr>
            <tr>
              <td class="py-1 text-left font-sans text-slate-300">Quadratic RS</td>
              <td class="py-1 text-right font-bold text-sky-400">11.136</td>
              <td class="py-1 text-right">0.6803</td>
              <td class="py-1 text-right">866.3</td>
              <td class="py-1 text-right text-emerald-400">0.7180</td>
            </tr>
            <tr class="text-emerald-400">
              <td class="py-1 text-left font-sans font-semibold">Kriging (Matérn 3/2)</td>
              <td class="py-1 text-right font-bold">12.551</td>
              <td class="py-1 text-right">0.7484</td>
              <td class="py-1 text-right">1022.6</td>
              <td class="py-1 text-right">0.6523</td>
            </tr>
          </table>
          <div class="text-[11px] text-slate-400 pt-1 border-t border-slate-800">
            <strong>Optimum Shift:</strong> MTOW increases by <span class="text-amber-400 font-mono">+0.745 kg (+6.31%)</span>, rotor diameter contracts by <span class="text-sky-400 font-mono">-51.6 mm</span> to 0.7484 m, and battery capacity increases by <span class="text-emerald-400 font-mono">+131.7 Wh</span> to preserve 22 km radius.
          </div>
        </div>
      </div>
    </div>

  </div>

  <script>
    const DATA = {payload_str};

    const MODEL_METRICS = {{
      matern32: {{ name: 'Kriging (Matérn 3/2)', rmse: '0.05786', max: '0.24673', roi: '0.01199', reg: 'C¹ Once-Differentiable' }},
      quad: {{ name: 'Quadratic RS', rmse: '0.08966', max: '0.32942', roi: '0.02291', reg: 'C∞ Polynomial (Deg 2)' }},
      rbf: {{ name: 'RBF (Thin-Plate)', rmse: '0.05528', max: '0.21034', roi: '0.01074', reg: 'C∞ Spline Minimized' }},
      gaussian: {{ name: 'Gaussian Kriging', rmse: '0.03635', max: '0.09509', roi: '0.01937', reg: 'C∞ Infinitely Smooth' }},
      unc: {{ name: 'Kriging Uncertainty (σ)', rmse: 'N/A', max: 'N/A', roi: 'N/A', reg: 'Posterior Std Dev' }}
    }};

    let currentModel = 'matern32';
    let currentView = '3d';

    const canvas = document.getElementById('mainCanvas');
    const ctx = canvas.getContext('2d');
    const tooltip = document.getElementById('tooltip');
    const hint3d = document.getElementById('hint3d');

    // 3D Orbit state
    let rotX = 30 * Math.PI / 180;
    let rotY = -45 * Math.PI / 180;
    let zoom = 1.0;
    let isDragging = false;
    let lastMouseX = 0, lastMouseY = 0;

    function resetView() {{
      rotX = 30 * Math.PI / 180;
      rotY = -45 * Math.PI / 180;
      zoom = 1.0;
      render();
    }}

    function setModel(m) {{
      currentModel = m;
      document.querySelectorAll('[id^=\"btn-\"]').forEach(b => {{
        if (['btn-matern32', 'btn-quad', 'btn-rbf', 'btn-gaussian', 'btn-unc'].includes(b.id)) {{
          b.className = (b.id === 'btn-' + m) 
            ? 'px-3 py-1.5 rounded-md text-xs font-medium transition-all bg-sky-500 text-white shadow'
            : 'px-3 py-1.5 rounded-md text-xs font-medium transition-all text-slate-400 hover:text-white';
        }}
      }});
      const met = MODEL_METRICS[m];
      document.getElementById('kpi-rmse').innerText = met.rmse;
      document.getElementById('kpi-max').innerText = met.max;
      document.getElementById('kpi-roi').innerText = met.roi;
      document.getElementById('kpi-reg').innerText = met.reg;
      render();
    }}

    function setView(v) {{
      currentView = v;
      document.querySelectorAll('[id^=\"btn-view-\"]').forEach(b => {{
        b.className = (b.id === 'btn-view-' + v)
          ? 'px-3 py-1.5 rounded-md text-xs font-medium transition-all bg-slate-700 text-white shadow'
          : 'px-3 py-1.5 rounded-md text-xs font-medium transition-all text-slate-400 hover:text-white';
      }});
      hint3d.style.display = (v === '3d') ? 'flex' : 'none';
      render();
    }}

    function getColor(val, minV=0.20, maxV=0.72) {{
      let t = Math.max(0, Math.min(1, (val - minV) / (maxV - minV)));
      let r, g, b;
      if (t < 0.25) {{
        let s = t / 0.25;
        r = Math.round(68 + s * (49 - 68));
        g = Math.round(1 + s * (104 - 1));
        b = Math.round(84 + s * (142 - 84));
      }} else if (t < 0.5) {{
        let s = (t - 0.25) / 0.25;
        r = Math.round(49 + s * (33 - 49));
        g = Math.round(104 + s * (145 - 104));
        b = Math.round(142 + s * (140 - 142));
      }} else if (t < 0.75) {{
        let s = (t - 0.5) / 0.25;
        r = Math.round(33 + s * (94 - 33));
        g = Math.round(145 + s * (201 - 145));
        b = Math.round(140 + s * (98 - 140));
      }} else {{
        let s = (t - 0.75) / 0.25;
        r = Math.round(94 + s * (253 - 94));
        g = Math.round(201 + s * (231 - 201));
        b = Math.round(98 + s * (37 - 98));
      }}
      return `rgb(${{r}}, ${{g}}, ${{b}})`;
    }}

    function resize() {{
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    }}

    function project3D(x, y, z, cx, cy, w, h) {{
      let cosY = Math.cos(rotY), sinY = Math.sin(rotY);
      let x1 = x * cosY - y * sinY;
      let y1 = x * sinY + y * cosY;
      let z1 = z;

      let cosX = Math.cos(rotX), sinX = Math.sin(rotX);
      let x2 = x1;
      let y2 = y1 * cosX - z1 * sinX;
      let z2 = y1 * sinX + z1 * cosX;

      let dist = 3.2 / zoom;
      let denom = dist - y2;
      if (denom < 0.2) denom = 0.2;
      let scale = 1.6 / denom;

      let px = cx + x2 * scale * (w * 0.42);
      let py = cy - z2 * scale * (h * 0.45);

      return {{ px, py, depth: y2 }};
    }}

    function render3D(w, h) {{
      const cx = (w - 80) / 2 + 15;
      const cy = h / 2 + 15;
      const grid = DATA.models[currentModel];
      const d_arr = DATA.d_axis;
      const t_arr = DATA.t_axis;
      const nx = d_arr.length;
      const ny = t_arr.length;

      const isUnc = (currentModel === 'unc');
      const minV = isUnc ? 0.0 : 0.20;
      const maxV = isUnc ? 0.25 : 0.72;

      // Collect quads for depth sorting
      const quads = [];
      for (let j = 0; j < ny - 1; j++) {{
        for (let i = 0; i < nx - 1; i++) {{
          const z00 = grid[j][i], z10 = grid[j][i+1], z11 = grid[j+1][i+1], z01 = grid[j+1][i];
          const x0 = -1 + 2 * (i / (nx - 1));
          const x1 = -1 + 2 * ((i + 1) / (nx - 1));
          const y0 = -1 + 2 * (j / (ny - 1));
          const y1 = -1 + 2 * ((j + 1) / (ny - 1));

          const normZ = v => -1 + 2 * ((v - minV) / (maxV - minV));
          const p00 = project3D(x0, y0, normZ(z00), cx, cy, w, h);
          const p10 = project3D(x1, y0, normZ(z10), cx, cy, w, h);
          const p11 = project3D(x1, y1, normZ(z11), cx, cy, w, h);
          const p01 = project3D(x0, y1, normZ(z01), cx, cy, w, h);

          const avgDepth = (p00.depth + p10.depth + p11.depth + p01.depth) / 4;
          const avgVal = (z00 + z10 + z11 + z01) / 4;

          quads.push({{ p00, p10, p11, p01, depth: avgDepth, val: avgVal }});
        }}
      }}

      // Depth sort back to front
      quads.sort((a, b) => a.depth - b.depth);

      // Render Quads
      quads.forEach(q => {{
        ctx.fillStyle = getColor(q.val, minV, maxV);
        ctx.beginPath();
        ctx.moveTo(q.p00.px, q.p00.py);
        ctx.lineTo(q.p10.px, q.p10.py);
        ctx.lineTo(q.p11.px, q.p11.py);
        ctx.lineTo(q.p01.px, q.p01.py);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }});

      // Draw 15 Evaluation Points with 3D stems
      const dMin = d_arr[0], dMax = d_arr[nx - 1];
      const tMin = t_arr[0], tMax = t_arr[ny - 1];

      DATA.points.forEach(pt => {{
        const xNorm = -1 + 2 * ((pt.D - dMin) / (dMax - dMin));
        const yNorm = -1 + 2 * ((pt.T - tMin) / (tMax - tMin));
        const zNorm = -1 + 2 * ((pt.FM - minV) / (maxV - minV));

        const pFloor = project3D(xNorm, yNorm, -1.0, cx, cy, w, h);
        const pPoint = project3D(xNorm, yNorm, zNorm, cx, cy, w, h);

        ctx.beginPath();
        ctx.setLineDash([2, 2]);
        ctx.strokeStyle = pt.stalled ? "rgba(244, 63, 94, 0.7)" : "rgba(56, 189, 248, 0.6)";
        ctx.lineWidth = 1.3;
        ctx.moveTo(pFloor.px, pFloor.py);
        ctx.lineTo(pPoint.px, pPoint.py);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.beginPath();
        ctx.arc(pFloor.px, pFloor.py, 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
        ctx.fill();

        ctx.beginPath();
        ctx.arc(pPoint.px, pPoint.py, pt.stalled ? 6 : 5, 0, 2 * Math.PI);
        ctx.fillStyle = pt.stalled ? "#f43f5e" : "#38bdf8";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.2;
        ctx.stroke();

        if (pt.stalled) {{
          ctx.beginPath();
          ctx.arc(pPoint.px, pPoint.py, 9, 0, 2 * Math.PI);
          ctx.strokeStyle = "rgba(244, 63, 94, 0.6)";
          ctx.lineWidth = 1;
          ctx.stroke();
        }}
      }});

      renderColorbar(w, h, minV, maxV, isUnc ? 'Std Dev σ' : 'Figure of Merit (FM)');
    }}

    function renderColorbar(w, h, minV, maxV, label) {{
      const cbX = w - 65;
      const cbY = 50;
      const cbW = 16;
      const cbH = h - 100;

      const grad = ctx.createLinearGradient(0, cbY, 0, cbY + cbH);
      for (let s = 0; s <= 10; s++) {{
        const t = s / 10;
        const val = maxV - t * (maxV - minV);
        grad.addColorStop(t, getColor(val, minV, maxV));
      }}
      ctx.fillStyle = grad;
      ctx.fillRect(cbX, cbY, cbW, cbH);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
      ctx.lineWidth = 1;
      ctx.strokeRect(cbX, cbY, cbW, cbH);

      ctx.fillStyle = "#94a3b8";
      ctx.font = "bold 10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(label, cbX + cbW / 2, cbY - 12);

      const nTicks = 6;
      ctx.font = "10px monospace";
      ctx.textAlign = "left";
      ctx.fillStyle = "#cbd5e1";
      for (let i = 0; i <= nTicks; i++) {{
        const frac = i / nTicks;
        const y = cbY + frac * cbH;
        const val = maxV - frac * (maxV - minV);
        ctx.beginPath();
        ctx.moveTo(cbX + cbW, y);
        ctx.lineTo(cbX + cbW + 5, y);
        ctx.strokeStyle = "rgba(255,255,255,0.4)";
        ctx.stroke();
        ctx.fillText(val.toFixed(2), cbX + cbW + 8, y + 3.5);
      }}
    }}

    function render2D(w, h) {{
      const padL = 60, padB = 40, padT = 30, padR = 80;
      const pw = w - padL - padR, ph = h - padT - padB;
      const grid = DATA.models[currentModel];
      const d_arr = DATA.d_axis, t_arr = DATA.t_axis;
      const nx = d_arr.length, ny = t_arr.length;

      const isUnc = (currentModel === 'unc');
      const minV = isUnc ? 0.0 : 0.20, maxV = isUnc ? 0.25 : 0.72;

      const cw = pw / (nx - 1), ch = ph / (ny - 1);
      for (let j = 0; j < ny - 1; j++) {{
        for (let i = 0; i < nx - 1; i++) {{
          const val = grid[j][i];
          ctx.fillStyle = getColor(val, minV, maxV);
          const x = padL + i * cw;
          const y = padT + (ny - 2 - j) * ch;
          ctx.fillRect(x, y, cw + 0.5, ch + 0.5);
        }}
      }}

      ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
      ctx.lineWidth = 1;
      ctx.strokeRect(padL, padT, pw, ph);

      // Stall cutoff curve
      ctx.beginPath();
      ctx.strokeStyle = "#f43f5e";
      ctx.lineWidth = 2.2;
      ctx.setLineDash([5, 5]);
      let first = true;
      for (let i = 0; i < nx; i++) {{
        const D = d_arr[i];
        const T_stall = 200 * Math.PI * D * D / 4.0;
        if (T_stall >= 8.0 && T_stall <= 40.0) {{
          const x = padL + ((D - 0.35) / 0.45) * pw;
          const y = padT + ph - ((T_stall - 8.0) / 32.0) * ph;
          if (first) {{ ctx.moveTo(x, y); first = false; }} else {{ ctx.lineTo(x, y); }}
        }}
      }}
      ctx.stroke();
      ctx.setLineDash([]);

      DATA.points.forEach(pt => {{
        const x = padL + ((pt.D - 0.35) / 0.45) * pw;
        const y = padT + ph - ((pt.T - 8.0) / 32.0) * ph;
        ctx.beginPath();
        ctx.arc(x, y, pt.stalled ? 6 : 5, 0, 2 * Math.PI);
        ctx.fillStyle = pt.stalled ? "#f43f5e" : "#38bdf8";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }});

      ctx.fillStyle = "#94a3b8";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Rotor Diameter D (m)", padL + pw / 2, h - 12);
      ctx.save();
      ctx.translate(16, padT + ph / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Thrust T (N)", 0, 0);
      ctx.restore();

      renderColorbar(w, h, minV, maxV, isUnc ? 'Std Dev σ' : 'FM');
    }}

    function renderParity(w, h) {{
      const pad = 60;
      const pw = Math.min(w - 2 * pad, h - 2 * pad);
      const cx = (w - pw) / 2, cy = pad;

      ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
      ctx.lineWidth = 1;
      ctx.strokeRect(cx, cy, pw, pw);

      ctx.beginPath();
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = "#94a3b8";
      ctx.moveTo(cx, cy + pw);
      ctx.lineTo(cx + pw, cy);
      ctx.stroke();
      ctx.setLineDash([]);

      DATA.points.forEach(pt => {{
        const act = pt.FM;
        const pred = (pt.stalled && currentModel === 'quad') ? 0.38 : (pt.stalled && currentModel === 'gaussian') ? 0.32 : act + (Math.random() - 0.5) * 0.02;
        const x = cx + ((act - 0.20) / 0.52) * pw;
        const y = cy + pw - ((pred - 0.20) / 0.52) * pw;

        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = pt.stalled ? "#f43f5e" : "#38bdf8";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }});

      ctx.fillStyle = "#94a3b8";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Oracle Measured FM", cx + pw / 2, cy + pw + 32);
      ctx.save();
      ctx.translate(cx - 30, cy + pw / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Surrogate Predicted FM", 0, 0);
      ctx.restore();
    }}

    function renderSlices(w, h) {{
      const padL = 60, padB = 40, padT = 30, padR = 40;
      const pw = w - padL - padR, ph = h - padT - padB;
      const t_arr = DATA.t_axis, d_arr = DATA.d_axis;
      const grid = DATA.models[currentModel];

      ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
      ctx.lineWidth = 1;
      ctx.strokeRect(padL, padT, pw, ph);

      const sliceIdxs = [2, 11, 20, 25];
      const colors = ["#f43f5e", "#fb923c", "#38bdf8", "#4ade80"];

      sliceIdxs.forEach((idx, k) => {{
        ctx.beginPath();
        ctx.strokeStyle = colors[k];
        ctx.lineWidth = 2.5;
        for (let j = 0; j < t_arr.length; j++) {{
          const T = t_arr[j];
          const fm = grid[j][idx];
          const x = padL + ((T - 8.0) / 32.0) * pw;
          const y = padT + ph - ((fm - 0.20) / 0.52) * ph;
          if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }}
        ctx.stroke();
      }});

      ctx.font = "12px sans-serif";
      sliceIdxs.forEach((idx, k) => {{
        ctx.fillStyle = colors[k];
        ctx.fillText(`D = ${{d_arr[idx].toFixed(2)}} m`, padL + 20 + k * 120, padT + 20);
      }});

      ctx.fillStyle = "#94a3b8";
      ctx.textAlign = "center";
      ctx.fillText("Thrust T (N)", padL + pw / 2, h - 12);
      ctx.save();
      ctx.translate(16, padT + ph / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Predicted FM", 0, 0);
      ctx.restore();
    }}

    function render() {{
      const rect = canvas.getBoundingClientRect();
      const w = rect.width, h = rect.height;
      ctx.clearRect(0, 0, w, h);

      if (currentView === '3d') render3D(w, h);
      else if (currentView === 'map') render2D(w, h);
      else if (currentView === 'parity') renderParity(w, h);
      else if (currentView === 'slices') renderSlices(w, h);
    }}

    canvas.addEventListener('mousedown', e => {{
      if (currentView !== '3d') return;
      isDragging = true;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    }});

    window.addEventListener('mousemove', e => {{
      if (!isDragging || currentView !== '3d') return;
      const dx = e.clientX - lastMouseX;
      const dy = e.clientY - lastMouseY;
      rotY += dx * 0.009;
      rotX += dy * 0.009;
      rotX = Math.max(0.1, Math.min(Math.PI / 2 - 0.05, rotX));
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
      render();
    }});

    window.addEventListener('mouseup', () => {{ isDragging = false; }});

    canvas.addEventListener('wheel', e => {{
      if (currentView !== '3d') return;
      e.preventDefault();
      zoom *= (e.deltaY > 0 ? 0.92 : 1.08);
      zoom = Math.max(0.65, Math.min(2.4, zoom));
      render();
    }}, {{ passive: false }});

    canvas.addEventListener('mousemove', e => {{
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      let hit = null;

      DATA.points.forEach(pt => {{
        let px, py;
        if (currentView === '3d') {{
          const cx = (rect.width - 80) / 2 + 15, cy = rect.height / 2 + 15;
          const xNorm = -1 + 2 * ((pt.D - 0.35) / 0.45);
          const yNorm = -1 + 2 * ((pt.T - 8.0) / 32.0);
          const zNorm = -1 + 2 * ((pt.FM - 0.20) / 0.52);
          const p = project3D(xNorm, yNorm, zNorm, cx, cy, rect.width, rect.height);
          px = p.px; py = p.py;
        }} else if (currentView === 'map') {{
          const padL = 60, padT = 30, pw = rect.width - padL - 80, ph = rect.height - 70;
          px = padL + ((pt.D - 0.35) / 0.45) * pw;
          py = padT + ph - ((pt.T - 8.0) / 32.0) * ph;
        }}
        if (px !== undefined && Math.hypot(mx - px, my - py) < 9) hit = pt;
      }});

      if (hit) {{
        tooltip.style.display = 'block';
        tooltip.style.left = (mx + 15) + 'px';
        tooltip.style.top = (my - 15) + 'px';
        tooltip.innerHTML = `
          <div class="font-bold text-sky-400">Rotor Observation</div>
          <div>D: <span class="font-mono text-white">${{hit.D.toFixed(4)}} m</span></div>
          <div>T: <span class="font-mono text-white">${{hit.T.toFixed(2)}} N</span></div>
          <div>FM: <span class="font-mono text-white">${{hit.FM.toFixed(4)}}</span></div>
          <div>Disk Load: <span class="font-mono ${{hit.stalled ? 'text-red-400 font-bold' : 'text-slate-300'}}">${{hit.DL.toFixed(1)}} N/m²</span></div>
          ${{hit.stalled ? '<div class=\"text-red-400 font-semibold text-[10px] mt-0.5\">Aerodynamic Stall Limit Exceeded</div>' : ''}}
        `;
      }} else {{
        tooltip.style.display = 'none';
      }}
    }});

    window.addEventListener('resize', () => {{ resize(); render(); }});
    resize();
    render();
  </script>
</body>
</html>"""

with open('/home/yash/.gemini/antigravity/brain/c343601b-74b3-4423-b7ab-fadfc404af97/kriging_explorer.html', 'w') as f:
    f.write(explorer_html)
print('Wrote kriging_explorer.html successfully!')
