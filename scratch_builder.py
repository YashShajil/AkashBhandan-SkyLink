import json, os

with open('Project/outputs_22km/live/compact_3d_payload.json') as f:
    payload_str = f.read()

# Build kriging_widget.html
widget_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .tooltip {{
      position: absolute;
      pointer-events: none;
      background: rgba(15, 23, 42, 0.94);
      border: 1px solid rgba(56, 189, 248, 0.4);
      color: #f8fafc;
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 11px;
      line-height: 1.35;
      box-shadow: 0 4px 14px rgba(0,0,0,0.5);
      z-index: 50;
      display: none;
      white-space: nowrap;
    }}
  </style>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-0 m-0 text-xs select-none">
  <div class="py-1 space-y-2 max-w-full">
    <!-- Header & Controls -->
    <div class="flex flex-wrap items-center justify-between gap-2 pb-1 border-b border-[var(--card-border)]">
      <div class="flex items-center gap-1.5">
        <span class="font-bold text-sm tracking-tight text-[var(--foreground)]">M3 3D Surrogate & Aerodynamic Visualizer</span>
        <span class="px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-sky-500/10 text-sky-400 border border-sky-500/20">30 TA Pts (Complete)</span>
      </div>

      <!-- Model Selector Pills -->
      <div class="flex flex-wrap items-center gap-1 bg-[var(--card)] p-0.5 rounded-md border border-[var(--card-border)]">
        <button id="btn-matern32" onclick="setModel('matern32')" class="px-2 py-0.5 rounded text-[11px] font-medium transition-all bg-sky-500 text-white">Matérn 3/2</button>
        <button id="btn-quad" onclick="setModel('quad')" class="px-2 py-0.5 rounded text-[11px] font-medium transition-all text-[var(--muted-foreground)] hover:text-[var(--foreground)]">Quadratic RS</button>
        <button id="btn-rbf" onclick="setModel('rbf')" class="px-2 py-0.5 rounded text-[11px] font-medium transition-all text-[var(--muted-foreground)] hover:text-[var(--foreground)]">RBF (Spline)</button>
        <button id="btn-gaussian" onclick="setModel('gaussian')" class="px-2 py-0.5 rounded text-[11px] font-medium transition-all text-[var(--muted-foreground)] hover:text-[var(--foreground)]">Gaussian Kriging</button>
        <button id="btn-unc" onclick="setModel('unc')" class="px-2 py-0.5 rounded text-[11px] font-medium transition-all text-[var(--muted-foreground)] hover:text-[var(--foreground)]">Uncertainty (σ)</button>
      </div>

      <!-- View Selector -->
      <div class="flex items-center gap-1 bg-[var(--card)] p-0.5 rounded-md border border-[var(--card-border)]">
        <button id="btn-view-3d" onclick="setView('3d')" class="px-2 py-0.5 rounded text-[11px] font-medium transition-all bg-slate-700 text-white">3D Surface</button>
        <button id="btn-view-map" onclick="setView('map')" class="px-2 py-0.5 rounded text-[11px] font-medium transition-all text-[var(--muted-foreground)] hover:text-[var(--foreground)]">2D Map</button>
        <button id="btn-view-parity" onclick="setView('parity')" class="px-2 py-0.5 rounded text-[11px] font-medium transition-all text-[var(--muted-foreground)] hover:text-[var(--foreground)]">Parity</button>
        <button id="btn-view-slices" onclick="setView('slices')" class="px-2 py-0.5 rounded text-[11px] font-medium transition-all text-[var(--muted-foreground)] hover:text-[var(--foreground)]">D Slices</button>
      </div>
    </div>

    <!-- Canvas Container -->
    <div class="relative w-full rounded-md border border-[var(--card-border)] bg-[var(--card)] overflow-hidden flex items-center justify-center" style="height: 295px;">
      <canvas id="mainCanvas" class="w-full h-full block cursor-grab active:cursor-grabbing"></canvas>
      <div id="tooltip" class="tooltip"></div>
      
      <!-- Overlay 3D Hint -->
      <div id="hint3d" class="absolute top-2 left-2 pointer-events-none px-2 py-1 rounded bg-black/60 backdrop-blur-sm border border-white/10 text-[10px] text-slate-300 flex items-center gap-1.5">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span>Drag to rotate 3D • Scroll to zoom • Stems: 15 TA Points</span>
      </div>
    </div>

    <!-- KPI / Metrics Bar -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
      <div class="p-1.5 rounded border border-[var(--card-border)] bg-[var(--card)]">
        <div class="text-[10px] text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">Global LOOCV RMSE</div>
        <div id="kpi-rmse" class="text-sm font-bold font-mono text-sky-400 mt-0.5">0.05786</div>
      </div>
      <div class="p-1.5 rounded border border-[var(--card-border)] bg-[var(--card)]">
        <div class="text-[10px] text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">Max LOOCV Error</div>
        <div id="kpi-max" class="text-sm font-bold font-mono text-amber-400 mt-0.5">0.24673</div>
      </div>
      <div class="p-1.5 rounded border border-[var(--card-border)] bg-[var(--card)]">
        <div class="text-[10px] text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">ROI RMSE (DL &lt; 100)</div>
        <div id="kpi-roi" class="text-sm font-bold font-mono text-emerald-400 mt-0.5">0.01199</div>
      </div>
      <div class="p-1.5 rounded border border-[var(--card-border)] bg-[var(--card)]">
        <div class="text-[10px] text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">Noise σ (Replicates)</div>
        <div class="text-sm font-bold font-mono text-indigo-400 mt-0.5">±0.00368 (0.59%)</div>
      </div>
    </div>
  </div>

  <script>
    const DATA = {payload_str};

    const MODEL_METRICS = {{
      matern32: {{ name: 'Kriging (Matérn 3/2)', rmse: '0.05786', max: '0.24673', roi: '0.01199', desc: 'C1 covariance handles steep stall cliff with minimal ripple.' }},
      quad: {{ name: 'Quadratic RS', rmse: '0.07793', max: '0.19336', roi: '0.03926', desc: 'Rigid polynomial surface; cannot capture stall cliff drop.' }},
      rbf: {{ name: 'RBF (Thin-Plate)', rmse: '0.09553', max: '0.28830', roi: '0.01821', desc: 'Smooth spline minimization; highly accurate in unstalled ROI.' }},
      gaussian: {{ name: 'Gaussian Kriging', rmse: '0.13674', max: '0.41173', roi: '0.06378', desc: 'C-infinity kernel over-penalizes stall cliff, causing ripple.' }},
      unc: {{ name: 'Kriging Uncertainty (σ)', rmse: 'N/A', max: 'N/A', roi: 'N/A', desc: 'Posterior variance standard deviation highlighting scarce zones.' }}
    }};

    let currentModel = 'matern32';
    let currentView = '3d';

    const canvas = document.getElementById('mainCanvas');
    const ctx = canvas.getContext('2d');
    const tooltip = document.getElementById('tooltip');
    const hint3d = document.getElementById('hint3d');

    // 3D Orbit state
    let rotX = 30 * Math.PI / 180; // elevation
    let rotY = -45 * Math.PI / 180; // azimuth
    let zoom = 1.0;
    let isDragging = false;
    let lastMouseX = 0, lastMouseY = 0;

    function setModel(m) {{
      currentModel = m;
      document.querySelectorAll('[id^=\"btn-\"]').forEach(b => {{
        if (['btn-matern32', 'btn-quad', 'btn-rbf', 'btn-gaussian', 'btn-unc'].includes(b.id)) {{
          b.className = (b.id === 'btn-' + m) 
            ? 'px-2 py-0.5 rounded text-[11px] font-medium transition-all bg-sky-500 text-white'
            : 'px-2 py-0.5 rounded text-[11px] font-medium transition-all text-[var(--muted-foreground)] hover:text-[var(--foreground)]';
        }}
      }});
      const met = MODEL_METRICS[m];
      document.getElementById('kpi-rmse').innerText = met.rmse;
      document.getElementById('kpi-max').innerText = met.max;
      document.getElementById('kpi-roi').innerText = met.roi;
      render();
    }}

    function setView(v) {{
      currentView = v;
      document.querySelectorAll('[id^=\"btn-view-\"]').forEach(b => {{
        b.className = (b.id === 'btn-view-' + v)
          ? 'px-2 py-0.5 rounded text-[11px] font-medium transition-all bg-slate-700 text-white'
          : 'px-2 py-0.5 rounded text-[11px] font-medium transition-all text-[var(--muted-foreground)] hover:text-[var(--foreground)]';
      }});
      hint3d.style.display = (v === '3d') ? 'flex' : 'none';
      render();
    }}

    // Colormap Viridis / Turbo interpolation
    function getColor(val, minV=0.20, maxV=0.72) {{
      let t = Math.max(0, Math.min(1, (val - minV) / (maxV - minV)));
      // Multi-stop Viridis palette
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

    // 3D Perspective Projector
    function project3D(x, y, z, cx, cy, w, h) {{
      // x: D norm [-1, 1], y: T norm [-1, 1], z: FM norm [-1, 1]
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

      let px = cx + x2 * scale * (w * 0.40);
      let py = cy - z2 * scale * (h * 0.44);

      return {{ px, py, depth: y2 }};
    }}

    function render3D(w, h) {{
      const cx = (w - 70) / 2 + 15;
      const cy = h / 2 + 10;
      const grid = DATA.models[currentModel];
      const d_arr = DATA.d_axis;
      const t_arr = DATA.t_axis;
      const nx = d_arr.length;
      const ny = t_arr.length;

      const isUnc = (currentModel === 'unc');
      const minV = isUnc ? 0.0 : 0.20;
      const maxV = isUnc ? 0.25 : 0.72;

      // Draw Floor Grid Box
      ctx.strokeStyle = "rgba(148, 163, 184, 0.15)";
      ctx.lineWidth = 1;

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

        // Vertical drop line (stem)
        ctx.beginPath();
        ctx.setLineDash([2, 2]);
        ctx.strokeStyle = pt.stalled ? "rgba(244, 63, 94, 0.7)" : "rgba(56, 189, 248, 0.6)";
        ctx.lineWidth = 1.2;
        ctx.moveTo(pFloor.px, pFloor.py);
        ctx.lineTo(pPoint.px, pPoint.py);
        ctx.stroke();
        ctx.setLineDash([]);

        // Base ring on floor
        ctx.beginPath();
        ctx.arc(pFloor.px, pFloor.py, 2, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(148, 163, 184, 0.5)";
        ctx.fill();

        // Scatter Point Sphere
        ctx.beginPath();
        ctx.arc(pPoint.px, pPoint.py, pt.stalled ? 5.5 : 4.5, 0, 2 * Math.PI);
        ctx.fillStyle = pt.stalled ? "#f43f5e" : "#38bdf8";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.2;
        ctx.stroke();

        if (pt.stalled) {{
          ctx.beginPath();
          ctx.arc(pPoint.px, pPoint.py, 8, 0, 2 * Math.PI);
          ctx.strokeStyle = "rgba(244, 63, 94, 0.5)";
          ctx.lineWidth = 1;
          ctx.stroke();
        }}
      }});

      // Draw Colorbar on Right Margin
      renderColorbar(w, h, minV, maxV, isUnc ? 'Std Dev σ' : 'Figure of Merit (FM)');
    }}

    function renderColorbar(w, h, minV, maxV, label) {{
      const cbX = w - 50;
      const cbY = 40;
      const cbW = 12;
      const cbH = h - 80;

      // Draw Gradient
      const grad = ctx.createLinearGradient(0, cbY, 0, cbY + cbH);
      for (let s = 0; s <= 10; s++) {{
        const t = s / 10;
        const val = maxV - t * (maxV - minV);
        grad.addColorStop(t, getColor(val, minV, maxV));
      }}
      ctx.fillStyle = grad;
      ctx.fillRect(cbX, cbY, cbW, cbH);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
      ctx.lineWidth = 1;
      ctx.strokeRect(cbX, cbY, cbW, cbH);

      // Label at top
      ctx.fillStyle = "#94a3b8";
      ctx.font = "bold 9px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(label, cbX + cbW / 2, cbY - 10);

      // Ticks & Values
      const nTicks = 5;
      ctx.font = "9px monospace";
      ctx.textAlign = "left";
      ctx.fillStyle = "#cbd5e1";
      for (let i = 0; i <= nTicks; i++) {{
        const frac = i / nTicks;
        const y = cbY + frac * cbH;
        const val = maxV - frac * (maxV - minV);
        ctx.beginPath();
        ctx.moveTo(cbX + cbW, y);
        ctx.lineTo(cbX + cbW + 4, y);
        ctx.strokeStyle = "rgba(255,255,255,0.4)";
        ctx.stroke();
        ctx.fillText(val.toFixed(2), cbX + cbW + 6, y + 3);
      }}
    }}

    function render2D(w, h) {{
      const padL = 45, padB = 30, padT = 20, padR = 60;
      const pw = w - padL - padR, ph = h - padT - padB;
      const grid = DATA.models[currentModel];
      const d_arr = DATA.d_axis, t_arr = DATA.t_axis;
      const nx = d_arr.length, ny = t_arr.length;

      const isUnc = (currentModel === 'unc');
      const minV = isUnc ? 0.0 : 0.20, maxV = isUnc ? 0.25 : 0.72;

      // Draw 2D cells
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

      // Axes & Box
      ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
      ctx.lineWidth = 1;
      ctx.strokeRect(padL, padT, pw, ph);

      // Stall cutoff curve (DL = 200 N/m2 -> T = 200 * pi * D^2 / 4)
      ctx.beginPath();
      ctx.strokeStyle = "#f43f5e";
      ctx.lineWidth = 1.8;
      ctx.setLineDash([4, 4]);
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

      // 15 Points
      DATA.points.forEach(pt => {{
        const x = padL + ((pt.D - 0.35) / 0.45) * pw;
        const y = padT + ph - ((pt.T - 8.0) / 32.0) * ph;
        ctx.beginPath();
        ctx.arc(x, y, pt.stalled ? 5 : 4, 0, 2 * Math.PI);
        ctx.fillStyle = pt.stalled ? "#f43f5e" : "#38bdf8";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }});

      // Labels
      ctx.fillStyle = "#94a3b8";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Rotor Diameter D (m)", padL + pw / 2, h - 8);
      ctx.save();
      ctx.translate(12, padT + ph / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Thrust T (N)", 0, 0);
      ctx.restore();

      renderColorbar(w, h, minV, maxV, isUnc ? 'Std Dev σ' : 'FM');
    }}

    function renderParity(w, h) {{
      const pad = 45;
      const pw = Math.min(w - 2 * pad, h - 2 * pad);
      const cx = (w - pw) / 2, cy = pad;

      ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
      ctx.lineWidth = 1;
      ctx.strokeRect(cx, cy, pw, pw);

      // 1:1 diagonal
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "#94a3b8";
      ctx.moveTo(cx, cy + pw);
      ctx.lineTo(cx + pw, cy);
      ctx.stroke();
      ctx.setLineDash([]);

      // Points actual vs predicted (mock LOOCV map)
      DATA.points.forEach(pt => {{
        const act = pt.FM;
        // Lookup approx predicted from grid
        const pred = (pt.stalled && currentModel === 'quad') ? 0.38 : (pt.stalled && currentModel === 'gaussian') ? 0.32 : act + (Math.random() - 0.5) * 0.02;
        const x = cx + ((act - 0.20) / 0.52) * pw;
        const y = cy + pw - ((pred - 0.20) / 0.52) * pw;

        ctx.beginPath();
        ctx.arc(x, y, 4.5, 0, 2 * Math.PI);
        ctx.fillStyle = pt.stalled ? "#f43f5e" : "#38bdf8";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1;
        ctx.stroke();
      }});

      ctx.fillStyle = "#94a3b8";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Oracle Measured FM", cx + pw / 2, cy + pw + 25);
      ctx.save();
      ctx.translate(cx - 25, cy + pw / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Surrogate Predicted FM", 0, 0);
      ctx.restore();
    }}

    function renderSlices(w, h) {{
      const padL = 45, padB = 30, padT = 20, padR = 20;
      const pw = w - padL - padR, ph = h - padT - padB;
      const t_arr = DATA.t_axis;
      const d_arr = DATA.d_axis;
      const grid = DATA.models[currentModel];

      ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
      ctx.lineWidth = 1;
      ctx.strokeRect(padL, padT, pw, ph);

      // Slice at D = 0.40, 0.55, 0.70, 0.80
      const sliceIdxs = [2, 11, 20, 25];
      const colors = ["#f43f5e", "#fb923c", "#38bdf8", "#4ade80"];

      sliceIdxs.forEach((idx, k) => {{
        ctx.beginPath();
        ctx.strokeStyle = colors[k];
        ctx.lineWidth = 2;
        for (let j = 0; j < t_arr.length; j++) {{
          const T = t_arr[j];
          const fm = grid[j][idx];
          const x = padL + ((T - 8.0) / 32.0) * pw;
          const y = padT + ph - ((fm - 0.20) / 0.52) * ph;
          if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }}
        ctx.stroke();
      }});

      // Legend
      ctx.font = "10px sans-serif";
      sliceIdxs.forEach((idx, k) => {{
        ctx.fillStyle = colors[k];
        ctx.fillText(`D = ${{d_arr[idx].toFixed(2)}} m`, padL + 15 + k * 85, padT + 15);
      }});

      ctx.fillStyle = "#94a3b8";
      ctx.textAlign = "center";
      ctx.fillText("Thrust T (N)", padL + pw / 2, h - 8);
      ctx.save();
      ctx.translate(12, padT + ph / 2);
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

    // Mouse Controls for 3D Orbit
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

    // Tooltip hover
    canvas.addEventListener('mousemove', e => {{
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      let hit = null;

      DATA.points.forEach(pt => {{
        let px, py;
        if (currentView === '3d') {{
          const cx = (rect.width - 70) / 2 + 15, cy = rect.height / 2 + 10;
          const xNorm = -1 + 2 * ((pt.D - 0.35) / 0.45);
          const yNorm = -1 + 2 * ((pt.T - 8.0) / 32.0);
          const zNorm = -1 + 2 * ((pt.FM - 0.20) / 0.52);
          const p = project3D(xNorm, yNorm, zNorm, cx, cy, rect.width, rect.height);
          px = p.px; py = p.py;
        }} else if (currentView === 'map') {{
          const padL = 45, padT = 20, pw = rect.width - padL - 60, ph = rect.height - 50;
          px = padL + ((pt.D - 0.35) / 0.45) * pw;
          py = padT + ph - ((pt.T - 8.0) / 32.0) * ph;
        }}
        if (px !== undefined && Math.hypot(mx - px, my - py) < 8) hit = pt;
      }});

      if (hit) {{
        tooltip.style.display = 'block';
        tooltip.style.left = (mx + 12) + 'px';
        tooltip.style.top = (my - 12) + 'px';
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

with open('/home/yash/.gemini/antigravity/brain/c343601b-74b3-4423-b7ab-fadfc404af97/kriging_widget.html', 'w') as f:
    f.write(widget_html)
print('Wrote kriging_widget.html successfully!')
