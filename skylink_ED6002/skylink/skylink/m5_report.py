"""M5 driver: verification (M/M/c), days-for-CI study, and the live day view (HTML)."""
from __future__ import annotations
import os, json
import numpy as np, pandas as pd
from . import data, m5_sim as m5

HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>SkyLink - one simulated day</title>
<style>
body{margin:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#0f1419;color:#e6e6e6}
#wrap{display:flex;gap:16px;padding:14px;flex-wrap:wrap}
canvas{background:#16202a;border-radius:8px}
#side{min-width:300px;flex:1}
h1{font-size:17px;margin:0 0 6px} .muted{color:#8a99a8;font-size:12px}
.kpi{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}
.k{background:#1c2733;border-radius:8px;padding:8px} .k b{font-size:20px;display:block}
.hub{margin:6px 0;font-size:12px} .bar{height:8px;border-radius:4px;background:#2a3846;margin-top:2px;position:relative;overflow:hidden}
.bar span{position:absolute;top:0;bottom:0;left:0}
button{background:#2b6cb0;color:#fff;border:0;border-radius:6px;padding:6px 12px;cursor:pointer;margin-right:4px}
.flag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;margin-right:6px}
</style></head><body><div id="wrap">
<canvas id="c" width="780" height="585"></canvas>
<div id="side"><h1>SkyLink — Kaveripuram, one simulated day</h1>
<div class="muted" id="meta"></div>
<div style="margin:10px 0"><button id="pp">Pause</button><button data-s="1">1×</button><button data-s="4">4×</button><button data-s="12">12×</button>
<input type="range" id="t" min="0" max="1440" value="0" style="width:100%;margin-top:8px"></div>
<div style="font-size:28px;font-weight:600" id="clock">00:00</div>
<div id="flags"></div>
<div class="kpi"><div class="k">Emergencies<b id="ne">0</b></div><div class="k">Delivered ≤45 min<b id="s45">–</b></div>
<div class="k">Drones in air<b id="air">0</b></div><div class="k">Sent by road<b id="road">0</b></div></div>
<div class="muted">Per hub: queue (orange), idle drones (green), packs charging (blue)</div><div id="hubs"></div>
<div class="muted" style="margin-top:10px">Red dots: emergency flights · white: routine · cyan: lab returns. Shaded: no-fly zones.</div>
</div></div>
<script>
const D = __DATA__;
const cv=document.getElementById('c'),cx=cv.getContext('2d');const S=13;
const X=x=>x*S, Y=y=>cv.height-y*S;
let t=0,speed=4,play=true,last=null;
document.getElementById('meta').textContent=D.meta;
document.querySelectorAll('button[data-s]').forEach(b=>b.onclick=()=>speed=+b.dataset.s);
document.getElementById('pp').onclick=e=>{play=!play;e.target.textContent=play?'Pause':'Play'};
const sl=document.getElementById('t');sl.oninput=()=>{t=+sl.value};
function pos(p,f){ // position along polyline p at fraction f
 let L=0,seg=[];for(let i=1;i<p.length;i++){const d=Math.hypot(p[i][0]-p[i-1][0],p[i][1]-p[i-1][1]);seg.push(d);L+=d}
 let s=f*L;for(let i=0;i<seg.length;i++){if(s<=seg[i]){const r=seg[i]?s/seg[i]:0;return [p[i][0]+r*(p[i+1][0]-p[i][0]),p[i][1]+r*(p[i+1][1]-p[i][1])]}s-=seg[i]}return p[p.length-1]}
function draw(){
 cx.clearRect(0,0,cv.width,cv.height);
 cx.strokeStyle='#22303d';for(let i=0;i<=60;i+=5){cx.beginPath();cx.moveTo(X(i),0);cx.lineTo(X(i),cv.height);cx.stroke()}for(let j=0;j<=45;j+=5){cx.beginPath();cx.moveTo(0,Y(j));cx.lineTo(cv.width,Y(j));cx.stroke()}
 D.zones.forEach(z=>{cx.fillStyle='rgba(220,60,60,.22)';cx.beginPath();z.forEach((p,i)=>i?cx.lineTo(X(p[0]),Y(p[1])):cx.moveTo(X(p[0]),Y(p[1])));cx.fill()});
 for(const k in D.paths){const p=D.paths[k];cx.strokeStyle='rgba(120,150,180,.25)';cx.beginPath();p.forEach((q,i)=>i?cx.lineTo(X(q[0]),Y(q[1])):cx.moveTo(X(q[0]),Y(q[1])));cx.stroke()}
 D.sites.forEach(s=>{const hub=D.hubs.includes(s.id);cx.fillStyle=hub?'#f6c343':(s.id[0]=='P'?'#cfd8e3':'#5d6b7a');
  if(hub){cx.fillRect(X(s.x)-7,Y(s.y)-7,14,14)}else{cx.beginPath();cx.arc(X(s.x),Y(s.y),3.5,0,7);cx.fill()}
  cx.fillStyle='#8a99a8';cx.font='10px sans-serif';cx.fillText(s.id,X(s.x)+6,Y(s.y)-5)});
 let air=0;D.trips.forEach(r=>{if(t<r.t0||t>r.t2)return;air++;const p=D.paths[r.hub+'-'+r.phc];if(!p)return;
  let q; if(t<r.t1){q=pos(p,(t-r.t0)/(r.t1-r.t0))}else{q=pos(p,1-(t-r.t1)/(r.t2-r.t1))}
  cx.fillStyle=r.cls=='em'?'#ff4d4d':(r.cls=='lab'?'#3dd6f5':'#ffffff');cx.beginPath();cx.arc(X(q[0]),Y(q[1]),r.cls=='em'?5:3.5,0,7);cx.fill()});
 const h=Math.floor(t/60)%24,m=Math.floor(t%60);document.getElementById('clock').textContent=String(h).padStart(2,'0')+':'+String(m).padStart(2,'0');
 document.getElementById('air').textContent=air;
 const em=D.em.filter(e=>e.t0<=t);const done=em.filter(e=>e.td<=t);const ok=done.filter(e=>e.td-e.t0<=45);
 document.getElementById('ne').textContent=em.length;document.getElementById('s45').textContent=done.length?Math.round(100*ok.length/done.length)+'%':'–';
 document.getElementById('road').textContent=D.road.filter(x=>x<=t).length;
 let f='';if(D.grounded[h])f+='<span class="flag" style="background:#c53030">GROUNDED (gust ≥14 m/s)</span>';
 if(D.outage&&t>=840&&t<1020)f+='<span class="flag" style="background:#b7791f">GRID OUTAGE – genset 3 kW</span>';
 f+='<span class="flag" style="background:#2d3748">wind '+D.wind[h].toFixed(1)+' m/s</span>';document.getElementById('flags').innerHTML=f;
 let hh='';D.hubs.forEach(hb=>{const L=D.q[hb];let i=0;while(i<L.length-1&&L[i+1][0]<=t)i++;const s=L[i]||[0,0,0,0,0];const n=D.n[hb];
  hh+=`<div class="hub"><b>${hb}</b> — queue ${s[1]}, idle ${s[2]}/${n}, charging ${s[3]}, spare packs ${s[4]}
  <div class="bar"><span style="width:${Math.min(100,s[1]*10)}%;background:#ed8936"></span></div>
  <div class="bar"><span style="width:${100*s[2]/n}%;background:#48bb78"></span></div>
  <div class="bar"><span style="width:${Math.min(100,s[3]*25)}%;background:#4299e1"></span></div></div>`});
 document.getElementById('hubs').innerHTML=hh;sl.value=t;
}
function loop(ts){if(last!==null&&play){t+=(ts-last)/1000*speed;if(t>1440)t=0}last=ts;draw();requestAnimationFrame(loop)}
requestAnimationFrame(loop);
</script></body></html>"""


def live_view(cfg, day, path_out, seed=3, label=""):
    s, summ = m5.record_day(cfg, day, seed)
    S = data.sites()
    paths = {}
    for h, info in cfg.sol["hubs"].items():
        for p in info["phcs"]:
            if p != h: paths[f"{h}-{p}"] = [list(map(float, q)) for q in data.path(h, p)]
    q = {h: [] for h in cfg.hubs}
    for (t, h, ql, idle, ch, bat) in s.qlog:
        q[h].append([t, ql, idle, ch, bat])
    em = [dict(t0=r["t0"], td=r["t_done"]) for r in s.records if r["cls"] == "em"]
    road = [r["t0"] for r in s.records if r["mode"] == "road"]
    W = data.wind(); wd = W[W.day == day].sort_values("hour")
    D = dict(meta=f"{label} · day {day} of the wind year ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][s.weekday]}) · "
                  f"{summ['n_orders']} orders, {summ['n_em']} emergencies, S45 = {summ['n_em_45']/max(summ['n_em'],1):.0%}, "
                  f"{summ['grounded_hours']} grounded hours" + (" · wind data SYNTHETIC" if data.sources()["wind_days.csv"] != "real" else ""),
             zones=[[list(map(float, c)) for c in z.exterior.coords] for z in data.nofly().values()],
             sites=[dict(id=k, x=float(r.x_km), y=float(r.y_km)) for k, r in S.iterrows()], hubs=cfg.hubs,
             paths=paths, trips=[{k: (round(v, 2) if isinstance(v, float) else v) for k, v in tr.items()} for tr in s.trace],
             q=q, n={h: s.hubs[h].n for h in cfg.hubs}, em=em, road=road, grounded=[bool(x) for x in s.grounded],
             wind=[float(x) for x in s.w], outage=bool(s.weekday in (0, 2, 4)))
    open(path_out, "w").write(HTML.replace("__DATA__", json.dumps(D, default=lambda o: o.item() if hasattr(o, "item") else str(o))))
    return summ


def pick_story_day():
    """a weekday with a grid outage and some (not total) grounding -> shows every event type"""
    W = data.wind()
    g = W.groupby("day").agg(gr=("grounded", "sum"), wk=("weekday", "first"), m=("month", "first"))
    c = g[(g.wk.isin([0, 2, 4])) & (g.gr >= 2) & (g.gr <= 5)]
    return int(c.index[0]) if len(c) else 1


def run(sol, vp, reserve=0.2, tag="m5", label=""):
    tabd = os.path.join(data.OUT, "tables"); figd = os.path.join(data.OUT, "figs")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    mmc = [m5.verify_mmc(vp, c=c, lam_per_hr=lam, mean_service_min=svc, reps=40, horizon_hr=4000, seed0=1000 * k)
           for k, (c, lam, svc) in enumerate([(1, 2.0, 20.0), (3, 6.0, 24.0), (5, 10.0, 25.0)])]
    pd.DataFrame(mmc).to_csv(os.path.join(tabd, f"{tag}_mmc_verification.csv"), index=False)
    cfg = m5.Config(sol, vp, reserve=reserve)
    ev, hist = m5.days_for_ci(cfg)
    hist = pd.DataFrame(hist, columns=["days", "S45", "halfwidth"]); hist.to_csv(os.path.join(tabd, f"{tag}_ci_days.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].errorbar(hist.days, hist.S45 * 100, yerr=hist.halfwidth * 100, fmt="o-", capsize=3)
    ax[0].set_xlabel("simulated days (random over the year)"); ax[0].set_ylabel("S45 (%) with 95% CI"); ax[0].grid(alpha=.3)
    ax[0].set_title(f"Days needed for CI < ±3 pp: {int(hist.days.iloc[-1])}")
    ax[1].hist(ev["per_day_S45"] * 100, bins=20); ax[1].set_xlabel("per-day S45 (%)"); ax[1].set_ylabel("days"); ax[1].set_title("Day-to-day spread (weather days dominate the tail)")
    plt.tight_layout(); plt.savefig(os.path.join(figd, f"{tag}_ci.png"), dpi=120); plt.close()
    day = pick_story_day()
    os.makedirs(os.path.join(data.OUT, "live"), exist_ok=True)
    lv = live_view(cfg, day, os.path.join(data.OUT, "live", "skylink_day.html"), label=label)
    summ = dict(mmc=mmc, ci=dict(days=int(hist.days.iloc[-1]), **{k: v for k, v in ev.items() if k != "per_day_S45"}),
                story_day=day, story_summary={k: v for k, v in lv.items() if k != "em_times"})
    json.dump(summ, open(os.path.join(tabd, f"{tag}_summary.json"), "w"), indent=1, default=float)
    return summ


if __name__ == "__main__":
    sol = json.load(open(os.path.join(data.OUT, "tables", "m4_solutions.json")))["V4"]
    s = run(sol, sol["vp"], label="V4 network")
    print(json.dumps({k: v for k, v in s.items() if k != "mmc"}, indent=1, default=float))
    print(pd.DataFrame(s["mmc"])[["c", "rho", "Wq_theory_min", "Wq_sim_min", "Wq_sim_ci95", "rel_err"]])
