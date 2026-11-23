"""M5 - discrete-event simulation (SimPy) of one day of operations.

Entities: orders (non-homogeneous Poisson by PHC, class, hour), drones, batteries, charging bays, hubs.
Events: order arrival, dispatch, take-off, delivery/pick-up, return, battery swap, charge complete,
grounding by gust (hourly), outage start/end (Mon/Wed/Fri 14-17), emergency road timeout (30 min).
"""
from __future__ import annotations
import math, os, json
import numpy as np, simpy
from . import data

CLASSES = ("em", "ro", "lab")


class Config:
    def __init__(self, sol, vp, spares=None, reserve=0.2, demand_mult=1.0, availability=1.0,
                 routine_wait_max_min=360, wind_mode="file", wind_scale=1.0, record=False, weekday_shift=0):
        self.sol, self.vp = sol, vp
        self.hubs = list(sol["hubs"].keys())
        self.spares = spares if spares is not None else {h: sol["hubs"][h]["drones"] for h in self.hubs}
        self.reserve, self.demand_mult, self.availability = reserve, demand_mult, availability
        self.routine_wait_max = routine_wait_max_min
        self.wind_mode, self.wind_scale, self.record = wind_mode, wind_scale, record
        self.weekday_shift = weekday_shift


def _prep_static():
    D, _ = data.distance_matrix(); S = data.sites(); dem = data.demand(); hp = data.hourly(); W = data.wind()
    Dd = {(a, b): float(D.loc[a, b]) for a in D.index for b in D.columns}
    XY = {k: (float(r.x_km), float(r.y_km)) for k, r in S.iterrows()}
    demd = {p: {c: float(dem.loc[p, c]) for c in CLASSES} for p in dem.index}
    hpa = {c: hp[c].values.astype(float) for c in CLASSES}
    Wd = {}
    for d, g in W.groupby("day"):
        g = g.sort_values("hour")
        Wd[int(d)] = (g.mean_wind_ms.values, g.gust_ms.values, g.direction_deg.values if "direction_deg" in g else np.zeros(24), int(g.weekday.iloc[0]))
    return Dd, XY, demd, hpa, Wd


_STATIC = None
_ROADC = dict(emergency_inr_per_trip=2500, routine_lab_inr_per_trip=500)


def static():
    global _STATIC
    if _STATIC is None:
        _STATIC = _prep_static()
    return _STATIC


class Hub:
    def __init__(self, sim, name, info, cfg, rng):
        self.sim, self.name, self.cfg = sim, name, cfg
        env = sim.env
        n = info["drones"]
        self.n_total = n
        self.n = int(rng.binomial(n, cfg.availability)) if cfg.availability < 1 else n
        self.n = max(self.n, 0)
        self.idle = self.n
        self.reserve_k = math.ceil(cfg.reserve * self.n) if cfg.reserve > 0 else 0
        self.queue = []                                 # (priority, t_arrival, seq, order)
        self.batteries = simpy.Store(env)              # charged spare packs (Wh capacity tags)
        for _ in range(cfg.spares.get(name, 0)):
            self.batteries.put(1)
        self.bays = info["bays"]
        self.charging = []                              # remaining Wh to put in, per pack being charged
        self.waiting_charge = []
        self.busy_time = 0.0
        self.energy_grid_Wh = 0.0; self.energy_gen_Wh = 0.0
        self.wake = None
        env.process(self.charger())

    # --- charging: 1-minute tick, power shared among packs on bays
    def charger(self):
        env = self.sim.env
        while True:
            if not self.charging and not self.waiting_charge:
                self.wake = env.event(); yield self.wake; continue
            while len(self.charging) < self.bays and self.waiting_charge:
                self.charging.append(self.waiting_charge.pop(0))
            if self.charging:
                outage = self.sim.outage(env.now)
                p_avail = min(len(self.charging) * 1500.0, 3000.0 if outage else 1e9)   # W at the plug
                p_each = p_avail / len(self.charging) * 0.9                                 # W into the pack
                done = []
                for k in range(len(self.charging)):
                    self.charging[k] -= p_each / 60.0
                    if self.charging[k] <= 0: done.append(k)
                if outage: self.energy_gen_Wh += p_avail / 60.0
                else: self.energy_grid_Wh += p_avail / 60.0
                for k in reversed(done):
                    self.charging.pop(k); self.batteries.put(1)
                    self.sim.dispatch(self)
            yield env.timeout(1.0)


class Sim:
    def __init__(self, cfg, day, seed, erlang=None):
        self.cfg, self.day, self.erlang = cfg, day, erlang
        self.env = simpy.Environment()
        self.rng = np.random.default_rng(seed)
        D, XY, dem, hp, W = static()
        self.D, self.XY = D, XY
        w, g, dr, wk = W[int(day)]
        self.w = w * cfg.wind_scale
        self.wdir = dr
        self.grounded = (g * cfg.wind_scale) >= 14.0
        self.weekday = (wk + cfg.weekday_shift) % 7
        self.fcache = {}
        if erlang:
            self.w[:] = 0; self.grounded[:] = False
        self.assign = {p: h for h, info in cfg.sol["hubs"].items() for p in info["phcs"]}
        self.hubs = {h: Hub(self, h, info, cfg, self.rng) for h, info in cfg.sol["hubs"].items()}
        self.records = []; self.trace = []; self.seq = 0
        self.qlog = []

    def outage(self, t):
        if self.erlang: return False
        return self.weekday in (0, 2, 4) and 14 * 60 <= (t % 1440) < 17 * 60

    def hour(self, t):
        return int(t // 60) % 24

    # ---------------------------------------------------------- flight model
    def flight(self, hub, phc, t):
        vp = self.cfg.vp; V = vp["V"]
        L = self.D[(hub, phc)] * 1000
        if L == 0: return 0.0, 0.0, 0.0, True
        h = self.hour(t); w = self.w[h]
        (hx, hy), (px, py) = self.XY[hub], self.XY[phc]
        course = math.degrees(math.atan2(px - hx, py - hy)) % 360          # bearing from north
        head = w * math.cos(math.radians(self.wdir[h] - course))            # wind FROM dir: +ve = headwind outbound
        if abs(head) >= V - 1: return 0, 0, 0, False
        t_out = L / (V - head); t_back = L / (V + head)
        return t_out, t_back, head, True

    def energy(self, t_loaded, t_empty):
        vp = self.cfg.vp
        return (vp["Pcl"] * t_loaded + vp["Pce"] * t_empty + 60 * (vp["Phl"] + vp["Phe"])) / 3600

    # ---------------------------------------------------------- orders
    def arrivals(self):
        _, _, dem, hp, _ = static(); env = self.env
        evs = []
        if self.erlang:
            lam = self.erlang["lam_per_min"]; t = 0.0; hub = list(self.hubs)[0]
            phc = self.cfg.sol["hubs"][hub]["phcs"][0]
            while t < self.erlang["horizon_min"]:
                t += self.rng.exponential(1 / lam); evs.append((t, phc, "em"))
        else:
            for p in self.assign:
                for c in CLASSES:
                    for h in range(24):
                        mu = dem[p][c] * hp[c][h] * self.cfg.demand_mult
                        for _ in range(self.rng.poisson(mu)):
                            evs.append((h * 60 + self.rng.uniform(0, 60), p, c))
            # PHCs not served by any hub: road
            for p in dem:
                if p not in self.assign:
                    for c in CLASSES:
                        for _ in range(self.rng.poisson(dem[p][c] * self.cfg.demand_mult)):
                            self.records.append(dict(phc=p, cls=c, t0=0.0, mode="road", reason="unassigned", t_done=120 if c == "em" else 180))
        evs.sort()
        for t, p, c in evs:
            yield env.timeout(t - env.now)
            self.new_order(p, c)

    def new_order(self, p, c):
        hub = self.assign[p]; H = self.hubs[hub]; t = self.env.now
        o = dict(phc=p, cls=c, hub=hub, t0=t, mode=None)
        self.seq += 1
        pri = 0 if c == "em" else 1
        if hub == p:   # co-located: hand-carried
            o.update(mode="walk", t_done=t + 5); self.records.append(o); return
        H.queue.append((pri, t, self.seq, o))
        if c == "em" and not self.erlang:
            self.env.process(self.timeout_road(o, 30))
        elif not self.erlang:
            self.env.process(self.timeout_road(o, self.cfg.routine_wait_max))
        self.dispatch(H)

    def timeout_road(self, o, wait):
        yield self.env.timeout(wait)
        if o["mode"] is None:
            o["mode"] = "road"; o["reason"] = "waited"
            o["t_done"] = self.env.now + (120 if o["cls"] == "em" else 180)
            H = self.hubs[o["hub"]]
            H.queue = [q for q in H.queue if q[3] is not o]
            self.records.append(o)

    # ---------------------------------------------------------- dispatch
    def dispatch(self, H):
        t = self.env.now
        if self.grounded[self.hour(t)]: return
        H.queue.sort(key=lambda q: (q[0], q[1], q[2]))
        k = 0
        while k < len(H.queue) and H.idle > 0:
            pri, t0, s, o = H.queue[k]
            if pri > 0 and H.idle <= H.reserve_k:
                k += 1; continue                       # hold the reserve for emergencies
            t_out, t_back, head, ok = self.flight(H.name, o["phc"], t)
            if ok:
                loaded, empty = (t_out, t_back) if o["cls"] != "lab" else (t_back, t_out)
                e = self.energy(loaded, empty)
                if e > self.cfg.vp["Euse_Wh"]: ok = False
            if not ok:                                   # cannot be flown in this wind -> road now
                H.queue.pop(k); o["mode"] = "road"; o["reason"] = "wind-range"
                o["t_done"] = t + (120 if o["cls"] == "em" else 180); self.records.append(o); continue
            H.queue.pop(k); H.idle -= 1; o["mode"] = "drone"
            self.env.process(self.sortie(H, o, t_out, t_back, e))

    def sortie(self, H, o, t_out, t_back, e):
        env = self.env; t_disp = env.now
        if self.erlang:
            svc = self.rng.exponential(self.erlang["mean_service_min"])
            o["t_dispatch"] = t_disp; o["t_done"] = t_disp; self.records.append(o)
            yield env.timeout(svc); H.busy_time += svc; H.idle += 1; self.dispatch(H); return
        yield env.timeout(3.0 + t_out / 60)                        # load+take-off, outbound
        yield env.timeout(1.0)                                     # drop / pick-up
        o["t_dispatch"] = t_disp
        o["t_done"] = env.now if o["cls"] != "lab" else env.now + t_back / 60
        self.records.append(o)
        if self.cfg.record:
            self.trace.append(dict(hub=H.name, phc=o["phc"], cls=o["cls"], t0=t_disp, t1=t_disp + 3 + t_out / 60, t2=env.now + t_back / 60))
        yield env.timeout(t_back / 60)
        H.waiting_charge.append(e / 1.0)                            # Wh to restore
        if H.wake is not None and not H.wake.triggered: H.wake.succeed()
        yield H.batteries.get()                                    # swap needs a charged pack
        yield env.timeout(6.0)
        H.busy_time += env.now - t_disp
        H.idle += 1
        self.dispatch(H)

    def ticker(self):
        while True:
            yield self.env.timeout(60 - (self.env.now % 60) + 1e-6)
            if self.cfg.record:
                for h, H in self.hubs.items():
                    self.qlog.append((self.env.now, h, len(H.queue), H.idle, len(H.charging), H.batteries.items.__len__()))
            for H in self.hubs.values(): self.dispatch(H)

    def qticker(self):
        while True:
            yield self.env.timeout(5)
            for h, H in self.hubs.items():
                self.qlog.append((round(self.env.now, 1), h, len(H.queue), H.idle, len(H.charging), len(H.batteries.items)))

    def run(self, horizon=1440 + 300):
        self.env.process(self.arrivals()); self.env.process(self.ticker())
        if self.cfg.record: self.env.process(self.qticker())
        self.env.run(until=horizon if not self.erlang else self.erlang["horizon_min"] + 600)
        return self.summary()

    def summary(self):
        recs = self.records
        em = [r for r in recs if r["cls"] == "em"]
        dt = np.array([r["t_done"] - r["t0"] for r in em]) if em else np.array([])
        n_tot = len(recs); road = sum(r["mode"] == "road" for r in recs)
        busy = sum(H.busy_time for H in self.hubs.values()); ndr = sum(H.n for H in self.hubs.values())
        cm = _ROADC
        road_cost = sum((cm["emergency_inr_per_trip"] if r["cls"] == "em" else cm["routine_lab_inr_per_trip"]) for r in recs if r["mode"] == "road")
        return dict(n_em=len(em), n_em_45=int((dt <= 45).sum()), em_times=dt, n_orders=n_tot, n_road=road,
                    util=busy / max(ndr * 1440, 1), energy_grid_kWh=sum(H.energy_grid_Wh for H in self.hubs.values()) / 1000,
                    energy_gen_kWh=sum(H.energy_gen_Wh for H in self.hubs.values()) / 1000, road_cost=road_cost,
                    grounded_hours=int(self.grounded.sum()))


# ------------------------------------------------------------------ multi-day evaluation
def sample_days(n, seed=0, stratified=True, pool=None):
    rng = np.random.default_rng(seed)
    if pool is not None:
        return list(rng.choice(pool, n, replace=True))
    if stratified:   # equal days per month -> annual average with less variance (CRN-friendly)
        W = data.wind(); days = []
        per = max(1, n // 12)
        for m in range(1, 13):
            dm = W[W.month == m].day.unique(); days += list(rng.choice(dm, per, replace=False))
        return days[:n] if len(days) >= n else days + list(rng.integers(1, 366, n - len(days)))
    return list(rng.integers(1, 366, n))


def evaluate(cfg, days, seeds=None):
    res = []
    for k, d in enumerate(days):
        s = Sim(cfg, int(d), seed=(seeds[k] if seeds is not None else 1000 + k)).run()
        res.append(s)
    n_em = sum(r["n_em"] for r in res); n45 = sum(r["n_em_45"] for r in res)
    times = np.concatenate([r["em_times"] for r in res]) if res else np.array([])
    per_day = np.array([r["n_em_45"] / max(r["n_em"], 1) for r in res])
    # ratio-estimator CI over days (days are i.i.d. batches)
    y = np.array([r["n_em_45"] for r in res], float); x = np.array([r["n_em"] for r in res], float)
    S = n45 / max(n_em, 1); nd = len(res)
    se = math.sqrt(((y - S * x) ** 2).sum() / (nd - 1) / nd) / x.mean() if nd > 1 else np.nan
    return dict(S45=S, S45_halfwidth=1.96 * se, P95_em_min=float(np.percentile(times, 95)) if len(times) else np.nan,
                mean_em_min=float(times.mean()) if len(times) else np.nan,
                util=float(np.mean([r["util"] for r in res])), road_share=sum(r["n_road"] for r in res) / max(sum(r["n_orders"] for r in res), 1),
                energy_kWh_day=float(np.mean([r["energy_grid_kWh"] + r["energy_gen_kWh"] for r in res])),
                genset_kWh_day=float(np.mean([r["energy_gen_kWh"] for r in res])),
                road_cost_day=float(np.mean([r["road_cost"] for r in res])), n_days=nd, per_day_S45=per_day)


# ------------------------------------------------------------------ verification: M/M/c
def erlang_c(lam, mu, c):
    a = lam / mu; rho = a / c
    s = sum(a ** k / math.factorial(k) for k in range(c))
    pc = a ** c / math.factorial(c) / (1 - rho)
    P = pc / (s + pc)
    return P / (c * mu - lam), P


def verify_mmc(vp, c=3, lam_per_hr=6.0, mean_service_min=24.0, reps=20, horizon_hr=2000, seed0=0):
    sol = {"hubs": {"H0": dict(phcs=["P11"], drones=c, bays=99)}}
    cfg = Config(sol, vp, spares={"H0": 10_000}, reserve=0.0)
    lam = lam_per_hr / 60; mu = 1 / mean_service_min
    Wq_th, Pw = erlang_c(lam, mu, c)
    W = []
    for r in range(reps):
        s = Sim(cfg, 100, seed=seed0 + r, erlang=dict(lam_per_min=lam, mean_service_min=mean_service_min, horizon_min=horizon_hr * 60))
        s.run()
        recs = [o for o in s.records if o["t0"] > 600]                 # drop warm-up
        W.append(np.mean([o["t_dispatch"] - o["t0"] for o in recs]))
    W = np.array(W)
    return dict(c=c, lam_per_hr=lam_per_hr, mean_service_min=mean_service_min, rho=lam / (c * mu), P_wait=Pw,
                Wq_theory_min=Wq_th, Wq_sim_min=float(W.mean()), Wq_sim_ci95=float(1.96 * W.std(ddof=1) / math.sqrt(reps)),
                rel_err=float(W.mean() / Wq_th - 1), reps=reps, horizon_hr=horizon_hr)


def days_for_ci(cfg, target=0.03, start=24, step=12, max_days=600, seed=11):
    """sequential: add days (random over the year) until the 95% CI on S45 is narrower than +-target"""
    rng = np.random.default_rng(seed); days = list(rng.integers(1, 366, start)); hist = []
    while True:
        ev = evaluate(cfg, days, seeds=list(range(5000, 5000 + len(days))))
        hist.append((len(days), ev["S45"], ev["S45_halfwidth"]))
        if ev["S45_halfwidth"] < target or len(days) >= max_days:
            return ev, hist
        days += list(rng.integers(1, 366, step))


def record_day(cfg, day, seed=1):
    cfg.record = True
    s = Sim(cfg, day, seed); summ = s.run(); cfg.record = False
    return s, summ
