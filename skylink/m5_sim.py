"""M5 - discrete-event simulator (SimPy) of one day of operations + verification, CI study, live view.

Entities: orders (non-homogeneous Poisson by PHC, class, hour), drones, batteries, charging bays, hubs.
Events: order arrival, dispatch, take-off, delivery/pick-up, return, battery swap, charge complete,
grounding by gust (hourly), outage start/end (Mon/Wed/Fri 14-17), emergency road timeout (30 min).
Run:  python m5_sim.py   (~1 min). Uses outputs/tables/m4_solutions.json from your last M4 run
      (or designs the network itself if that file is missing).
Prints: Erlang-C verification, days for a +-3 pp CI on S45, S45, P95, utilisation, road share, energy.
Saves:  outputs/tables/m5_*.csv/json, outputs/figs/m5_ci.png, outputs/live/skylink_day.html
"""
from __future__ import annotations
import sys, os
if __package__ in (None, ""):          # allow running this file directly (PyCharm Run button)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "skylink"
import math, os, json
import numpy as np, pandas as pd, simpy
from . import data

CLASSES = ("em", "ro", "lab")          # demand.csv classes
EMERG = ("em", "blood")                  # emergency classes: antivenom (from the PHC's hub) and blood (only from H0)
RESTOCK_HOUR = 6          # blood restock runs leave H0 at 06:00 (same as M4)
RESUPPLY_PER_DAY = 1      # ASSUMPTION (same as M4): restock runs per day to each open hospital hub


class Config:
    def __init__(self, sol, vp, spares=None, reserve=0.2, demand_mult=1.0, availability=1.0,
                 routine_wait_max_min=360, em_wait_max_min=90, wind_mode="file", wind_scale=1.0, record=False, weekday_shift=0):
        self.sol, self.vp = sol, vp
        self.hubs = list(sol["hubs"].keys())
        self.spares = spares if spares is not None else {h: sol["hubs"][h]["drones"] for h in self.hubs}
        self.reserve, self.demand_mult, self.availability = reserve, demand_mult, availability
        self.routine_wait_max = routine_wait_max_min
        self.em_wait_max = em_wait_max_min      # emergency waits up to 90 min for a drone (road 120 min - flight ~25 min: until then a drone is faster), then road
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
        self.gust = g * cfg.wind_scale
        self.grounded = self.gust >= 14.0
        self.weekday = (wk + cfg.weekday_shift) % 7
        self.fcache = {}
        if erlang:
            self.w[:] = 0; self.grounded[:] = False
        self.assign = {p: h for h, info in cfg.sol["hubs"].items() for p in info["phcs"]}
        self.hubs = {h: Hub(self, h, info, cfg, self.rng) for h, info in cfg.sol["hubs"].items()}
        self.records = []; self.trace = []; self.seq = 0
        self.restocks = []                     # blood restock runs H0 -> hospital hubs (not counted as PHC orders)
        self.qlog = []

    def outage(self, t):
        if self.erlang: return False
        return self.weekday in (0, 2, 4) and 14 * 60 <= (t % 1440) < 17 * 60

    def hour(self, t):
        return int(t // 60) % 24

    def clear(self, t, dur):
        """True if no grounded hour starts between t and t+dur (minutes). Grounding comes from the hourly
        gust forecast, so the hub does not launch a flight that would still be in the air when it begins."""
        h0, h1 = int(t // 60), int((t + dur) // 60)
        return not any(self.grounded[h % 24] for h in range(h0, h1 + 1))

    def next_clear(self, t, dur):
        """earliest time >= t at which a flight of `dur` minutes can launch and land before any grounded hour"""
        if self.clear(t, dur): return t
        for k in range(1, 48):
            t2 = (int(t // 60) + k) * 60
            if self.clear(t2, dur): return t2
        return math.inf

    def divert_if_grounded(self, H, t):
        """Gusts are forecast hourly. If an order cannot take off until the grounding ends, and waiting for the sky
        to clear + flying would take longer than the road (emergency, 120 min), send it by road NOW instead of
        waiting for the queue time-out. Routine/lab (target: same day) go by road now only if the drone could not
        take off within their wait limit (6 h) - otherwise they wait for the sky to clear (cheaper)."""
        keep = []
        for q in H.queue:
            o = q[3]
            t_out, t_back, _, ok = self.flight(H.name, o["phc"], t)
            dur = 3.0 + t_out / 60 + 1.0 + t_back / 60
            tc = self.next_clear(t, dur) if ok else t
            road = 120 if o["cls"] in EMERG else 180
            by_drone = (tc - t) + 3.0 + (t_out + (t_back if o["cls"] == "lab" else 0)) / 60 + 1.0
            if o["cls"] in EMERG:   # emergency: whichever is faster
                divert = by_drone > road
            else:                   # routine / lab: target is same day -> wait for the drone if it can still take off within the wait limit
                divert = (tc - o["t0"]) > self.cfg.routine_wait_max
            if ok and tc > t and divert:
                o["mode"] = "road"; o["reason"] = "grounded"; o["t_done"] = t + road; self.records.append(o)
            else:
                keep.append(q)
        H.queue = keep

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
                t += self.rng.exponential(1 / lam); evs.append((t, phc, "em", hub))
        else:
            sol = self.cfg.sol
            bfrac = sol.get("blood_frac", 0.0); bhub = sol.get("blood_hub", "H0")
            bfrom = set(sol.get("blood_from_H0", []))
            for p in dem:
                rates = {"em": (1 - bfrac) * dem[p]["em"], "blood": bfrac * dem[p]["em"], "ro": dem[p]["ro"], "lab": dem[p]["lab"]}
                for c, lam_c in rates.items():
                    if lam_c <= 0: continue
                    if c == "blood":
                        hub = bhub if (p in bfrom and bhub in self.hubs) else None
                    else:
                        hub = self.assign.get(p)
                    prof = hp["em" if c == "blood" else c]
                    for h in range(24):
                        for _ in range(self.rng.poisson(lam_c * prof[h] * self.cfg.demand_mult)):
                            evs.append((h * 60 + self.rng.uniform(0, 60), p, c, hub))
        evs.sort(key=lambda e: e[0])
        for t, p, c, hub in evs:
            yield env.timeout(t - env.now)
            self.new_order(p, c, hub)

    def new_order(self, p, c, hub):
        t = self.env.now
        if hub is None:      # no drone can serve this PHC for this class (out of range / blood out of H0's reach): road
            self.records.append(dict(phc=p, cls=c, hub=None, t0=t, mode="road", reason="unassigned",
                                     t_done=t + (120 if c in EMERG else 180)))
            return
        H = self.hubs[hub]
        o = dict(phc=p, cls=c, hub=hub, t0=t, mode=None)
        self.seq += 1
        pri = 0 if c in EMERG else 1
        if hub == p:   # co-located: hand-carried
            o.update(mode="walk", t_done=t + 5); self.records.append(o); return
        H.queue.append((pri, t, self.seq, o))
        if c in EMERG and not self.erlang:
            self.env.process(self.timeout_road(o, self.cfg.em_wait_max))
        elif not self.erlang:
            self.env.process(self.timeout_road(o, self.cfg.routine_wait_max))
        self.dispatch(H)

    def timeout_road(self, o, wait):
        yield self.env.timeout(wait)
        if o["mode"] is None:
            o["mode"] = "road"; o["reason"] = "waited"
            o["t_done"] = self.env.now + (120 if o["cls"] in EMERG else 180)
            H = self.hubs[o["hub"]]
            H.queue = [q for q in H.queue if q[3] is not o]
            self.records.append(o)

    # ---------------------------------------------------------- blood restock H0 -> hospital hubs (routes from M4)
    def restock(self):
        for k in range(RESUPPLY_PER_DAY):
            yield self.env.timeout(RESTOCK_HOUR * 60 + k * 1440 / RESUPPLY_PER_DAY - self.env.now)
            for h, rr in self.cfg.sol["resupply"].items():
                if rr["mode"] == "drone" and rr["route"][0] in self.hubs:
                    self.env.process(self.restock_run(h, rr["route"]))
                else:   # road van, ~2 h
                    self.restocks.append(dict(hub=h, route=rr["route"], mode="road", t0=self.env.now, t_done=self.env.now + 120))
                    if self.cfg.record:   # van drives H0 -> hub (2 h) and back (2 h)
                        self.trace.append(dict(hub=rr["route"][0], phc=h, cls="van", t0=self.env.now, t1=self.env.now + 120, t2=self.env.now + 240))

    def restock_run(self, h, route):
        """One drone from H0 flies the hop chain (battery swap at each stop), drops the blood box at hub h,
        flies back the same way and is recharged at H0. Waits for an idle H0 drone and flyable wind."""
        env = self.env; H = self.hubs[route[0]]; t_sched = env.now
        while H.idle <= H.reserve_k or self.grounded[self.hour(env.now)]:     # never take an emergency-reserve drone
            yield env.timeout(1.0)
        H.idle -= 1; t_disp = env.now; e_last = 0.0
        legs = list(zip(route[:-1], route[1:]))
        for direction, seq in (("out", legs), ("back", [(b, a) for a, b in reversed(legs)])):
            for i, (a, b) in enumerate(seq):
                while True:
                    t_out, _, _, ok = self.flight(a, b, env.now)
                    if ok and self.clear(env.now, (3.0 if i == 0 else 1.0) + t_out / 60): break
                    yield env.timeout(10.0)                                 # wait at the stop for calmer wind
                t0 = env.now
                yield env.timeout((3.0 if i == 0 else 1.0) + t_out / 60)
                e_last = self.energy(t_out if direction == "out" else 0, t_out if direction == "back" else 0) / 2
                if self.cfg.record:
                    self.trace.append(dict(hub=a, phc=b, cls="restock", t0=t0, t1=env.now, t2=env.now, ow=1))
                if b == h and direction == "out":
                    yield env.timeout(5.0)                                   # hand over the blood box
                    self.restocks.append(dict(hub=h, route=route, mode="drone", t0=t_sched, t_done=env.now))
                elif b != route[0]:
                    yield env.timeout(6.0)                                   # battery swap at the stop
        H.waiting_charge.append(e_last)
        if H.wake is not None and not H.wake.triggered: H.wake.succeed()
        yield H.batteries.get()
        yield env.timeout(6.0)
        H.busy_time += env.now - t_disp; H.idle += 1
        self.dispatch(H)

    # ---------------------------------------------------------- dispatch
    def dispatch(self, H):
        t = self.env.now
        if not self.erlang: self.divert_if_grounded(H, t)
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
            if ok and not self.clear(t, 3.0 + t_out / 60 + 1.0 + t_back / 60):
                k += 1; continue                         # would still be flying when grounding starts: hold it
            if not ok:                                   # cannot be flown in this wind -> road now
                H.queue.pop(k); o["mode"] = "road"; o["reason"] = "wind-range"
                o["t_done"] = t + (120 if o["cls"] in EMERG else 180); self.records.append(o); continue
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
                    self.qlog.append(self.qrow(h, H, self.env.now))
            for H in self.hubs.values(): self.dispatch(H)

    def qrow(self, h, H, t):
        p_W = min(len(H.charging) * 1500.0, 3000.0 if self.outage(t) else 1e9) if H.charging else 0.0   # at the plug
        return (t, h, len(H.queue), H.idle, len(H.charging), len(H.batteries.items), p_W, H.energy_grid_Wh + H.energy_gen_Wh)

    def qticker(self):
        while True:
            yield self.env.timeout(5)
            for h, H in self.hubs.items():
                self.qlog.append(self.qrow(h, H, round(self.env.now, 1)))

    def run(self, horizon=1440 + 300):
        self.env.process(self.arrivals()); self.env.process(self.ticker())
        if not self.erlang and self.cfg.sol.get("resupply"): self.env.process(self.restock())
        if self.cfg.record: self.env.process(self.qticker())
        self.env.run(until=horizon if not self.erlang else self.erlang["horizon_min"] + 600)
        return self.summary()

    def summary(self):
        recs = self.records
        em = [r for r in recs if r["cls"] in EMERG]
        dt = np.array([r["t_done"] - r["t0"] for r in em]) if em else np.array([])
        n_tot = len(recs); road = sum(r["mode"] == "road" for r in recs)
        busy = sum(H.busy_time for H in self.hubs.values()); ndr = sum(H.n for H in self.hubs.values())
        cm = _ROADC
        road_cost = sum((cm["emergency_inr_per_trip"] if r["cls"] in EMERG else cm["routine_lab_inr_per_trip"]) for r in recs if r["mode"] == "road")
        bl = [r for r in em if r["cls"] == "blood"]
        n_bl45 = int(sum((r["t_done"] - r["t0"]) <= 45 for r in bl))
        by_cls = {c: dict(n=sum(r["cls"] == c for r in recs), n_drone=sum(r["cls"] == c and r["mode"] not in ("road", "walk") for r in recs),
                          n_road=sum(r["cls"] == c and r["mode"] == "road" for r in recs),
                          times=np.array([r["t_done"] - r["t0"] for r in recs if r["cls"] == c]))
                  for c in ("em", "blood", "ro", "lab")}
        by_cls["restock"] = dict(n=len(self.restocks), n_drone=sum(r["mode"] == "drone" for r in self.restocks),
                                 n_road=sum(r["mode"] == "road" for r in self.restocks),
                                 times=np.array([r["t_done"] - r["t0"] for r in self.restocks]))
        return dict(n_em=len(em), n_em_45=int((dt <= 45).sum()), em_times=dt, n_orders=n_tot, n_road=road, by_cls=by_cls,
                    n_blood=len(bl), n_blood_45=n_bl45,
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
    nb = sum(r.get("n_blood", 0) for r in res); nb45 = sum(r.get("n_blood_45", 0) for r in res)
    return dict(S45=S, S45_halfwidth=1.96 * se, P95_em_min=float(np.percentile(times, 95)) if len(times) else np.nan,
                S45_blood=nb45 / nb if nb else np.nan, S45_antivenom=(n45 - nb45) / (n_em - nb) if n_em > nb else np.nan,
                mean_em_min=float(times.mean()) if len(times) else np.nan,
                util=float(np.mean([r["util"] for r in res])), road_share=sum(r["n_road"] for r in res) / max(sum(r["n_orders"] for r in res), 1),
                energy_kWh_day=float(np.mean([r["energy_grid_kWh"] + r["energy_gen_kWh"] for r in res])),
                genset_kWh_day=float(np.mean([r["energy_gen_kWh"] for r in res])),
                road_cost_day=float(np.mean([r["road_cost"] for r in res])), n_days=nd, per_day_S45=per_day,
                by_class=class_table(res))


CLASS_NAMES = dict(em="emergency (antivenom etc.)", blood="emergency blood", ro="routine supplies", lab="lab samples PHC->hub",
                   restock="blood restock H0->hubs")


def class_table(res):
    """Every order type, not just emergencies: orders/day, % flown, % by road, mean/P95 time, % within 45 min / 6 h."""
    rows = []
    for c in ("em", "blood", "ro", "lab", "restock"):
        n = sum(r["by_cls"][c]["n"] for r in res)
        if n == 0: continue
        nd_ = sum(r["by_cls"][c]["n_drone"] for r in res)
        nr_ = sum(r["by_cls"][c]["n_road"] for r in res)
        t = np.concatenate([r["by_cls"][c]["times"] for r in res])
        rows.append(dict(order_type=CLASS_NAMES[c], orders_per_day=n / len(res), flown_pct=100 * nd_ / n, road_pct=100 * nr_ / n,
                         mean_min=float(t.mean()), P95_min=float(np.percentile(t, 95)),
                         within_45min_pct=100 * float((t <= 45).mean()), within_6h_pct=100 * float((t <= 360).mean())))
    return rows


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


# =====================================================================================
# DRIVER: verification (M/M/c vs Erlang C), days-for-CI study, live day view (HTML)
# =====================================================================================
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
.hrow{display:grid;grid-template-columns:34px 1fr 1fr 1fr 70px;gap:10px;align-items:end;font-size:12px;margin:5px 0}
.ky{display:inline-block;width:10px;height:10px;border-radius:5px;margin-right:6px;vertical-align:middle}
.kl{display:inline-block;width:22px;height:0;margin-right:6px;vertical-align:middle}
.flag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;margin-right:6px}
</style></head><body><div id="wrap">
<div><canvas id="c" width="780" height="585"></canvas>
<div id="key" style="width:780px;display:grid;grid-template-columns:repeat(3,1fr);gap:4px 14px;font-size:12px;margin-top:8px;color:#cfd8e3">
<div><b>Sites</b></div><div><b>Moving on the map</b></div><div><b>Lines</b></div>
<div><span class="ky" style="background:#f6c343;border-radius:0"></span>open hub (hospital)</div><div><span class="ky" style="background:#ff4d4d"></span>emergency drone (blood / antivenom)</div><div><span class="kl" style="border-top:1px solid #6f8aa6"></span>drone route hub → PHC</div>
<div><span class="ky" style="background:#cfd8e3"></span>PHC</div><div><span class="ky" style="background:#7ee787"></span>routine-supplies drone</div><div><span class="kl" style="border-top:2px dashed #d946ef"></span>blood restock by drone</div>
<div><span class="ky" style="background:#5d6b7a"></span>hospital, not a hub</div><div><span class="ky" style="background:#3dd6f5"></span>lab-sample drone (PHC → hub)</div><div><span class="kl" style="border-top:2px dotted #d946ef"></span>blood restock by road</div>
<div><span style="color:#e6395a">★</span> blood bank (H0)</div><div><span class="ky" style="background:#d946ef;border:1px solid #fff"></span>blood restock drone</div><div><span class="ky" style="background:rgba(220,60,60,.4);border-radius:0"></span>no-fly zone</div>
<div><span class="ky" style="background:#f6c343;transform:rotate(45deg);border-radius:0"></span>battery-swap relay</div><div><span class="ky" style="background:#f59e0b;border-radius:0;width:14px"></span>blood van (06:00 restock)</div><div><span class="kl" style="border-top:2px dashed #f59e0b"></span>road route in use now</div>
<div></div><div><span class="ky" style="background:#7ee787;border:2px solid #f59e0b;border-radius:0"></span>order sent by road (fill = type)</div><div></div>
</div></div>
<div id="side"><h1>SkyLink — Kaveripuram, one simulated day</h1>
<div class="muted" id="meta"></div>
<div style="margin:10px 0"><button id="pp">Pause</button><button data-s="1">1×</button><button data-s="4">4×</button><button data-s="12">12×</button>
<input type="range" id="t" min="0" max="1440" value="0" style="width:100%;margin-top:8px"></div>
<div style="font-size:28px;font-weight:600" id="clock">00:00</div>
<div id="flags"></div>
<div class="kpi"><div class="k">Drones in air now<b id="air">0</b></div><div class="k">Vehicles on the road now<b id="onroad">0</b></div>
<div class="k">Emergencies delivered ≤45 min<b id="s45">–</b></div><div class="k">Total sent by road so far<b id="road">0</b></div>
<div class="k">Charging power now (all hubs added)<b id="pw">0 kW</b></div><div class="k">Charging energy so far<b id="ew">0 kWh</b></div></div>
<div class="muted">All orders so far, by type (placed = done by drone + sent by road + open)</div><table id="cls" style="width:100%;font-size:12px;border-collapse:collapse;margin:4px 0 6px"></table>
<div id="rst" style="font-size:12px;margin:0 0 10px"></div>
<div class="muted" style="margin-bottom:4px">Per hub</div>
<div class="hrow muted" style="font-size:11px"><span></span><span><i class="ky" style="background:#ed8936;border-radius:0"></i>orders queued</span><span><i class="ky" style="background:#48bb78;border-radius:0"></i>idle drones / total</span><span><i class="ky" style="background:#4299e1;border-radius:0"></i>packs charging</span><span>spare packs · kW now</span></div>
<div id="hubs"></div>
</div></div>
<script>
const D = __DATA__;
const COL={em:'#ff4d4d',blood:'#ff9f1c',ro:'#7ee787',lab:'#3dd6f5',restock:'#d946ef'};
const NAME={em:'Emergency (blood / antivenom from the hub)',blood:'Emergency blood from H0',ro:'Routine supplies',lab:'Lab samples',restock:'Blood restock H0 → hubs (06:00)'};
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
 (D.rlegs||[]).forEach(r=>{cx.strokeStyle='rgba(217,70,239,.8)';cx.setLineDash(r.mode=='drone'?[7,4]:[2,4]);cx.lineWidth=r.mode=='drone'?2:1.2;cx.beginPath();
  const p=D.paths[r.k]||[r.a,r.b];p.forEach((q,i)=>i?cx.lineTo(X(q[0]),Y(q[1])):cx.moveTo(X(q[0]),Y(q[1])));cx.stroke();cx.setLineDash([]);cx.lineWidth=1});
 D.sites.forEach(s=>{const hub=D.hubs.includes(s.id);cx.fillStyle=hub?'#f6c343':(s.id[0]=='P'?'#cfd8e3':'#5d6b7a');
  if(hub){cx.fillRect(X(s.x)-7,Y(s.y)-7,14,14)}else{cx.beginPath();cx.arc(X(s.x),Y(s.y),3.5,0,7);cx.fill()}
  if((D.relays||[]).includes(s.id)&&!hub){cx.fillStyle='#f6c343';cx.beginPath();cx.moveTo(X(s.x),Y(s.y)-7);cx.lineTo(X(s.x)+7,Y(s.y));cx.lineTo(X(s.x),Y(s.y)+7);cx.lineTo(X(s.x)-7,Y(s.y));cx.fill()}
  if(s.id==D.bhub){cx.fillStyle='#e6395a';cx.font='16px sans-serif';cx.fillText('★',X(s.x)-6,Y(s.y)+22)}
  cx.fillStyle='#8a99a8';cx.font='10px sans-serif';cx.fillText(s.id+((D.relays||[]).includes(s.id)&&!hub?' (swap relay)':''),X(s.x)+6,Y(s.y)-5)});
 D.trips.forEach(r=>{if(t<r.t0||t>r.t2||!(r.cls=='van'||r.cls.startsWith('road_')))return;const p=D.paths[r.hub+'-'+r.phc];if(!p)return;
  cx.strokeStyle='#f59e0b';cx.lineWidth=2;cx.setLineDash([3,4]);cx.beginPath();p.forEach((q,i)=>i?cx.lineTo(X(q[0]),Y(q[1])):cx.moveTo(X(q[0]),Y(q[1])));cx.stroke();cx.setLineDash([]);cx.lineWidth=1});
 let air=0,onroad=0;const inair={};D.trips.forEach(r=>{if(t<r.t0||t>r.t2)return;const p=D.paths[r.hub+'-'+r.phc];if(!p)return;
  if(r.cls.startsWith('road_')){const q=t<r.t1?pos(p,(t-r.t0)/(r.t1-r.t0)):pos(p,1-(t-r.t1)/(r.t2-r.t1));const c=r.cls.slice(5);
   cx.fillStyle=COL[c];cx.strokeStyle='#f59e0b';cx.lineWidth=2;cx.fillRect(X(q[0])-4,Y(q[1])-4,8,8);cx.strokeRect(X(q[0])-4,Y(q[1])-4,8,8);cx.lineWidth=1;onroad++;return}
  if(r.cls=='van'){const q=t<r.t1?pos(p,(t-r.t0)/(r.t1-r.t0)):pos(p,1-(t-r.t1)/(r.t2-r.t1));cx.fillStyle='#f59e0b';cx.strokeStyle='#fff';cx.fillRect(X(q[0])-7,Y(q[1])-5,14,10);cx.strokeRect(X(q[0])-7,Y(q[1])-5,14,10);
   cx.fillStyle='#f59e0b';cx.font='10px sans-serif';cx.fillText(t<r.t1?'blood van':'van returning',X(q[0])+9,Y(q[1])+4);onroad++;return}
  air++;
  let q; if(r.ow||t<r.t1){q=pos(p,Math.min(1,(t-r.t0)/Math.max(r.t1-r.t0,1e-6)))}else{q=pos(p,1-(t-r.t1)/(r.t2-r.t1))}
  if(r.cls=='restock'){cx.fillStyle='#d946ef';cx.strokeStyle='#fff';cx.beginPath();cx.arc(X(q[0]),Y(q[1]),6,0,7);cx.fill();cx.stroke();inair.restock=(inair.restock||0)+1;return}
  cx.fillStyle=COL[r.cls]||'#fff';cx.strokeStyle='#0b1118';cx.lineWidth=1.5;cx.beginPath();cx.arc(X(q[0]),Y(q[1]),5,0,7);cx.fill();cx.stroke();cx.lineWidth=1;inair[r.cls]=(inair[r.cls]||0)+1});
 let rows='';for(const c of ['em','blood','ro','lab']){const o=D.ords.filter(e=>e.c==c);if(!o.length)continue;
  const placed=o.filter(e=>e.t0<=t),dd=placed.filter(e=>e.m!='road'&&e.td<=t).length,rd=placed.filter(e=>e.m=='road'&&e.td-(c=='em'||c=='blood'||c=='restock'?120:180)<=t).length;
  rows+=`<tr><td><span style="display:inline-block;width:10px;height:10px;border-radius:5px;background:${COL[c]};margin-right:6px"></span>${NAME[c]}</td><td>${placed.length}</td><td>${dd}</td><td>${rd}</td><td>${placed.length-dd-rd}</td></tr>`}
 document.getElementById('cls').innerHTML='<tr class="muted"><td>order type</td><td>placed</td><td>done by drone</td><td>sent by road</td><td>open (queued / in air)</td></tr>'+rows;
 const rs=D.ords.filter(e=>e.c=='restock'&&e.t0<=t);
 document.getElementById('rst').innerHTML=rs.length?`<span class="ky" style="background:${COL.restock}"></span>Blood restock H0 → hubs (stock top-up, not a PHC order): ${rs.filter(e=>e.m!='road').length} by drone, ${rs.filter(e=>e.m=='road').length} by van, ${rs.filter(e=>e.td<=t).length} of ${rs.length} arrived`:'<span class="muted">Blood restock H0 → hubs starts at 06:00</span>';
 const h=Math.floor(t/60)%24,m=Math.floor(t%60);document.getElementById('clock').textContent=String(h).padStart(2,'0')+':'+String(m).padStart(2,'0');
 document.getElementById('air').textContent=air;
 const em=D.em.filter(e=>e.t0<=t);const done=em.filter(e=>e.td<=t);const ok=done.filter(e=>e.td-e.t0<=45);
 document.getElementById('s45').textContent=done.length?Math.round(100*ok.length/done.length)+'%':'–';
 {const no=D.road.filter(x=>x<=t).length,nv=D.ords.filter(e=>e.c=='restock'&&e.m=='road'&&e.t0<=t).length;
  document.getElementById('road').innerHTML=(no+nv)+' <span class="muted" style="font-size:12px;font-weight:400">('+no+' PHC orders + '+nv+' blood vans)</span>';}
 let f='';if(D.grounded[h])f+='<span class="flag" style="background:#c53030">GROUNDED (gust ≥14 m/s)</span>';
 if(D.outage&&t>=840&&t<1020)f+='<span class="flag" style="background:#b7791f">GRID OUTAGE – each hub on its own 3 kW genset</span>';
 f+='<span class="flag" style="background:#2d3748">mean wind '+D.wind[h].toFixed(1)+' m/s · gust '+D.gust[h].toFixed(1)+' m/s (grounded at ≥14)</span>';
 document.getElementById('air').textContent=air;document.getElementById('onroad').textContent=onroad;document.getElementById('flags').innerHTML=f;
 let hh='';D.hubs.forEach(hb=>{const L=D.q[hb];let i=0;while(i<L.length-1&&L[i+1][0]<=t)i++;const s=L[i]||[0,0,0,0,0];const n=D.n[hb];
  hh+=`<div class="hrow"><b>${hb}</b>
  <span>${s[1]}<div class="bar"><span style="width:${Math.min(100,s[1]*10)}%;background:#ed8936"></span></div></span>
  <span>${s[2]}/${n}<div class="bar"><span style="width:${100*s[2]/n}%;background:#48bb78"></span></div></span>
  <span>${s[3]}<div class="bar"><span style="width:${Math.min(100,s[3]*25)}%;background:#4299e1"></span></div></span>
  <span>${s[4]} · ${((s[5]||0)/1000).toFixed(1)}</span></div>`});
 document.getElementById('hubs').innerHTML=hh;
 let PW=0,EW=0;D.hubs.forEach(hb=>{const L=D.q[hb];let i=0;while(i<L.length-1&&L[i+1][0]<=t)i++;const s=L[i]||[];PW+=s[5]||0;EW+=s[6]||0});
 document.getElementById('pw').innerHTML=(PW/1000).toFixed(1)+' kW'+(D.outage&&t>=840&&t<1020?' <span class="muted" style="font-size:12px;font-weight:400">genset, max 3 kW at each hub</span>':'');
 document.getElementById('ew').textContent=(EW/1000).toFixed(1)+' kWh';sl.value=t;
}
function loop(ts){if(last!==null&&play){t+=(ts-last)/1000*speed;if(t>1440)t=0}last=ts;draw();requestAnimationFrame(loop)}
requestAnimationFrame(loop);
</script></body></html>"""


def live_view(cfg, day, path_out, seed=3, label=""):
    s, summ = record_day(cfg, day, seed)
    S = data.sites()
    paths = {}
    for h, info in cfg.sol["hubs"].items():
        for p in info["phcs"]:
            if p != h: paths[f"{h}-{p}"] = [list(map(float, q)) for q in data.path(h, p)]
    rlegs = []                                       # blood restock hop legs H0 -> stop -> hub
    for h, rr in cfg.sol.get("resupply", {}).items():
        for a, b in zip(rr["route"][:-1], rr["route"][1:]):
            if rr["mode"] == "drone":
                for u, v in ((a, b), (b, a)): paths[f"{u}-{v}"] = [list(map(float, q)) for q in data.path(u, v)]
                rlegs.append(dict(k=f"{a}-{b}", mode="drone"))
            else:
                rlegs.append(dict(k=f"{a}-{b}", mode="road", a=[float(S.loc[a].x_km), float(S.loc[a].y_km)], b=[float(S.loc[b].x_km), float(S.loc[b].y_km)]))
                paths[f"{a}-{b}"] = [list(map(float, q)) for q in data.path(a, b)]   # van route (around the no-go zones)
    relays = sorted({n for rr in cfg.sol.get("resupply", {}).values() for n in rr["route"][1:-1]})
    bhub = cfg.sol.get("blood_hub", "H0")
    q = {h: [] for h in cfg.hubs}
    for (t, h, ql, idle, ch, bat, pw, ewh) in s.qlog:
        q[h].append([t, ql, idle, ch, bat, round(pw), round(ewh)])
    em = [dict(t0=r["t0"], td=r["t_done"]) for r in s.records if r["cls"] in EMERG]
    road = [r["t0"] for r in s.records if r["mode"] == "road"]
    for r in s.records:                                   # orders sent by road: a vehicle from the hub to the PHC and back
        if r["mode"] == "road" and r.get("hub") and r["hub"] != r["phc"]:
            dur = 120 if r["cls"] in EMERG else 180
            s.trace.append(dict(hub=r["hub"], phc=r["phc"], cls="road_" + r["cls"], t0=r["t_done"] - dur, t1=r["t_done"], t2=r["t_done"] + dur))
    ords = [dict(c=r["cls"], t0=round(r["t0"], 1), td=round(r["t_done"], 1), m=r["mode"]) for r in s.records] + \
           [dict(c="restock", t0=round(r["t0"], 1), td=round(r["t_done"], 1), m=r["mode"]) for r in s.restocks]
    W = data.wind(); wd = W[W.day == day].sort_values("hour")
    D = dict(meta=f"{label} · day {day} of the wind year ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][s.weekday]}) · "
                  f"{summ['n_orders']} orders, {summ['n_em']} emergencies, S45 = {summ['n_em_45']/max(summ['n_em'],1):.0%}, "
                  f"{summ['grounded_hours']} grounded hours" + (" · wind data SYNTHETIC" if data.sources()["wind_days.csv"] != "real" else ""),
             zones=[[list(map(float, c)) for c in z.exterior.coords] for z in data.nofly().values()],
             sites=[dict(id=k, x=float(r.x_km), y=float(r.y_km)) for k, r in S.iterrows()], hubs=cfg.hubs,
             paths=paths, trips=[{k: (round(v, 2) if isinstance(v, float) else v) for k, v in tr.items()} for tr in s.trace],
             rlegs=rlegs, relays=relays, bhub=bhub, ords=ords,
             q=q, n={h: s.hubs[h].n for h in cfg.hubs}, em=em, road=road, grounded=[bool(x) for x in s.grounded],
             wind=[float(x) for x in s.w], gust=[float(x) for x in s.gust], outage=bool(s.weekday in (0, 2, 4)))
    open(path_out, "w", encoding="utf-8").write(HTML.replace("__DATA__", json.dumps(D, default=lambda o: o.item() if hasattr(o, "item") else str(o))))
    return summ


def pick_story_day():
    """a weekday with a grid outage and some (not total) grounding -> shows every event type"""
    W = data.wind()
    g = W.groupby("day").agg(gr=("grounded", "sum"), wk=("weekday", "first"), m=("month", "first"))
    hrs = W[W.grounded.astype(bool)].groupby("day").hour.apply(set)
    def ok(d):
        h = hrs.get(d, set())
        return not (h & {14, 15, 16}) and len(h & set(range(7, 20))) >= 2      # daytime grounding, separate from the outage
    c = g[(g.wk.isin([0, 2, 4])) & (g.gr >= 2) & (g.gr <= 5)]
    c = [d for d in c.index if ok(d)]
    return int(c[0]) if c else 1


def run_report(sol, vp, reserve=0.2, tag="m5", label=""):
    tabd = os.path.join(data.OUT, "tables"); figd = os.path.join(data.OUT, "figs")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    mmc = [verify_mmc(vp, c=c, lam_per_hr=lam, mean_service_min=svc, reps=40, horizon_hr=4000, seed0=1000 * k)
           for k, (c, lam, svc) in enumerate([(1, 2.0, 20.0), (3, 6.0, 24.0), (5, 10.0, 25.0)])]
    pd.DataFrame(mmc).to_csv(os.path.join(tabd, f"{tag}_mmc_verification.csv"), index=False)
    cfg = Config(sol, vp, reserve=reserve)
    ev, hist = days_for_ci(cfg)
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
                story_day=day, story_summary={k: v for k, v in lv.items() if k not in ("em_times", "by_cls")})
    json.dump(summ, open(os.path.join(tabd, f"{tag}_summary.json"), "w", encoding="utf-8"), indent=1, default=float)
    return summ


if __name__ == "__main__":
    for _d in ("figs", "tables", "live"):
        os.makedirs(os.path.join(data.OUT, _d), exist_ok=True)
    # network: from your last M4 run if present, else designed now (M1 at R_REQ + M4 MILP)
    p4 = os.path.join(data.OUT, "tables", "m4_solutions.json")
    if os.path.exists(p4):
        sols = json.load(open(p4, encoding="utf-8")); name = [k for k in sols if not k.endswith("forced")][0]; sol = sols[name]
        print(f"Using the M4 network in {p4}")
    else:
        from . import m1_vehicle as m1, m4_network as m4
        p2 = os.path.join(data.OUT, "tables", "m2_summary.json")
        mp = json.load(open(p2, encoding="utf-8"))["picked_plate_mass_kg"] if os.path.exists(p2) else 0.25
        _, d, _ = m1.run(m_plate=mp, fm=None, R_req=m1.R_REQ, tag="m1", quick=True)
        _, sols, _ = m4.run(d, mp, None); name = d.index[0]; sol = sols[name]
        print(f"No M4 result found: designed the network now for R_REQ = {m1.R_REQ:g} km")
    s = run_report(sol, sol["vp"], label=f"{name} network (M4 optimum)")
    pd.set_option("display.width", 200)
    print(f"\n=== M5 on the M4 network: hubs {list(sol['hubs'])}, {sum(h['drones'] for h in sol['hubs'].values())} drones, "
          f"road-only PHCs {len(sol.get('road_only', []))} ===")
    print("\nVerification - M/M/c (one hub, exponential service, no wind, no outage) vs Erlang C:")
    print(pd.DataFrame(s["mmc"])[["c", "lam_per_hr", "mean_service_min", "rho", "Wq_theory_min", "Wq_sim_min", "Wq_sim_ci95", "rel_err"]].round(3).to_string(index=False))
    ci = s["ci"]
    print(f"\nDays for a 95% CI on S45 < +-3 pp: {ci['days']} random days of the wind year (added 12 at a time)")
    split = (f"  (blood {100*ci['S45_blood']:.1f}%, antivenom {100*ci['S45_antivenom']:.1f}%)" if ci["S45_blood"] == ci["S45_blood"]
             else "  (all emergencies incl. blood restocked at the hubs)")
    print(f"S45 = {100*ci['S45']:.1f}% +- {100*ci['S45_halfwidth']:.1f} pp{split}"
          f" | P95 emergency time = {ci['P95_em_min']:.1f} min | mean = {ci['mean_em_min']:.1f} min")
    print(f"Drone utilisation = {100*ci['util']:.1f}% | road-fallback share = {100*ci['road_share']:.1f}% of orders | "
          f"energy = {ci['energy_kWh_day']:.1f} kWh/day ({ci['genset_kWh_day']:.1f} from genset) | road cost = Rs {ci['road_cost_day']:.0f}/day")
    tab = pd.DataFrame(ci["by_class"]).round(1)
    print("\nAll drone orders by type (per simulated day; time = order placed -> delivered):")
    print(tab.to_string(index=False))
    tab.to_csv(os.path.join(data.OUT, "tables", "m5_by_class.csv"), index=False)
    print(f"Live view: outputs/live/skylink_day.html (day {s['story_day']} - open it in a browser)")
    print("Saved: outputs/tables/m5_mmc_verification.csv, m5_ci_days.csv, m5_by_class.csv, m5_summary.json; outputs/figs/m5_ci.png")
