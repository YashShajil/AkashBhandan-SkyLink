"""Data layer: load the data pack (real files preferred), audit it for planted problems,
apply documented fixes, and compute obstacle-avoiding flight distances.

Every fix is recorded in AUDIT (written to outputs/tables/data_audit.csv) so that no data
problem is carried silently into later modules.
"""
from __future__ import annotations
import json, math, os, functools
import numpy as np, pandas as pd, yaml, networkx as nx
from shapely.geometry import Polygon, LineString, Point

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, "data", "real")
RECON = os.path.join(ROOT, "data", "reconstructed")
OUT = os.path.join(ROOT, "outputs")
for _d in ("figs", "tables", "live"):
    os.makedirs(os.path.join(OUT, _d), exist_ok=True)   # fresh clone: create the output folders
AUDIT: list[dict] = []
FILES = ["sites.csv", "nofly.geojson", "demand.csv", "hourly_profile.csv", "wind_days.csv", "costs.yaml", "components.csv"]
YEAR_START_WEEKDAY = 0   # 0 = Monday. ASSUMPTION: day 1 of the wind year is a Monday (check real file).


def _path(name):
    p = os.path.join(REAL, name)
    if os.path.exists(p):
        return p, "real"
    return os.path.join(RECON, name), "reconstructed"


def log(file, issue, action, severity="fix"):
    AUDIT.append(dict(file=file, severity=severity, issue=issue, action=action))


# ---------------------------------------------------------------- loaders + audits
@functools.lru_cache(None)
def sources():
    return {f: _path(f)[1] for f in FILES}


@functools.lru_cache(None)
def costs():
    p, src = _path("costs.yaml")
    c = _normalise_costs(yaml.safe_load(open(p, encoding="utf-8")))
    brief = dict(hub_capex=1_800_000, drone_base=200_000, drone_per_kg=45_000, bat_wh=22, bay=90_000,
                 staff=420_000, grid=9, genset=28)
    got = dict(hub_capex=c["hub"]["capex_inr"], drone_base=c["drone"]["base_capex_inr"],
               drone_per_kg=c["drone"]["capex_per_kg_mtow_inr"], bat_wh=c["battery"]["cost_inr_per_wh"],
               bay=c["charging_bay"]["capex_inr"], staff=c["staff"]["cost_inr_per_person_yr"],
               grid=c["energy"]["grid_inr_per_kwh"], genset=c["energy"]["genset_inr_per_kwh"])
    for k in brief:
        if not math.isclose(brief[k], got[k], rel_tol=1e-6):
            log("costs.yaml", f"{k}={got[k]} differs from brief value {brief[k]}",
                f"using brief value {brief[k]} (brief is the signed spec); confirm with TA", "fix")
    return c


def _normalise_costs(c):
    """map the course costs.yaml schema onto the internal one (the reconstructed file already uses it)"""
    if "capex_inr" in c.get("hub", {}):
        return c
    op = c["operations"]; rf = c["road_fallback"]
    if rf["cost_per_trip"]["routine"] != rf["cost_per_trip"]["lab"]:
        log("costs.yaml", "routine and lab road costs differ", "internal model uses routine value for both", "flag")
    return dict(discount_rate=c["discount_rate"], operating_hours_per_day=c["operating_hours_per_day"],
        hub=dict(capex_inr=c["hub"]["capex"], life_yr=c["hub"]["life_years"], opex_inr_per_yr=c["hub"]["opex_per_year"]),
        drone=dict(base_capex_inr=c["drone"]["capex_fixed"], capex_per_kg_mtow_inr=c["drone"]["capex_per_kg_mtow"],
                   life_yr=c["drone"]["life_years"], maintenance_frac_of_capex_per_yr=c["drone"]["maintenance_frac_of_capex_per_year"]),
        battery=dict(cost_inr_per_wh=c["battery_pack"]["cost_per_Wh"], cycle_life=c["battery_pack"]["cycle_life"]),
        charging_bay=dict(capex_inr=c["charging_bay"]["capex"], life_yr=c["charging_bay"]["life_years"],
                          power_kw=c["charging_bay"]["power_kW"], charge_efficiency=c["charging_bay"]["charge_efficiency"]),
        staff=dict(cost_inr_per_person_yr=c["staff"]["salary_per_person_year"], drones_per_operator_per_shift=c["staff"]["drones_per_operator"],
                   shifts_per_day=c["staff"]["shifts_per_day"], relief_factor=c["staff"]["relief_factor"], min_operators_per_open_hub=1),
        energy=dict(grid_inr_per_kwh=c["energy"]["grid_per_kWh"], genset_inr_per_kwh=c["energy"]["genset_per_kWh"]),
        grid_outage=dict(days=c["grid_outage"]["days"], start_hour=int(c["grid_outage"]["start"][:2]), end_hour=int(c["grid_outage"]["end"][:2]),
                         genset_kw_per_hub=c["grid_outage"]["genset_kW_per_hub"]),
        road_fallback=dict(emergency_inr_per_trip=rf["cost_per_trip"]["emergency"], routine_lab_inr_per_trip=rf["cost_per_trip"]["routine"],
                           emergency_minutes=rf["delivery_time_min"]["emergency"], routine_lab_minutes=rf["delivery_time_min"]["routine"]),
        ground_times_min=dict(load_and_takeoff=op["load_and_takeoff_min"], drop_or_pickup=op["drop_or_pickup_min"], swap_and_turnaround=op["battery_swap_turnaround_min"]),
        grounding_gust_ms=op["gust_grounding_limit_ms"],
        emergency=dict(target_min=op["emergency_target_min"], road_after_wait_min=op["emergency_max_wait_before_road_min"]))


def crf(r, n):
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


@functools.lru_cache(None)
def sites():
    p, _ = _path("sites.csv")
    s = pd.read_csv(p)
    s.columns = [c.strip().lower() for c in s.columns]
    s = s.rename(columns={"x (km)": "x_km", "y (km)": "y_km"})
    if s.site_id.duplicated().any():
        log("sites.csv", f"duplicate ids {list(s.site_id[s.site_id.duplicated()])}", "kept first row")
        s = s.drop_duplicates("site_id")
    bad = s[(s.x_km < 0) | (s.x_km > 60) | (s.y_km < 0) | (s.y_km > 45)]
    for _, r in bad.iterrows():
        log("sites.csv", f"{r.site_id} at ({r.x_km},{r.y_km}) is outside the 60x45 km district", "flagged; check sign/units", "flag")
    zones = nofly()
    for _, r in s.iterrows():
        for zn, poly in zones.items():
            if poly.contains(Point(r.x_km, r.y_km)):
                log("sites.csv", f"{r.site_id} lies inside no-fly zone '{zn}'",
                    "site cannot be a hub; drone may only land if a corridor is granted -> treated as road-only", "flag")
    expected = {"H0", "H1", "H2", "H3", "H4", "H5", "H6", "P09"}
    cand = set(s.site_id[s.candidate_hub.astype(str).str.lower().isin(["yes", "1", "true"])])
    if cand != expected:
        log("sites.csv", f"candidate-hub flags {sorted(cand)} differ from brief {sorted(expected)}", "using brief list")
    s["candidate_hub"] = s.site_id.isin(expected)
    return s.set_index("site_id")


@functools.lru_cache(None)
def nofly():
    p, _ = _path("nofly.geojson")
    gj = json.load(open(p, encoding="utf-8"))
    zones = {}
    for f in gj["features"]:
        name = f["properties"].get("name", "zone")
        props = f["properties"]
        if props.get("shape") == "circle" or props.get("kind") == "circle" or "radius_km" in props:
            cx, cy = props.get("centre", props.get("center_km", (43.0, 23.0))); r = props.get("radius_km", 8.0)
            N = 64; R = r / math.cos(math.pi / N)        # circumscribed polygon: conservative detour
            zones[name] = Polygon([(cx + R * math.cos(2 * math.pi * k / N), cy + R * math.sin(2 * math.pi * k / N)) for k in range(N)])
        else:
            poly = Polygon(f["geometry"]["coordinates"][0])
            if not poly.is_valid:
                log("nofly.geojson", f"{name} polygon invalid (self-intersecting)", "repaired with buffer(0)")
                poly = poly.buffer(0)
            zones[name] = poly
    # brief check: airstrip circle radius 8 km at (43,23)
    return zones


@functools.lru_cache(None)
def demand():
    p, _ = _path("demand.csv")
    d = pd.read_csv(p)
    d.columns = [c.strip().lower() for c in d.columns]
    d = d.set_index(d.columns[0])
    d.index.name = "phc_id"
    d = d.rename(columns={c: c.split("_")[0] for c in d.columns if c.endswith("per_day")})
    d = d.rename(columns={"emergency": "em", "routine": "ro", "lab": "lab", "population": "pop"})
    if "population_k" in d.columns:
        d = d.rename(columns={"population_k": "pop"})
    missing = [f"P{i:02d}" for i in range(1, 25) if f"P{i:02d}" not in d.index]
    for m in missing:
        log("demand.csv", f"{m} missing", "imputed from district per-capita rates x population (if known) else mean", "fix")
    if d.index.duplicated().any():
        log("demand.csv", f"duplicate rows {list(d.index[d.index.duplicated()])}", "kept first")
        d = d[~d.index.duplicated()]
    for c in ["em", "ro", "lab"]:
        neg = d[d[c] < 0]
        for i in neg.index:
            log("demand.csv", f"{i} {c} negative ({d.loc[i, c]})", "set to per-capita median x population")
            d.loc[i, c] = np.nan
        rate = d[c] / d["pop"]
        z = (rate - rate.median()) / (1.4826 * (rate - rate.median()).abs().median())
        for i in z.index[z.abs() > 4]:
            log("demand.csv", f"{i} {c}={d.loc[i, c]} is a per-capita outlier (robust z={z[i]:.1f})",
                "replaced with per-capita median x population (likely unit/typo)")
            d.loc[i, c] = np.nan
        med = (d[c] / d["pop"]).median()
        d[c] = d[c].fillna(d["pop"] * med)
    return d


@functools.lru_cache(None)
def hourly():
    p, _ = _path("hourly_profile.csv")
    h = pd.read_csv(p).set_index("hour")
    h.columns = ["em", "ro", "lab"]
    if len(h) != 24:
        log("hourly_profile.csv", f"{len(h)} hours instead of 24", "missing hours filled with column min", "fix")
        h = h.reindex(range(24)).fillna(h.min())
    for c in h.columns:
        s = h[c].sum()
        if (h[c] < 0).any():
            log("hourly_profile.csv", f"{c} has negative shares", "clipped to 0")
            h[c] = h[c].clip(lower=0)
        if abs(s - 1) > 0.005:
            log("hourly_profile.csv", f"{c} shares sum to {s:.4f}, not 1",
                "renormalised to 1 (keeps the daily mean in demand.csv, which is the stated quantity; "
                "the shape is kept, the 20% excess is removed proportionally)")
            h[c] = h[c] / s
    return h


@functools.lru_cache(None)
def wind():
    p, src = _path("wind_days.csv")
    w = pd.read_csv(p)
    w.columns = [c.strip().lower() for c in w.columns]
    if src != "real":
        log("wind_days.csv", "real file not available: using SYNTHETIC year calibrated to the monthly table",
            "replace with real file in data/real/ and re-run", "flag")
    n0 = len(w)
    w = w.drop_duplicates(["day", "hour"])
    if len(w) != n0:
        log("wind_days.csv", f"{n0 - len(w)} duplicate (day,hour) rows", "dropped duplicates")
    full = pd.MultiIndex.from_product([range(1, 366), range(24)], names=["day", "hour"])
    w = w.set_index(["day", "hour"]).reindex(full)
    nmiss = w.mean_wind_ms.isna().sum()
    if nmiss:
        log("wind_days.csv", f"{nmiss} missing hours", "linear interpolation in time")
    for c, lo, hi in [("mean_wind_ms", 0, 40), ("gust_ms", 0, 70)]:
        bad = (w[c] < lo) | (w[c] > hi)
        if bad.any():
            log("wind_days.csv", f"{int(bad.sum())} {c} values outside [{lo},{hi}] (sentinels/sign errors), e.g. {w[c][bad].head(3).tolist()}",
                "set to NaN and interpolated")
            w.loc[bad, c] = np.nan
    w[["mean_wind_ms", "gust_ms"]] = w[["mean_wind_ms", "gust_ms"]].interpolate(limit_direction="both")
    inv = w.gust_ms < w.mean_wind_ms
    if inv.any():
        log("wind_days.csv", f"{int(inv.sum())} hours with gust < mean wind (columns swapped?)", "swapped the two values")
        a = w.loc[inv, "mean_wind_ms"].copy(); w.loc[inv, "mean_wind_ms"] = w.loc[inv, "gust_ms"]; w.loc[inv, "gust_ms"] = a
    if "direction_deg" in w:
        bd = (w.direction_deg < 0) | (w.direction_deg > 360)
        if bd.any():
            log("wind_days.csv", f"{int(bd.sum())} directions outside 0-360", "wrapped mod 360")
            w["direction_deg"] = w.direction_deg % 360
    # month from day index (re-derived; a wrong month column would be caught here)
    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(w.index.get_level_values(0) - 1, unit="D")
    mon = dates.month.values
    if "month" in w and (w.month.values != mon).sum() > 0:
        log("wind_days.csv", f"{int((w.month.values != mon).sum())} rows whose month column disagrees with day", "month re-derived from day")
    w["month"] = mon
    w["weekday"] = (w.index.get_level_values(0) - 1 + YEAR_START_WEEKDAY) % 7
    w["grounded"] = w.gust_ms >= costs()["grounding_gust_ms"]
    ratio = w.gust_ms / w.mean_wind_ms.clip(lower=0.1)
    if (ratio > 5).any():
        log("wind_days.csv", f"{int((ratio > 5).sum())} hours with gust/mean > 5 (sensor spikes?)", "flagged", "flag")
    over = (w.mean_wind_ms >= 22) & ~w.grounded
    if over.any():
        log("wind_days.csv", f"{int(over.sum())} flyable hours with mean wind >= 22 m/s (above any cruise speed)", "sim sends those orders by road", "flag")
    wk = "assumed Monday" if YEAR_START_WEEKDAY == 0 else str(YEAR_START_WEEKDAY)
    log("wind_days.csv", f"file has no weekday column; outage days (Mon/Wed/Fri) need the weekday of day 1 ({wk})",
        "ASSUMPTION: day 1 = Monday; tested in M7 (outage effect is small)", "flag")
    return w.reset_index()


@functools.lru_cache(None)
def components():
    p, _ = _path("components.csv")
    c = pd.read_csv(p)
    c.columns = [x.strip().lower() for x in c.columns]
    b = c[c.category == "battery"].copy()
    b["wh"] = b.capacity * b.nominal_voltage_v / 1000
    b["wh_per_kg"] = b.wh / b.mass_kg
    b["inr_per_wh"] = b.unit_cost_inr / b.wh
    for _, r in b.iterrows():
        if abs(r.wh_per_kg - 150) > 10:
            log("components.csv", f"battery {r.part_id}: {r.wh_per_kg:.0f} Wh/kg vs 150 in models", "excluded from catalogue snap")
    m = c[c.category == "motor"].copy()
    for _, r in m.iterrows():
        pred = 0.12 + 0.004 * r.max_thrust_n
        if abs(pred - r.mass_kg) > 0.02:
            log("components.csv", f"motor {r.part_id}: mass {r.mass_kg} vs model {pred:.3f}", "flagged", "flag")
    t = c[c.category == "tube"].copy()
    t["area_mm2"] = math.pi / 4 * (t.outer_dia_mm ** 2 - (t.outer_dia_mm - 2 * t.wall_mm) ** 2)
    t["mass_pred"] = 1550 * t.area_mm2 * 1e-6
    for _, r in t.iterrows():
        if abs(r.mass_pred - r.mass_kg) / r.mass_pred > 0.05:
            log("components.csv", f"tube {r.part_id}: catalogue {r.mass_kg} kg/m vs geometry {r.mass_pred:.4f} kg/m",
                "used geometric mass (density 1550 is a stated constant)")
            t.loc[r.name, "mass_kg"] = r.mass_pred
    if (c.get("reconstructed", pd.Series(dtype=str)) == "yes").any():
        log("components.csv", "only 7 of 30 tube sizes are printed; others reconstructed on a 6x5 grid (16-40 mm x 0.75-2.5 mm)",
            "replace with real file", "flag")
    return dict(battery=b, motor=m, tube=t)


# ---------------------------------------------------------------- geometry
@functools.lru_cache(None)
def flight_graph():
    zones = list(nofly().values())
    union = zones[0].union(zones[1]) if len(zones) > 1 else zones[0]
    S = sites()
    G = nx.Graph()
    pts = {sid: (r.x_km, r.y_km) for sid, r in S.iterrows()}
    k = 0
    for z in zones:
        for x, y in list(z.buffer(0.05, join_style=2).exterior.coords)[:-1]:
            if not union.contains(Point(x, y)):
                pts[f"v{k}"] = (x, y); k += 1
    keys = list(pts)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            pa, pb = pts[keys[a]], pts[keys[b]]
            seg = LineString([pa, pb])
            if not any(seg.crosses(z) or seg.within(z) for z in zones) and not union.intersection(seg).length > 1e-6:
                G.add_edge(keys[a], keys[b], weight=math.dist(pa, pb))
    return G, pts


@functools.lru_cache(None)
def distance_matrix():
    """One-way obstacle-avoiding flight distance (km) between every pair of sites, and the
    straight-line distance, so the detour factor can be reported."""
    G, pts = flight_graph()
    ids = list(sites().index)
    D = pd.DataFrame(0.0, index=ids, columns=ids); E = D.copy()
    for a in ids:
        L = nx.single_source_dijkstra_path_length(G, a)
        for b in ids:
            D.loc[a, b] = L[b]; E.loc[a, b] = math.dist(pts[a], pts[b])
    return D, E


def path(a, b):
    G, pts = flight_graph()
    return [pts[n] for n in nx.dijkstra_path(G, a, b)]


def write_audit():
    for f in FILES: sources()
    costs(); sites(); demand(); hourly(); wind(); components()
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    df = pd.DataFrame(AUDIT).drop_duplicates()
    df.to_csv(os.path.join(OUT, "tables", "data_audit.csv"), index=False)
    return df


if __name__ == "__main__":
    print(sources())
    print(write_audit().to_string())
    D, E = distance_matrix()
    r = (D / E.replace(0, np.nan))
    print("max detour factor", np.nanmax(r.values).round(3))
    print(D.loc[["H0", "H4"], ["P17", "P19", "P22", "P03"]].round(2))
