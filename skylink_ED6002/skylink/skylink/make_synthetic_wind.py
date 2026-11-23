"""Synthetic stand-in for wind_days.csv (the real 8,760-row file is in skylink_data.zip,
which was not available). Calibrated to the monthly table on p.7 of the brief:
monthly mean wind, highest gust and % hours with gust >= 14 m/s. Storm hours are clustered
into multi-hour episodes (monsoon squalls), not scattered at random.
THIS IS NOT THE COURSE DATA. Drop the real file into data/real/ and everything re-runs on it."""
import numpy as np, pandas as pd, sys
MON = [(31,3.1,24.5,1.1),(28,3.3,21.8,0.6),(31,3.3,26.0,0.8),(30,3.2,11.3,0.0),(31,3.6,21.0,1.7),
       (30,4.8,24.1,4.3),(31,7.3,40.9,24.6),(31,5.8,31.7,10.3),(30,4.3,24.7,2.4),(31,4.6,24.9,4.6),
       (30,6.0,29.0,11.5),(31,3.9,25.3,2.8)]
def make(seed=2026):
    rng = np.random.default_rng(seed); rows=[]; day0=0
    for mi,(nd,wbar,gmax,pct) in enumerate(MON):
        H = nd*24; hrs = np.arange(H)
        # AR(1) log-wind with diurnal cycle (afternoon peak)
        z = np.zeros(H); e = rng.normal(0,0.18,H)
        for k in range(1,H): z[k]=0.93*z[k-1]+e[k]
        w = np.exp(z)*(1+0.25*np.sin(2*np.pi*((hrs%24)-9)/24))
        gf = 1.45+0.12*rng.standard_normal(H).clip(-2,2)
        nstorm = int(round(pct/100*H)); storm = np.zeros(H,bool)
        while storm.sum() < nstorm:
            s = rng.integers(0,H); L = rng.integers(2,9); storm[s:min(H,s+L)] = True
            if storm.sum() > nstorm:  # trim
                idx = np.where(storm)[0]; storm[idx[nstorm:]] = False
        g = np.empty(H)
        if gmax < 14:   # calm month: no storms, scale so max gust = gmax
            w = w*wbar/w.mean(); g = w*gf; g *= gmax/g.max()
        else:
            ns = ~storm
            w[ns] = w[ns]*wbar/w.mean(); g[ns] = np.minimum(w[ns]*gf[ns], 13.9)
            if storm.any():
                gs = rng.uniform(14.0, 14.0+0.6*(gmax-14), storm.sum()); gs[rng.integers(0,len(gs))] = gmax
                g[storm] = gs; w[storm] = gs/rng.uniform(1.5,1.9,storm.sum())
            # rescale calm hours so the monthly mean matches
            target = wbar*H - w[storm].sum(); w[ns] *= target/w[ns].sum(); g[ns]=np.minimum(np.maximum(g[ns],w[ns]*1.1),13.9)
        w = np.minimum(w, g/1.15)          # physical: gust >= mean
        d = (rng.normal(225 if 5<=mi<=8 else 45, 40, H)) % 360   # SW monsoon Jun-Sep, NE otherwise
        for k in range(H):
            rows.append((day0+k//24+1, mi+1, k%24, round(w[k],2), round(g[k],2), round(d[k],0)))
        day0 += nd
    return pd.DataFrame(rows, columns=['day','month','hour','mean_wind_ms','gust_ms','direction_deg'])
if __name__=='__main__':
    df = make(); out = sys.argv[1] if len(sys.argv)>1 else 'data/reconstructed/wind_days.csv'
    df.to_csv(out, index=False)
    s = df.groupby('month').agg(mean=('mean_wind_ms','mean'), gmax=('gust_ms','max'), grounded=('gust_ms', lambda x:(x>=14).mean()*100))
    print(s.round(2)); print(len(df))
