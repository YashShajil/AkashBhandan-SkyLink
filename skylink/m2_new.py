"""M2 - hub plate: SIMP topology optimisation, FE stresses, and cut-out pattern comparison (one file).

Part 1: SIMP topology optimisation (density filter, OC update) + MBB benchmark + checks + M1 re-run.
Part 2: FE stresses in the chosen plate.   Part 3: hexagon / triangle / diagonal cut-outs vs SIMP.


Two FE models:
  * plane-stress Q4 (top88-style) for the MBB benchmark (verification of the optimiser);
  * Mindlin-Reissner Q4 plate-bending element for the hub plate, because the arm reactions
    (rotor thrust) and the battery weight both act OUT of the plate's plane.  An in-plane
    model would optimise the wrong load path.  The plate element itself is verified against
    Timoshenko's simply-supported square plate.
Units: mm, N, MPa.
"""
from __future__ import annotations
import sys, os
if __package__ in (None, ""):          # allow running this file directly (PyCharm Run button)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "skylink"
import os, json, math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy import ndimage
import pandas as pd
from . import data

# ------------------------------------------------------------------ elements
def ke_plane_stress(nu=0.3):
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8, -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    KE = 1/(1-nu**2)*np.array([
        [k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],[k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],
        [k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],[k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],
        [k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],[k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],
        [k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],[k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])
    return KE


def ke_mindlin(a, E, nu, t, shear_pts=1, stab=0.02):
    """Square element of side a. dofs per node (w, bx, by); nodes ordered (-1,-1),(1,-1),(1,1),(-1,1).
    curvatures k=[dbx/dx, dby/dy, dbx/dy+dby/dx], shear g=[dw/dx+bx, dw/dy+by]."""
    Db = E*t**3/(12*(1-nu**2))*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    Ds = 5/6*E/(2*(1+nu))*t*np.eye(2)
    xi_n = np.array([-1, 1, 1, -1]); eta_n = np.array([-1, -1, 1, 1])
    J = a/2
    def shape(xi, eta):
        N = 0.25*(1+xi*xi_n)*(1+eta*eta_n)
        dN = np.vstack([0.25*xi_n*(1+eta*eta_n), 0.25*eta_n*(1+xi*xi_n)])/J
        return N, dN
    K = np.zeros((12, 12))
    gp = [-1/math.sqrt(3), 1/math.sqrt(3)]
    for xi in gp:
        for eta in gp:
            N, dN = shape(xi, eta); Bb = np.zeros((3, 12))
            for i in range(4):
                Bb[0, 3*i+1] = dN[0, i]; Bb[1, 3*i+2] = dN[1, i]; Bb[2, 3*i+1] = dN[1, i]; Bb[2, 3*i+2] = dN[0, i]
            K += Bb.T@Db@Bb*J*J
    def Ks(pts, w):
        k = np.zeros((12, 12))
        for xi in pts:
            for eta in pts:
                N, dN = shape(xi, eta); Bs = np.zeros((2, 12))
                for i in range(4):
                    Bs[0, 3*i] = dN[0, i]; Bs[0, 3*i+1] = N[i]; Bs[1, 3*i] = dN[1, i]; Bs[1, 3*i+2] = N[i]
                k += Bs.T@Ds@Bs*J*J*w
        return k
    # selective reduced integration of shear (1 point) removes shear locking; a small fraction
    # `stab` of the fully integrated shear removes the two hourglass modes it introduces.
    K += Ks(gp, 1.0) if shear_pts == 2 else (1-stab)*Ks([0.0], 4.0)+stab*Ks(gp, 1.0)
    return K


# ------------------------------------------------------------------ generic SIMP
def density_filter(nelx, nely, rmin):
    if rmin < 1.0:
        return sp.identity(nelx*nely, format="csr")
    r = int(math.ceil(rmin)-1); rows, cols, vals = [], [], []
    for i in range(nelx):
        for j in range(nely):
            e = i*nely+j
            for k in range(max(i-r, 0), min(i+r+1, nelx)):
                for l in range(max(j-r, 0), min(j+r+1, nely)):
                    f = k*nely+l; w = max(0, rmin-math.hypot(i-k, j-l))
                    if w > 0: rows.append(e); cols.append(f); vals.append(w)
    H = sp.csr_matrix((vals, (rows, cols)), shape=(nelx*nely,)*2)
    return sp.diags(1/np.asarray(H.sum(1)).ravel())@H


def simp(nelx, nely, volfrac, penal, rmin, KE, edof, F, free, passive_solid=None, passive_void=None,
         maxit=200, tol=0.01, Emin=1e-9, ft="density", start=None):
    ne = nelx*nely; ndof = F.shape[0]; nd = KE.shape[0]
    iK = np.kron(edof, np.ones((nd, 1))).ravel(); jK = np.kron(edof, np.ones((1, nd))).ravel()
    H = density_filter(nelx, nely, rmin)
    x = np.full(ne, volfrac) if start is None else start.copy()
    ps = np.zeros(ne, bool) if passive_solid is None else passive_solid
    pv = np.zeros(ne, bool) if passive_void is None else passive_void
    x[ps] = 1; x[pv] = 0
    xP = H@x if ft == "density" else x.copy(); xP[ps] = 1; xP[pv] = 0
    hist = []; change = 1; it = 0
    while change > tol and it < maxit:
        it += 1
        sK = (KE.ravel()[None, :]*(Emin+xP[:, None]**penal*(1-Emin))).ravel()
        K = sp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        U = np.zeros(ndof); U[free] = spsolve(K[free][:, free], F[free])
        ce = np.einsum("ij,jk,ik->i", U[edof], KE, U[edof])
        c = float(((Emin+xP**penal*(1-Emin))*ce).sum())
        dc = -penal*xP**(penal-1)*(1-Emin)*ce; dv = np.ones(ne)
        if ft == "density":
            dc = H.T@dc; dv = H.T@dv
        elif ft == "sensitivity":
            dc = (H@(x*dc))/np.maximum(1e-3, x)
        l1, l2, move = 0.0, 1e9, 0.2
        target = volfrac*ne
        while (l2-l1)/(l1+l2+1e-30) > 1e-4:
            lm = 0.5*(l1+l2)
            xn = np.clip(x*np.sqrt(np.maximum(-dc, 0)/(dv*lm)), np.maximum(0, x-move), np.minimum(1, x+move))
            xn[ps] = 1; xn[pv] = 0
            xPn = H@xn if ft == "density" else xn.copy(); xPn[ps] = 1; xPn[pv] = 0
            if xPn.sum() > target: l1 = lm
            else: l2 = lm
        change = np.abs(xn-x).max(); x = xn; xP = xPn
        hist.append(c)
    return xP, c, hist, U


def compliance(nelx, nely, xP, penal, KE, edof, F, free, Emin=1e-9):
    nd = KE.shape[0]; ndof = F.shape[0]
    iK = np.kron(edof, np.ones((nd, 1))).ravel(); jK = np.kron(edof, np.ones((1, nd))).ravel()
    sK = (KE.ravel()[None, :]*(Emin+xP[:, None]**penal*(1-Emin))).ravel()
    K = sp.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    U = np.zeros(ndof); U[free] = spsolve(K[free][:, free], F[free]); return float(F@U), U


# ------------------------------------------------------------------ MBB benchmark
def mbb_setup(nelx=60, nely=20):
    edof = []
    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely+1)*elx+ely; n2 = (nely+1)*(elx+1)+ely
            edof.append([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1])
    edof = np.array(edof); ndof = 2*(nelx+1)*(nely+1)
    F = np.zeros(ndof); F[1] = -1
    fixed = np.union1d(np.arange(0, 2*(nely+1), 2), [ndof-1])
    free = np.setdiff1d(np.arange(ndof), fixed)
    return edof, F, free


def run_mbb(figd):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    nelx, nely = 60, 20; KE = ke_plane_stress(); edof, F, free = mbb_setup(nelx, nely)
    cases = [("density filter, p=3, rmin=1.5", dict(penal=3, rmin=1.5, ft="density")),
             ("sensitivity filter, p=3, rmin=1.5", dict(penal=3, rmin=1.5, ft="sensitivity")),
             ("NO filter, p=3", dict(penal=3, rmin=0.5, ft="none")),
             ("density filter, p=1", dict(penal=1, rmin=1.5, ft="density"))]
    out = []
    fig, ax = plt.subplots(len(cases), 1, figsize=(7, 9))
    for k, (lab, kw) in enumerate(cases):
        xP, c, hist, _ = simp(nelx, nely, 0.5, KE=KE, edof=edof, F=F, free=free, maxit=300, **kw)
        gray = float(((xP > 0.1) & (xP < 0.9)).mean())
        # checkerboard measure: fraction of 2x2 blocks with alternating pattern
        X = xP.reshape(nelx, nely).T > 0.5
        cb = float(((X[:-1, :-1] == X[1:, 1:]) & (X[1:, :-1] == X[:-1, 1:]) & (X[:-1, :-1] != X[1:, :-1])).mean())
        out.append(dict(case=lab, compliance=c, iterations=len(hist), gray_fraction=gray, checkerboard_fraction=cb))
        ax[k].imshow(-xP.reshape(nelx, nely).T, cmap="gray", interpolation="none"); ax[k].axis("off")
        ax[k].set_title(f"{lab}: c={c:.1f}, gray={gray:.0%}, checkerboard={cb:.1%}", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(figd, "m2_mbb_benchmark.png"), dpi=130); plt.close()
    # FE verification: solid MBB half-beam = cantilever-like simply supported beam; compare with
    # Timoshenko beam for full simply supported span 2L, central load 2P (symmetry), E=1, nu=.3, unit thickness
    c_solid, U = compliance(nelx, nely, np.ones(nelx*nely), 3, KE, edof, F, free)
    L, h, P, E, nu = 2*nelx, nely, 2.0, 1.0, 0.3
    I = h**3/12; G = E/(2*(1+nu))
    w_beam = P*L**3/(48*E/(1-nu**2)*I) + P*L/(4*5/6*G*h)   # plane stress plate strip -> E/(1-nu^2) bounds plane strain-like width effect
    w_beam_ps = P*L**3/(48*E*I) + P*L/(4*5/6*G*h)
    return out, dict(fe_load_point_deflection=abs(U[1]), timoshenko_beam_E=w_beam_ps, timoshenko_beam_Eprime=w_beam)


# ------------------------------------------------------------------ hub plate
E_AL, NU_AL, T_PL, RHO_AL = 68900.0, 0.33, 3.0, 2700.0


# Arm attachment: 2 bolts per arm (M5, 5.5 mm clearance hole) on the arm axis (plate diagonal),
# each with a solid boss (washer seat). Supports are applied on the bosses (the clamp seats there).
BOLTS = True
BOLT_S = (14.0, 34.0)      # bolt positions along the diagonal, measured from the plate corner (mm)
HOLE_R = 2.75              # M5 clearance hole radius (mm)
BOSS_R = 8.75              # solid boss radius (mm): 6 mm of metal around the hole = minimum member width


def bolt_centres(side=160.0):
    pts = []
    for cx, sx in ((0, 1), (side, -1)):
        for cy, sy in ((0, 1), (side, -1)):
            for s_ in BOLT_S:
                pts.append((cx + sx * s_ / math.sqrt(2), cy + sy * s_ / math.sqrt(2)))
    return pts


def hole_mask(n, side=160.0):
    """elements inside a bolt hole (forced void); empty if BOLTS is False"""
    a = side / n; xc = (np.arange(n) + 0.5) * a
    EX, EY = np.meshgrid(xc, xc, indexing="ij"); EX = EX.ravel(); EY = EY.ravel()
    m = np.zeros(n * n, bool)
    if BOLTS:
        for bx, by in bolt_centres(side):
            m |= np.hypot(EX - bx, EY - by) <= HOLE_R
    return m


def plate_setup(n=80, side=160.0, support_mm=10.0, load_mm=40.0, F_total=1.0):
    """Returns KE, edof, F, free, solid_support (passive-solid elements at the arm attachments:
    bolt bosses if BOLTS else 10 mm corner patches), centre (battery pad), a (element size).
    Holes: use hole_mask(n) as passive void."""
    a = side/n; nn = n+1
    node = lambda i, j: i*nn+j   # i along x, j along y
    edof = []
    for ex in range(n):
        for ey in range(n):
            ns = [node(ex, ey), node(ex+1, ey), node(ex+1, ey+1), node(ex, ey+1)]
            edof.append(sum([[3*q, 3*q+1, 3*q+2] for q in ns], []))
    edof = np.array(edof); ndof = 3*nn*nn
    xc = (np.arange(n)+0.5)*a
    EX, EY = np.meshgrid(xc, xc, indexing="ij"); EX = EX.ravel(); EY = EY.ravel()   # element e = ex*n+ey
    xs = np.arange(nn)*a; NX, NY = np.meshgrid(xs, xs, indexing="ij"); NX = NX.ravel(); NY = NY.ravel()
    corner = np.zeros(n*n, bool); sup = np.zeros(nn*nn, bool)
    if BOLTS:
        for bx, by in bolt_centres(side):
            r_e = np.hypot(EX-bx, EY-by); r_n = np.hypot(NX-bx, NY-by)
            corner |= (r_e > HOLE_R) & (r_e <= BOSS_R)
            sup |= (r_n >= HOLE_R - 1e-9) & (r_n <= BOSS_R)          # clamp seat on the boss: w = 0
        # clamp seat: solid strip joining each arm's two bosses (the arm clamp needs both bolts tied
        # together to pass the arm's bending moment into the plate)
        for cx, sx in ((0, 1), (side, -1)):
            for cy, sy in ((0, 1), (side, -1)):
                ux, uy = sx / math.sqrt(2), sy / math.sqrt(2)
                along = (EX-cx)*ux + (EY-cy)*uy; across = np.abs(-(EX-cx)*uy + (EY-cy)*ux)
                corner |= (along >= BOLT_S[0]) & (along <= BOLT_S[-1]) & (across <= BOSS_R)
        corner &= ~hole_mask(n, side)
    else:
        for cx in (0, side):
            for cy in (0, side):
                corner |= (np.abs(EX-cx) < support_mm) & (np.abs(EY-cy) < support_mm)
                sup |= (np.abs(NX-cx) <= support_mm) & (np.abs(NY-cy) <= support_mm)
    centre = (np.abs(EX-side/2) < load_mm/2) & (np.abs(EY-side/2) < load_mm/2)
    # load: uniform pressure over central patch, lumped to nodes
    F = np.zeros(ndof); q = F_total/(load_mm**2)
    for e in np.where(centre)[0]:
        for k in range(4): F[edof[e, 3*k]] += -q*a*a/4
    fixed = 3*np.where(sup)[0]
    free = np.setdiff1d(np.arange(ndof), fixed)
    KE = ke_mindlin(a, E_AL, NU_AL, T_PL)
    return KE, edof, F, free, corner, centre, a


def verify_plate_element():
    """Simply-supported square plate, uniform load: w_max = 0.00406 q a^4 / D (Timoshenko & Woinowsky-Krieger)."""
    res = []
    for n in (10, 20, 40):
        side = 160.0; a = side/n; nn = n+1
        KE = ke_mindlin(a, E_AL, NU_AL, T_PL)
        node = lambda i, j: i*nn+j
        edof = np.array([sum([[3*q, 3*q+1, 3*q+2] for q in [node(ex, ey), node(ex+1, ey), node(ex+1, ey+1), node(ex, ey+1)]], [])
                         for ex in range(n) for ey in range(n)])
        ndof = 3*nn*nn; F = np.zeros(ndof); q = 1e-3
        for e in range(n*n):
            for k in range(4): F[edof[e, 3*k]] += q*a*a/4
        edge = [node(i, j) for i in range(nn) for j in range(nn) if i in (0, n) or j in (0, n)]
        free = np.setdiff1d(np.arange(ndof), 3*np.array(edge))
        _, U = compliance(n, n, np.ones(n*n), 1, KE, edof, F, free)
        D = E_AL*T_PL**3/(12*(1-NU_AL**2)); w_th = 0.00406*q*side**4/D
        res.append(dict(mesh=f"{n}x{n}", w_fe=U[3*node(n//2, n//2)], w_theory=w_th, rel_err=U[3*node(n//2, n//2)]/w_th-1))
    return res


def sensitivity_check(seed=0):
    """Verification of the SIMP gradient: analytic dC/dx (through the density filter) vs central
    finite differences, on a small MBB mesh with random densities."""
    rng = np.random.default_rng(seed)
    nelx, nely, p, rmin = 12, 4, 3.0, 1.5
    KE = ke_plane_stress(); edof, F, free = mbb_setup(nelx, nely)
    H = density_filter(nelx, nely, rmin); Emin = 1e-9
    x = rng.uniform(0.2, 0.9, nelx*nely)
    def C(xx):
        return compliance(nelx, nely, H @ xx, p, KE, edof, F, free)[0]
    xP = H @ x; c, U = compliance(nelx, nely, xP, p, KE, edof, F, free)
    ce = np.einsum("ij,jk,ik->i", U[edof], KE, U[edof])
    dc = H.T @ (-p*xP**(p-1)*(1-Emin)*ce)
    h = 1e-6; fd = np.array([(C(x + h*np.eye(len(x))[e]) - C(x - h*np.eye(len(x))[e])) / (2*h) for e in range(len(x))])
    return dict(n_elements=len(x), max_abs_err=float(np.abs(fd-dc).max()), max_rel_err=float(np.max(np.abs(fd-dc)/np.abs(fd))))


def plate_load_N():
    """Plate load = 2 x battery weight of the current M1 design (2g max-thrust case).
    Uses R_REQ from m1_vehicle.py, so the load follows whatever range you set."""
    from . import m1_vehicle as m1
    fm = None
    try:
        from . import m3_surrogate as m3
        k, _, _ = m3.fit_all(make_figs=False); fm = m3.fm_function(k)
    except Exception:
        pass
    x, o, _ = m1.solve_mdf(m1.Vehicle(m_plate=0.25, fm=fm), m1.R_REQ)
    return 2 * 9.81 * x[1] / m1.EB          # x[1] = battery energy E (Wh), EB = 150 Wh/kg


def plate_mass(vf):
    return vf*0.16*0.16*0.003*RHO_AL + 0.04


def run(F_total=None, n=64, vfs=(0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import pandas as pd
    figd = os.path.join(data.OUT, "figs"); tabd = os.path.join(data.OUT, "tables")
    mbb, beam = run_mbb(figd)
    pv = verify_plate_element()
    if F_total is None:
        F_total = plate_load_N()   # 2 x battery weight of the current M1 design
    KE, edof, F, free, corner, centre, a = plate_setup(n=n, F_total=F_total)
    hole = hole_mask(n)
    xsolid = np.ones(n*n); xsolid[hole] = 1e-9                          # reference: solid plate WITH bolt holes
    c_solid, _ = compliance(n, n, xsolid, 3, KE, edof, F, free)
    rmin = 6.0/a   # 6 mm filter radius in elements -> member width >= ~6 mm
    rows, designs = [], {}
    prev = None
    for vf in vfs:
        xP, c, hist, _ = simp(n, n, vf, 3, rmin, KE, edof, F, free, passive_solid=corner | centre, passive_void=hole, maxit=150, tol=0.005)
        # crisp (waterjet) version: threshold at 0.5, re-evaluate
        xb = (xP > 0.5).astype(float); xb[corner | centre] = 1; xb[hole] = 0
        cb, _ = compliance(n, n, np.maximum(xb, 1e-9), 3, KE, edof, F, free)
        X = xb.reshape(n, n).T.astype(bool)
        r = int(round(3.0/a)); yy, xx = np.mgrid[-r:r+1, -r:r+1]; disk = xx**2+yy**2 <= r*r
        opened = ndimage.binary_opening(X, structure=disk)
        thin = float((X & ~opened).sum()/max(X.sum(), 1))
        rows.append(dict(volfrac=vf, plate_mass_kg=plate_mass(vf), compliance_gray=c, compliance_crisp=cb,
                         c_over_solid=cb/c_solid, crisp_vf=float(xb.mean()), frac_material_thinner_than_6mm=thin, iters=len(hist)))
        designs[vf] = xP
    df = pd.DataFrame(rows)
    feas = df[df.c_over_solid <= 2.5]
    pick = float(feas.volfrac.min()) if len(feas) else 0.5
    # figures
    fig, axs = plt.subplots(2, len(vfs)//2+1, figsize=(16, 6.5)); axs = axs.ravel()
    for k, vf in enumerate(vfs):
        axs[k].imshow(-designs[vf].reshape(n, n).T, cmap="gray", origin="lower", extent=[0, 160, 0, 160])
        axs[k].set_title(f"vf={vf}  C/C_solid={df.c_over_solid[k]:.2f}", fontsize=9); axs[k].axis("off")
    ax = axs[len(vfs)]
    ax.semilogy(df.plate_mass_kg*1000, df.compliance_crisp/c_solid, "o-", label="crisp (thresholded); vf=0.2 disconnects")
    ax.plot(df.plate_mass_kg*1000, df.compliance_gray/c_solid, "s--", label="SIMP (gray)")
    ax.axhline(2.5, color="r", ls=":", label="limit 2.5 x solid"); ax.axvline(plate_mass(pick)*1000, color="g", ls=":")
    ax.set_xlabel("plate mass (g)"); ax.set_ylabel("compliance / solid"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    for k in range(len(vfs)+1, len(axs)): axs[k].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(figd, "m2_plate_sweep.png"), dpi=120); plt.close()
    # outline for waterjet: contour of picked design, export SVG + DXF
    xo = designs[pick].copy(); xo[hole] = 1.0          # trace the outline with holes filled in ...
    xP = xo.reshape(n, n).T
    fig, ax = plt.subplots(figsize=(5, 5))
    cs = ax.contour(np.linspace(a/2, 160-a/2, n), np.linspace(a/2, 160-a/2, n), xP, levels=[0.5], colors="k")
    ax.set_aspect("equal"); ax.set_xlim(0, 160); ax.set_ylim(0, 160)
    ax.add_patch(plt.Rectangle((0, 0), 160, 160, fill=False, ls="--", color="grey"))
    if BOLTS:                                          # ... then draw the holes as true circles (as in the DXF)
        for bx, by in bolt_centres():
            ax.add_patch(plt.Circle((bx, by), HOLE_R, fill=False, color="k", lw=1.2))
    ax.set_title(f"Waterjet outline, vf={pick}, mass={plate_mass(pick)*1000:.0f} g\n8 x M5 bolt holes (5.5 mm), 2 per arm")
    plt.savefig(os.path.join(figd, "m2_plate_outline.png"), dpi=130); plt.close()
    segs = [p for p in cs.allsegs[0] if len(p) > 4]
    with open(os.path.join(tabd, "m2_plate_outline.dxf"), "w", encoding="utf-8") as f:
        f.write("0\nSECTION\n2\nENTITIES\n")
        for p in segs:
            f.write("0\nLWPOLYLINE\n8\nCUT\n90\n%d\n70\n1\n" % len(p))
            for x, y in p: f.write(f"10\n{x:.3f}\n20\n{y:.3f}\n")
        if BOLTS:
            for bx, by in bolt_centres():
                f.write(f"0\nCIRCLE\n8\nHOLES\n10\n{bx:.3f}\n20\n{by:.3f}\n40\n{HOLE_R:.3f}\n")
        f.write("0\nENDSEC\n0\nEOF\n")
    df.to_csv(os.path.join(tabd, "m2_plate_sweep.csv"), index=False)
    sc = sensitivity_check()
    summ = dict(mbb=mbb, mbb_fe_check=beam, plate_element_check=pv, simp_sensitivity_check=sc, bolts=dict(on=BOLTS, positions_mm=BOLT_S, hole_r=HOLE_R, boss_r=BOSS_R),
                c_solid=c_solid, F_total_N=F_total,
                picked_volfrac=pick, picked_plate_mass_kg=plate_mass(pick), rmin_mm=6.0)
    json.dump(summ, open(os.path.join(tabd, "m2_summary.json"), "w", encoding="utf-8"), indent=1, default=float)
    return summ, df


def rerun_m1(m_plate):
    """Brief: 'Put the new plate mass back into M1 and re-run once'. Uses R_REQ from m1_vehicle.py."""
    from . import m1_vehicle as m1
    fm = None
    try:
        from . import m3_surrogate as m3
        k, _, _ = m3.fit_all(make_figs=False); fm = m3.fm_function(k)
    except Exception:
        pass
    r0, d0, _ = m1.run(m_plate=0.25, fm=fm, tag="m1_base", make_figs=False, quick=True)
    r1, d1, _ = m1.run(m_plate=m_plate, fm=fm, tag="m1_plate", make_figs=False, quick=True)
    a, b = r0["mdf"], r1["mdf"]
    fm_str = "M3 surrogate" if fm is not None else "FM = 0.65"
    print(f"\n=== M1 re-run with the M2 plate (R >= {m1.R_REQ:g} km, {fm_str}) ===")
    print(f"{'':22s}{'plate 0.250 kg':>16s}{f'plate {m_plate:.3f} kg':>18s}{'change':>12s}")
    for lab, k, f in [("MTOW (kg)", "MTOW_kg", "{:.4f}"), ("battery E (Wh)", "E_Wh", "{:.1f}"), ("rotor D (m)", "D_m", "{:.3f}"),
                      ("tube t (mm)", "t_mm", "{:.3f}"), ("cruise V (m/s)", "V_ms", "{:.2f}"), ("radius R (km)", "R_km", "{:.2f}")]:
        print(f"{lab:22s}{f.format(a[k]):>16s}{f.format(b[k]):>18s}{f.format(b[k]-a[k]):>12s}")
    print(f"Snapped design: {d0.snap_MTOW_kg.iloc[0]:.2f} kg -> {d1.snap_MTOW_kg.iloc[0]:.2f} kg "
          f"({d1.snap_battery.iloc[0]}, {d1.snap_tube.iloc[0]}, {d1.snap_motor.iloc[0]})")
    print("Saved: outputs/tables/m1_plate_summary.json, m1_plate_designs.csv")
    return r0, r1


# =====================================================================================
# PART 2 - FINITE-ELEMENT STRESSES IN THE CHOSEN PLATE
# =====================================================================================
# M2 add-on - finite-element stresses in the hub plate (chosen SIMP design vs solid plate).
#
# Mindlin plate element (same model as M2). Per element, at the centre (xi = eta = 0):
#   curvatures   k = [d(bx)/dx, d(by)/dy, d(bx)/dy + d(by)/dx]          (1/mm)
#   moments      M = D_b k,  D_b = E t^3 / (12 (1 - nu^2)) [[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]   (N mm/mm)
#   surface stress  sigma = 6 M / t^2   (MPa)  -> von Mises from (sx, sy, txy)
#   transverse shear  q = kappa_s G t (gamma),  tau_max = 1.5 q / t   (MPa, mid-surface; small)
# Load case: battery weight x 2 (max-thrust 2g case), as in M2. Supports: w = 0 on the 8 bolt bosses. Only solid elements of the cut plate
# are reported (voids carry no load). Allowable: 6061-T6 yield 276 MPa.
YIELD = 276.0      # MPa, 6061-T6


def element_stresses(U, edof, a, n):
    """von Mises at top surface (bending) and max transverse shear, per element"""
    J = a / 2
    xi_n = np.array([-1, 1, 1, -1]); eta_n = np.array([-1, -1, 1, 1])
    dN = np.vstack([0.25 * xi_n, 0.25 * eta_n]) / J        # shape-function derivatives at the centre
    N0 = np.full(4, 0.25)
    Ue = U[edof]                                            # (ne, 12): per node (w, bx, by)
    w, bx, by = Ue[:, 0::3], Ue[:, 1::3], Ue[:, 2::3]
    kx = bx @ dN[0]; ky = by @ dN[1]; kxy = bx @ dN[1] + by @ dN[0]
    D = E_AL * T_PL ** 3 / (12 * (1 - NU_AL ** 2))
    Mx = D * (kx + NU_AL * ky); My = D * (ky + NU_AL * kx); Mxy = D * (1 - NU_AL) / 2 * kxy
    sx, sy, txy = 6 * Mx / T_PL ** 2, 6 * My / T_PL ** 2, 6 * Mxy / T_PL ** 2
    vm = np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * txy ** 2)
    G = E_AL / (2 * (1 + NU_AL))
    gx = w @ dN[0] + bx @ N0; gy = w @ dN[1] + by @ N0
    tau = 1.5 * (5 / 6) * G * np.hypot(gx, gy)              # 1.5 q/t with q = kappa G t gamma
    s1 = (sx + sy) / 2 + np.sqrt(((sx - sy) / 2) ** 2 + txy ** 2)
    s2 = (sx + sy) / 2 - np.sqrt(((sx - sy) / 2) ** 2 + txy ** 2)
    return dict(vm=vm, sx=sx, sy=sy, txy=txy, s1=s1, s2=s2, tau=tau)


def run_stress(n=64, F_total=None):
    F_total = plate_load_N() if F_total is None else F_total
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    figd = os.path.join(data.OUT, "figs"); tabd = os.path.join(data.OUT, "tables")
    s2 = json.load(open(os.path.join(tabd, "m2_summary.json"), encoding="utf-8"))
    vf = s2["picked_volfrac"]
    KE, edof, F, free, corner, centre, a = plate_setup(n=n, F_total=F_total)
    # chosen design: SIMP at the picked volume fraction, thresholded (as cut)
    hole = hole_mask(n)
    xP, c, h, _ = simp(n, n, vf, 3, 6.0 / a, KE, edof, F, free, passive_solid=corner | centre, passive_void=hole, maxit=150, tol=0.005)
    xb = (xP > 0.5).astype(float); xb[corner | centre] = 1; xb[hole] = 0
    xs = np.ones(n * n); xs[hole] = 0
    cases = {}
    for name, x in [("solid plate (with bolt holes)", xs), (f"SIMP vf={vf} (as cut)", xb)]:
        C, U = compliance(n, n, np.maximum(x, 1e-9), 3, KE, edof, F, free)
        st = element_stresses(U, edof, a, n)
        solid = x > 0.5
        st_s = {k: np.where(solid, v, np.nan) for k, v in st.items()}
        wmax = float(np.abs(U[0::3]).max())
        cases[name] = dict(x=x, U=U, st=st_s, C=C, wmax=wmax, solid=solid)
    # summary table (exclude the support patches, where the w=0 patch edge gives a local singularity)
    xc = (np.arange(n) + 0.5) * a; EX, EY = np.meshgrid(xc, xc, indexing="ij"); EX = EX.ravel(); EY = EY.ravel()
    near_sup = np.zeros(n * n, bool)
    for bx, by in bolt_centres():
        near_sup |= np.hypot(EX - bx, EY - by) <= BOSS_R + 3     # bosses + one element ring: clamp-edge singularity
    rows = []
    for name, cc in cases.items():
        vm = cc["st"]["vm"]; i = int(np.nanargmax(vm)); j = int(np.nanargmax(np.where(near_sup, np.nan, vm)))
        rows.append(dict(case=name, mass_g=plate_mass(float(cc["x"].mean())) * 1000, max_deflection_mm=cc["wmax"],
                         vm_max_MPa=float(np.nanmax(vm)), at_mm=f"({EX[i]:.0f}, {EY[i]:.0f})",
                         vm_max_away_from_clamps_MPa=float(vm[j]), at2_mm=f"({EX[j]:.0f}, {EY[j]:.0f})",
                         vm_mean_MPa=float(np.nanmean(vm)), tau_max_MPa=float(np.nanmax(cc["st"]["tau"])),
                         safety_factor_yield=YIELD / float(np.nanmax(vm))))
    tab = pd.DataFrame(rows); tab.to_csv(os.path.join(tabd, "m2_stress.csv"), index=False)
    # figure: von Mises on both plates (same colour scale) + principal stress + deflection
    vmax = max(np.nanpercentile(cc["st"]["vm"], 99.5) for cc in cases.values())
    fig, ax = plt.subplots(2, 3, figsize=(18, 11))
    for r, (name, cc) in enumerate(cases.items()):
        V = cc["st"]["vm"].reshape(n, n).T
        im = ax[r, 0].imshow(V, origin="lower", extent=[0, 160, 0, 160], cmap="jet", norm=Normalize(0, vmax))
        ax[r, 0].contour(np.linspace(a / 2, 160 - a / 2, n), np.linspace(a / 2, 160 - a / 2, n), cc["x"].reshape(n, n).T, [0.5], colors="k", linewidths=.6)
        plt.colorbar(im, ax=ax[r, 0], label="von Mises, top surface (MPa)")
        tr = tab.iloc[r]
        ax[r, 0].set_title(f"{name}: von Mises (top surface)\nmax {tr.vm_max_MPa:.1f} MPa at {tr.at_mm} mm, safety factor on yield {tr.safety_factor_yield:.1f}", fontsize=9.5)
        S1 = cc["st"]["s1"].reshape(n, n).T; S2 = cc["st"]["s2"].reshape(n, n).T
        P = np.where(np.abs(S1) >= np.abs(S2), S1, S2)
        lim = np.nanpercentile(np.abs(P), 99.5)
        im2 = ax[r, 1].imshow(P, origin="lower", extent=[0, 160, 0, 160], cmap="RdBu_r", norm=Normalize(-lim, lim))
        plt.colorbar(im2, ax=ax[r, 1], label="dominant principal stress (MPa)\n+ tension / - compression (top face)")
        ax[r, 1].set_title("dominant principal stress, top surface", fontsize=10)
        Wd = np.abs(cc["U"][0::3]).reshape(n + 1, n + 1).T
        Wd = np.where(np.kron(cc["solid"].reshape(n, n).T, np.ones((1, 1))).astype(bool), Wd[:-1, :-1], np.nan)
        im3 = ax[r, 2].imshow(Wd, origin="lower", extent=[0, 160, 0, 160], cmap="viridis")
        plt.colorbar(im3, ax=ax[r, 2], label="deflection |w| (mm)")
        ax[r, 2].set_title(f"deflection, max {cc['wmax']:.3f} mm", fontsize=10)
        for k in range(3):
            ax[r, k].set_xlabel("x (mm)"); ax[r, k].set_ylabel("y (mm)")
            for bx, by in bolt_centres():
                ax[r, k].add_patch(plt.Circle((bx, by), BOSS_R, fill=False, ec="w", lw=.8))
            ax[r, k].add_patch(plt.Rectangle((60, 60), 40, 40, fill=False, ec="w", ls="--", lw=1))
    fig.suptitle(f"Hub plate FE stresses - load {F_total:.0f} N on the 40x40 mm battery pad (2 x battery weight), supported on the 8 bolt bosses (w = 0); "
                 f"6061-T6, t = 3 mm, yield {YIELD:.0f} MPa", fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(figd, "m2_stress.png"), dpi=115); plt.close()
    return tab


# =====================================================================================
# PART 3 - REGULAR CUT-OUT PATTERNS vs SIMP
# =====================================================================================
# M2 add-on - regular cut-out patterns vs the SIMP optimum.
#
# Compares three manufacturable lightening patterns on the same plate, supports, load and
# Mindlin plate FE model as M2:
#   * HEXAGON cut-outs  (honeycomb ribs)
#   * TRIANGLE cut-outs (isogrid ribs: 0/60/120 deg)
#   * DIAGONAL grid (ribs at +-45 deg, along the corner-to-centre load paths)
# All use 6 mm ribs (= minimum member width), a 6 mm solid rim, 8 M5 bolt holes with solid bosses (supports on the bosses) and a
# solid 40x40 mm battery pad. The volume fraction is changed by changing the cell size.
# Efficiency = compliance at equal plate mass (lower is stiffer), normalised by the solid plate.
NC = 128                # elements per side for the cut-out study (1.25 mm) - resolves 6 mm ribs
SIDE = 160.0
RIB = 6.0               # rib width = minimum member width (mm)
RIM = 6.0               # solid border (mm)
SUB = 4                 # sub-samples per element edge for area fraction


def _grid():
    a = SIDE / NC
    s = (np.arange(NC * SUB) + 0.5) * a / SUB
    X, Y = np.meshgrid(s, s, indexing="ij")
    return X, Y


def _to_elements(mask):
    """area fraction per element from the sub-sampled mask, then crisp (>= 0.5 solid) like a waterjet part"""
    f = mask.reshape(NC, SUB, NC, SUB).mean(axis=(1, 3))
    return (f >= 0.5).astype(float).ravel()     # element e = ex*NC + ey (x-major), as in plate_setup


def _keep_solid(X, Y):
    c = SIDE / 2
    rim = (X < RIM) | (X > SIDE - RIM) | (Y < RIM) | (Y > SIDE - RIM)
    pad = (np.abs(X - c) <= 20) & (np.abs(Y - c) <= 20)
    bosses = np.zeros_like(X, bool)
    for bx, by in bolt_centres(SIDE):
        bosses |= np.hypot(X - bx, Y - by) <= BOSS_R
    # clamp seat: solid strip joining each arm's two bosses (same as the SIMP model), so a pattern can
    # never leave a bolt boss floating inside a cell
    clamp = np.zeros_like(X, bool)
    for cx, sx in ((0, 1), (SIDE, -1)):
        for cy, sy in ((0, 1), (SIDE, -1)):
            ux, uy = sx / math.sqrt(2), sy / math.sqrt(2)
            along = (X - cx) * ux + (Y - cy) * uy; across = np.abs(-(X - cx) * uy + (Y - cy) * ux)
            clamp |= (along >= BOLT_S[0]) & (along <= BOLT_S[-1]) & (across <= BOSS_R)
    return rim | pad | bosses | clamp


def _holes(X, Y):
    h = np.zeros_like(X, bool)
    for bx, by in bolt_centres(SIDE):
        h |= np.hypot(X - bx, Y - by) <= HOLE_R
    return h


def triangle_pattern(pitch):
    """isogrid: ribs along three line families at 0, 60, 120 deg, spacing `pitch` (rib centre-lines)"""
    X, Y = _grid(); c = SIDE / 2
    x, y = X - c, Y - c
    solid = np.zeros_like(X, bool)
    for ang in (0.0, 60.0, 120.0):
        t = math.radians(ang)
        u = -x * math.sin(t) + y * math.cos(t)           # distance along the normal of this family
        d = np.abs(((u + pitch / 2) % pitch) - pitch / 2)  # distance to nearest line of the family
        solid |= d <= RIB / 2
    return _to_elements((solid | _keep_solid(X, Y)) & ~_holes(X, Y))


def diagonal_pattern(pitch):
    """45-degree grid: ribs along both plate diagonals (the load paths) and parallel to them, spacing
    `pitch`. Square cells rotated 45 deg - same 90-degree symmetry as the plate, supports and load."""
    X, Y = _grid(); c = SIDE / 2
    x, y = X - c, Y - c
    solid = np.zeros_like(X, bool)
    for u in ((x + y) / math.sqrt(2), (x - y) / math.sqrt(2)):
        d = np.abs(((u + pitch / 2) % pitch) - pitch / 2)
        solid |= d <= RIB / 2
    return _to_elements((solid | _keep_solid(X, Y)) & ~_holes(X, Y))


def hexagon_pattern(cell):
    """honeycomb: flat-top hexagonal cells, centre spacing `cell` (across-flats of cell incl. rib)"""
    X, Y = _grid(); c = SIDE / 2
    x, y = X - c, Y - c
    a = cell / 2                                         # inradius of a cell (centre to rib centre-line)
    # hex lattice (pointy spacing for flat-top cells): basis b1=(sqrt3*a, a), b2=(0, 2a)
    b1 = np.array([math.sqrt(3) * a, a]); b2 = np.array([0.0, 2 * a])
    B = np.c_[b1, b2]; Bi = np.linalg.inv(B)
    q = np.einsum("ij,jkl->ikl", Bi, np.stack([x, y]))
    best = np.full(x.shape, np.inf)
    for di in (0, 1):
        for dj in (0, 1):
            i = np.floor(q[0]) + di; j = np.floor(q[1]) + dj
            cxl = i * b1[0] + j * b2[0]; cyl = i * b1[1] + j * b2[1]
            dx, dy = np.abs(x - cxl), np.abs(y - cyl)
            # hex "radius" for a cell whose flats face +-y and +-60 deg (pointy left/right)
            h = np.maximum(dy, (math.sqrt(3) / 2) * dx + 0.5 * dy)
            best = np.minimum(best, h)
    solid = best >= a - RIB / 2                          # within half a rib of the cell boundary
    return _to_elements((solid | _keep_solid(X, Y)) & ~_holes(X, Y))


def min_member_ok(xe):
    """fraction of material removed by a morphological opening with a 6 mm disk (0 = no thinner members)"""
    a = SIDE / NC; r = int(round((RIB / 2) / a)) - 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; disk = xx ** 2 + yy ** 2 <= r * r
    X = xe.reshape(NC, NC).astype(bool)
    op = ndimage.binary_opening(X, structure=disk)
    return float((X & ~op).sum() / X.sum())


def run_cutouts(F_total=None):
    F_total = plate_load_N() if F_total is None else F_total
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tabd = os.path.join(data.OUT, "tables"); figd = os.path.join(data.OUT, "figs")
    KE, edof, F, free, corner, centre, a = plate_setup(n=NC, F_total=F_total)
    hole = hole_mask(NC)
    xs = np.ones(NC * NC); xs[hole] = 0
    c_solid, _ = compliance(NC, NC, np.maximum(xs, 1e-9), 3, KE, edof, F, free)    # solid plate WITH bolt holes
    ev = lambda xe: compliance(NC, NC, np.maximum(xe, 1e-9), 3, KE, edof, F, free)[0]
    rows, keep = [], {}
    for name, fn, sizes in [("triangle", triangle_pattern, np.linspace(20, 110, 14)),
                            ("hexagon", hexagon_pattern, np.linspace(16, 80, 14)),
                            ("diagonal", diagonal_pattern, np.linspace(16, 110, 14))]:
        for s in sizes:
            xe = fn(s); vf = float(xe.mean())
            if vf > 0.85: continue
            c = ev(xe)
            rows.append(dict(pattern=name, cell_mm=round(float(s), 2), volfrac=vf, plate_mass_g=plate_mass(vf) * 1000,
                             c_over_solid=c / c_solid, thin_member_frac=min_member_ok(xe)))
            keep[(name, round(float(s), 2))] = xe
    # SIMP optimum: optimised on the standard M2 mesh (64x64), thresholded (= waterjet part), then
    # up-sampled to this 128 mesh and evaluated with the SAME FE model as the cut-out patterns
    KE6, ed6, F6, fr6, co6, ce6, a6 = plate_setup(n=64, F_total=F_total)
    h6 = hole_mask(64)
    for vf in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        xP, c, h, _ = simp(64, 64, vf, 3, 6.0 / a6, KE6, ed6, F6, fr6, passive_solid=co6 | ce6, passive_void=h6, maxit=150, tol=0.005)
        xb6 = (xP > 0.5).astype(float); xb6[co6 | ce6] = 1; xb6[h6] = 0
        xb = np.kron(xb6.reshape(64, 64), np.ones((2, 2))).ravel(); xb[corner | centre] = 1; xb[hole] = 0
        rows.append(dict(pattern="SIMP (thresholded)", cell_mm=np.nan, volfrac=float(xb.mean()), plate_mass_g=plate_mass(float(xb.mean())) * 1000,
                         c_over_solid=ev(xb) / c_solid, thin_member_frac=min_member_ok(xb)))
        keep[("SIMP", vf)] = xb
    df = pd.DataFrame(rows)
    df["connected"] = df.c_over_solid < 100          # huge compliance = battery pad cut off from the frame
    df.to_csv(os.path.join(tabd, "m2_cutouts.csv"), index=False)
    # compare at equal mass: interpolate each pattern's compliance at common masses (log scale)
    cmp = []
    ok = df[df.connected]
    tri = ok[ok.pattern == "triangle"].sort_values("plate_mass_g"); hexa = ok[ok.pattern == "hexagon"].sort_values("plate_mass_g")
    diag = ok[ok.pattern == "diagonal"].sort_values("plate_mass_g")
    sim = ok[ok.pattern.str.startswith("SIMP")].sort_values("plate_mass_g")
    lo = max(tri.plate_mass_g.min(), hexa.plate_mass_g.min(), diag.plate_mass_g.min())
    hi = min(tri.plate_mass_g.max(), hexa.plate_mass_g.max(), diag.plate_mass_g.max())
    for m in np.linspace(lo, hi, 6):
        ct = float(np.exp(np.interp(m, tri.plate_mass_g, np.log(tri.c_over_solid))))
        ch = float(np.exp(np.interp(m, hexa.plate_mass_g, np.log(hexa.c_over_solid))))
        cd = float(np.exp(np.interp(m, diag.plate_mass_g, np.log(diag.c_over_solid))))
        cs = float(np.exp(np.interp(m, sim.plate_mass_g, np.log(sim.c_over_solid)))) if sim.plate_mass_g.min() <= m <= sim.plate_mass_g.max() else np.nan
        cmp.append(dict(plate_mass_g=m, triangle=ct, hexagon=ch, diagonal=cd, SIMP=cs, hex_over_tri=ch / ct, diag_over_tri=cd / ct))
    cmp = pd.DataFrame(cmp); cmp.to_csv(os.path.join(tabd, "m2_cutouts_equal_mass.csv"), index=False)
    # mass needed to meet the brief's stiffness limit (compliance <= 2.5 x solid)
    lim = {}
    for nm, d in [("triangle", tri), ("hexagon", hexa), ("diagonal", diag), ("SIMP", sim)]:
        # lightest plate of this family whose compliance is <= 2.5 x solid (from the computed designs,
        # then refined by linear interpolation between that design and the next lighter one)
        d = d.sort_values("plate_mass_g").reset_index(drop=True)
        okk = d[d.c_over_solid <= 2.5]
        if okk.empty: lim[nm] = np.nan; continue
        i = okk.index.min()
        if i == 0: lim[nm] = float(d.plate_mass_g[0]); continue
        m0, c0, m1, c1 = d.plate_mass_g[i - 1], d.c_over_solid[i - 1], d.plate_mass_g[i], d.c_over_solid[i]
        lim[nm] = float(m0 + (m1 - m0) * (np.log(c0) - np.log(2.5)) / (np.log(c0) - np.log(c1)))
    # figure
    fig = plt.figure(figsize=(19, 9.5)); gs = fig.add_gridspec(2, 5)
    ax = fig.add_subplot(gs[:, :2])
    for nm, d, st in [("triangle cut-outs (isogrid, 0/60/120 deg)", tri, "^-"), ("hexagon cut-outs (honeycomb)", hexa, "h-"),
                      ("diagonal grid (45 deg, along load paths)", diag, "D-"), ("SIMP optimum, thresholded", sim, "o-")]:
        ax.semilogy(d.plate_mass_g, d.c_over_solid, st, label=nm, ms=6)
    ax.axhline(2.5, color="r", ls=":", label="limit: 2.5 x solid")
    ax.set_xlabel("plate mass (g)"); ax.set_ylabel("compliance / solid plate (log)"); ax.grid(alpha=.3, which="both")
    ax.set_title("Stiffness vs mass (all with 8 bolt holes): lower = stiffer at the same mass"); ax.legend(fontsize=9)
    target = 150.0
    def nearest(nm):
        d = ok[ok.pattern == nm]; r = d.iloc[(d.plate_mass_g - target).abs().argmin()]
        return keep[(nm, r.cell_mm)], r
    for k, nm in enumerate(["triangle", "hexagon", "diagonal"]):
        xe, r = nearest(nm)
        a2 = fig.add_subplot(gs[0, 2 + k]); a2.imshow(-xe.reshape(NC, NC).T, cmap="gray", origin="lower", extent=[0, 160, 0, 160])
        a2.set_title(f"{nm}: cell {r.cell_mm:.0f} mm, {r.plate_mass_g:.0f} g\nC/C_solid = {r.c_over_solid:.2f}", fontsize=9); a2.axis("off")
    simp_keys = [k for k in keep if k[0] == "SIMP"]
    j = int((sim.plate_mass_g - 113).abs().values.argmin()); s_ = sim.iloc[j]
    a3 = fig.add_subplot(gs[1, 2]); a3.imshow(-keep[simp_keys[j]].reshape(NC, NC).T, cmap="gray", origin="lower", extent=[0, 160, 0, 160])
    a3.set_title(f"SIMP: {s_.plate_mass_g:.0f} g, C/C_solid = {s_.c_over_solid:.2f}", fontsize=9); a3.axis("off")
    a4 = fig.add_subplot(gs[1, 3:]); a4.axis("off")
    txt = "Plate mass to meet C <= 2.5 x solid:\n" + "\n".join(f"  {k:9s}: {v:5.0f} g" if v == v else f"  {k:9s}: not reached" for k, v in lim.items())
    txt += "\n\nCompliance at equal mass, relative to triangle:\n   mass   hexagon  diagonal\n" + "\n".join(
        f"  {r.plate_mass_g:5.0f} g   {r.hex_over_tri:5.2f}    {r.diag_over_tri:5.2f}" for r in cmp.itertuples())
    a4.text(0, 1, txt, va="top", family="monospace", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(figd, "m2_cutouts.png"), dpi=115); plt.close()
    summ = dict(c_solid=c_solid, mesh=f"{NC}x{NC}", rib_mm=RIB, rim_mm=RIM, mass_for_limit_g=lim,
                hex_over_tri_at_equal_mass=cmp.hex_over_tri.round(3).tolist(), diag_over_tri_at_equal_mass=cmp.diag_over_tri.round(3).tolist())
    json.dump(summ, open(os.path.join(tabd, "m2_cutouts_summary.json"), "w", encoding="utf-8"), indent=1, default=float)
    return df, cmp, lim


if __name__ == "__main__":
    # python m2_topopt.py               -> everything: MBB + checks + vf sweep + outline, stresses,
    #                                      cut-out comparison, M1 re-run              (~10 min)
    # python m2_topopt.py --no-cutouts  -> skip the cut-out comparison                (~5 min)
    # python m2_topopt.py --m1-only     -> only re-run M1 with the saved plate mass   (~5 s)
    if "--m1-only" in sys.argv:
        s = json.load(open(os.path.join(data.OUT, "tables", "m2_summary.json"), encoding="utf-8"))
    else:
        pd.set_option("display.width", 220)
        print("=== PART 1: SIMP topology optimisation ===")
        s, df = run()
        print("MBB benchmark:"); print(pd.DataFrame(s["mbb"]).round(3).to_string(index=False))
        print("\nPlate element vs Timoshenko:"); print(pd.DataFrame(s["plate_element_check"]).round(4).to_string(index=False))
        sc = s["simp_sensitivity_check"]
        print(f"\nSIMP gradient check (analytic vs finite difference, {sc['n_elements']} elements): max rel. error {sc['max_rel_err']:.1e}")
        print("\nVolume-fraction sweep:"); print(df[["volfrac", "plate_mass_kg", "c_over_solid", "frac_material_thinner_than_6mm"]].round(3).to_string(index=False))
        print(f"\nPicked: vf = {s['picked_volfrac']}, plate mass = {s['picked_plate_mass_kg']*1000:.0f} g (lightest with C <= 2.5 x solid)")
        print("Pictures: outputs/figs/m2_mbb_benchmark.png, m2_plate_sweep.png, m2_plate_outline.png; cut file: outputs/tables/m2_plate_outline.dxf")
        print("\n=== PART 2: finite-element stresses ===")
        tab = run_stress()
        print(tab.round(3).to_string(index=False)); print("Picture: outputs/figs/m2_stress.png")
        if "--no-cutouts" not in sys.argv:
            print("\n=== PART 3: cut-out patterns vs SIMP ===")
            dfc, cmp, lim = run_cutouts()
            print("Compliance / solid at EQUAL plate mass:"); print(cmp.round(3).to_string(index=False))
            print("Plate mass needed to meet compliance <= 2.5 x solid:", {k: (round(v) if v == v else None) for k, v in lim.items()})
            print("Picture: outputs/figs/m2_cutouts.png")
    rerun_m1(s["picked_plate_mass_kg"])
