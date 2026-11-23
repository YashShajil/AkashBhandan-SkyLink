"""M2 - hub plate topology optimisation (SIMP, density filter, OC update).

Two FE models:
  * plane-stress Q4 (top88-style) for the MBB benchmark (verification of the optimiser);
  * Mindlin-Reissner Q4 plate-bending element for the hub plate, because the arm reactions
    (rotor thrust) and the battery weight both act OUT of the plate's plane.  An in-plane
    model would optimise the wrong load path.  The plate element itself is verified against
    Timoshenko's simply-supported square plate.
Units: mm, N, MPa.
"""
from __future__ import annotations
import os, json, math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy import ndimage
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


def plate_setup(n=80, side=160.0, support_mm=10.0, load_mm=40.0, F_total=1.0):
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
    corner = np.zeros(n*n, bool)
    for cx in (0, side):
        for cy in (0, side):
            corner |= (np.abs(EX-cx) < support_mm) & (np.abs(EY-cy) < support_mm)
    centre = (np.abs(EX-side/2) < load_mm/2) & (np.abs(EY-side/2) < load_mm/2)
    # load: uniform pressure over central patch, lumped to nodes
    F = np.zeros(ndof); q = F_total/(load_mm**2)
    for e in np.where(centre)[0]:
        for k in range(4): F[edof[e, 3*k]] += -q*a*a/4
    # supports: w = 0 at nodes of corner patches (arm clamps)
    xs = np.arange(nn)*a; NX, NY = np.meshgrid(xs, xs, indexing="ij"); NX = NX.ravel(); NY = NY.ravel()
    sup = np.zeros(nn*nn, bool)
    for cx in (0, side):
        for cy in (0, side):
            sup |= (np.abs(NX-cx) <= support_mm) & (np.abs(NY-cy) <= support_mm)
    fixed = 3*np.where(sup)[0]
    # remove rigid-body rotations: corners give 4 w-supports -> stable
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


def plate_mass(vf):
    return vf*0.16*0.16*0.003*RHO_AL + 0.04


def run(F_total=None, n=64, vfs=(0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import pandas as pd
    figd = os.path.join(data.OUT, "figs"); tabd = os.path.join(data.OUT, "tables")
    mbb, beam = run_mbb(figd)
    pv = verify_plate_element()
    if F_total is None:
        F_total = 2*6.1*9.81   # battery of the baseline design (~6.1 kg) at the 2g max-thrust condition
    KE, edof, F, free, corner, centre, a = plate_setup(n=n, F_total=F_total)
    c_solid, _ = compliance(n, n, np.ones(n*n), 3, KE, edof, F, free)
    rmin = 6.0/a   # 6 mm filter radius in elements -> member width >= ~6 mm
    rows, designs = [], {}
    prev = None
    for vf in vfs:
        xP, c, hist, _ = simp(n, n, vf, 3, rmin, KE, edof, F, free, passive_solid=corner | centre, maxit=150, tol=0.005)
        # crisp (waterjet) version: threshold at 0.5, re-evaluate
        xb = (xP > 0.5).astype(float); xb[corner | centre] = 1
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
    xP = designs[pick].reshape(n, n).T
    fig, ax = plt.subplots(figsize=(5, 5))
    cs = ax.contour(np.linspace(a/2, 160-a/2, n), np.linspace(a/2, 160-a/2, n), xP, levels=[0.5], colors="k")
    ax.set_aspect("equal"); ax.set_xlim(0, 160); ax.set_ylim(0, 160)
    ax.add_patch(plt.Rectangle((0, 0), 160, 160, fill=False, ls="--", color="grey"))
    for cx in (10, 150):
        for cy in (10, 150): ax.add_patch(plt.Circle((cx, cy), 2.5, fill=False, color="b"))   # M5 arm-clamp bolts
    ax.set_title(f"Waterjet outline, vf={pick}, mass={plate_mass(pick)*1000:.0f} g\n(blue: arm clamp bolt holes, M5)")
    plt.savefig(os.path.join(figd, "m2_plate_outline.png"), dpi=130); plt.close()
    segs = [p for p in cs.allsegs[0] if len(p) > 4]
    with open(os.path.join(tabd, "m2_plate_outline.dxf"), "w") as f:
        f.write("0\nSECTION\n2\nENTITIES\n")
        for p in segs:
            f.write("0\nLWPOLYLINE\n8\nCUT\n90\n%d\n70\n1\n" % len(p))
            for x, y in p: f.write(f"10\n{x:.3f}\n20\n{y:.3f}\n")
        f.write("0\nENDSEC\n0\nEOF\n")
    df.to_csv(os.path.join(tabd, "m2_plate_sweep.csv"), index=False)
    summ = dict(mbb=mbb, mbb_fe_check=beam, plate_element_check=pv, c_solid=c_solid, F_total_N=F_total,
                picked_volfrac=pick, picked_plate_mass_kg=plate_mass(pick), rmin_mm=6.0)
    json.dump(summ, open(os.path.join(tabd, "m2_summary.json"), "w"), indent=1, default=float)
    return summ, df


if __name__ == "__main__":
    s, df = run()
    print(json.dumps(s, indent=1, default=float)); print(df.round(4))
