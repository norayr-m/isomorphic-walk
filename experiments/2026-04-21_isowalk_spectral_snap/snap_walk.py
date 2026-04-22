#!/usr/bin:env python3
"""
IsoWalk spectral snap walk against a forced target — LOCKED implementation.

Protocol source of truth: DeepThink-Dusk registrar lock, 2026-04-21.
File: export/2026-04-21_0530_isowalk_spectral_snap_prereg_deepthink/
      DEEPTHINK_RESPONSE.md (when captured) — math frozen per his block 1.

Algorithm (verbatim from lock):
  1. Build 5.0 Å pure Euclidean Cα Laplacian L_A for start conformer A
     and L_B for end conformer B.
  2. Fiedler eigenvectors v_A, v_B — eigenvectors of L_A, L_B for λ₂.
  3. Sign alignment: if v_A · v_B < 0, invert v_B ← -v_B.
  4. Target gradient: Δv = v_B - v_A.
  5. Walk step: r* = argmax_r [(state_after(r) - state_before(r)) × Δv[r]]
     subject to per-node cap of 1. No tie-breakers (resolve literal ties
     by lowest index — arbitrary, neutral, DT forbids degree/sequence
     weighting).
  6. Halt when no legal flip has positive (state_after - state_before)×Δv[r].

Given the ternary ratchet (-1 → 0 → +1 → -1) with cap=1 and initial
state = 0 everywhere, the only allowed per-residue transition is 0 → +1
with advancement coefficient +1. The argmax rule therefore reduces to:
  Flip residues in order of decreasing Δv[r], stopping when Δv[r] ≤ 0.
Per-residue first-snap-frame = rank position of residue among flipped.
Unreached residues (Δv[r] ≤ 0) get -1.

Implementation discipline: NO heuristics, NO tie-breakers other than
lowest-index resolve. NO distance lookaheads. NO sequence-proximity.
NO parameter tuning. Graph is 5.0 Å pure Euclidean Cα. Frozen.

Baselines:
  - Hinge-Euclidean (mandatory): seed = argmax |v_A[i]|. Euclidean 3D
    distance from coords[seed] to all other Cα. This is the null DT
    demanded.

Metrics:
  - H1a per-residue: Spearman ρ(first-snap-frame, |Kabsch-displacement|)
    on residues with frame ≥ 0. Compare Δρ = ρ_snap - ρ_euclid where
    ρ_euclid is on the same subset.
  - H1b whole-system: Spearman ρ across the cohort of (total-snap-count,
    1 - TM-score) — TM-score from tmtools.
"""
import json, sys
from pathlib import Path
import numpy as np
import gemmi
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from tmtools import tm_align

SCRATCH = Path("/tmp/validate_isomorphic")
CUTOFF = 5.0  # Å, locked
LANCZOS_SIGMA = 1e-3  # small positive shift to dodge zero-eigenvalue plateau on disconnected graphs

def extract_chain_CA(cif, chain_id="A"):
    """Return list of (seqid, residue name, 1-letter code, xyz) for chain_id CA atoms."""
    s = gemmi.read_structure(str(cif))
    ch = next((c for c in s[0] if c.name == chain_id), None)
    if ch is None:
        # Fallback: first chain with ≥ 30 CAs
        for c in s[0]:
            n_ca = sum(1 for r in c for a in r if a.name == "CA")
            if n_ca >= 30:
                ch = c; break
        if ch is None: return []
    out = []
    for r in ch:
        for a in r:
            if a.name == "CA":
                tab = gemmi.find_tabulated_residue(r.name)
                oneletter = tab.one_letter_code.upper() if tab and tab.one_letter_code else "X"
                out.append((r.seqid.num, r.name, oneletter, (a.pos.x, a.pos.y, a.pos.z)))
                break
    return out

def match_within_window(atoms_A, atoms_B, window=2):
    """Match by seqid with ±window tolerance, preferring exact residue-type match."""
    bd_by_sid = {a[0]: a for a in atoms_B}
    matched = []
    for aA in atoms_A:
        sid_A = aA[0]
        # Look for exact match first, then ±1, then ±2
        for delta in range(0, window+1):
            for sign in (0, +1, -1) if delta > 0 else (0,):
                sid_try = sid_A + sign * delta
                if sid_try in bd_by_sid:
                    aB = bd_by_sid[sid_try]
                    matched.append((aA, aB))
                    del bd_by_sid[sid_try]
                    break
            else: continue
            break
    return matched

def kabsch(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Qc.T @ Pc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = P.mean(0) - R @ Q.mean(0)
    return R, t

def build_laplacian(coords, cutoff=CUTOFF):
    n = coords.shape[0]
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=cutoff, output_type="ndarray")
    if len(pairs) == 0: return None, 0
    rows = []; cols = []
    for a, b in pairs:
        rows.extend([int(a), int(b)]); cols.extend([int(b), int(a)])
    data = -np.ones(len(rows), dtype=np.float64)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    deg = -np.asarray(A.sum(axis=1)).ravel()
    diag = csr_matrix((deg, (np.arange(n), np.arange(n))), shape=(n, n))
    return A + diag, int(len(pairs))

def canonical_fiedler(L):
    """Compute Fiedler eigenvector of Laplacian L with disconnected-graph robustness.
    Returns (v, lambda_2, ncomp)."""
    # Number of components = number of zero eigenvalues
    Aadj = (L != 0).astype(np.int32); Aadj.setdiag(0); Aadj.eliminate_zeros()
    ncomp, _ = connected_components(Aadj, directed=False)
    n = L.shape[0]
    # Ask for ncomp + 2 eigenvalues around zero; the (ncomp+1)th is λ_2
    k = min(ncomp + 2, n - 1)
    eigvals, eigvecs = eigsh(L, k=k, sigma=LANCZOS_SIGMA, which="LM")
    order = np.argsort(eigvals)
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    # First non-trivial eigenvalue is at index = ncomp (after ncomp zero eigenvalues)
    fiedler_idx = ncomp
    if fiedler_idx >= k:
        # Fallback: take smallest positive eigenvalue we have
        nonzero = np.where(eigvals > 1e-6)[0]
        fiedler_idx = nonzero[0] if len(nonzero) else 0
    v = eigvecs[:, fiedler_idx]
    lam = eigvals[fiedler_idx]
    # Canonicalise sign: first non-zero entry positive
    for i in range(len(v)):
        if abs(v[i]) > 1e-9:
            if v[i] < 0: v = -v
            break
    return v, float(lam), int(ncomp)

def snap_walk_locked(v_A, v_B):
    """Compute first-snap-frame and total-snap-count per the locked rule.
    Sign-align v_B to v_A (flip if dot product negative), form Δv,
    then rank residues by descending Δv, flipping only those with Δv > 0.
    """
    if v_A @ v_B < 0: v_B = -v_B
    dv = v_B - v_A
    n = len(dv)
    # Positive-Δv residues get flipped in descending order; ties broken by lowest index.
    # Argsort with a secondary key of index ensures deterministic tie resolution.
    pos_mask = dv > 0
    pos_indices = np.where(pos_mask)[0]
    # Sort positive by (-dv, index): descending Δv, ascending index for ties
    sort_order = sorted(pos_indices, key=lambda i: (-float(dv[i]), int(i)))
    frame_of = np.full(n, -1, dtype=np.int32)
    for rank, i in enumerate(sort_order, start=1):
        frame_of[i] = rank
    total_snap_count = len(sort_order)
    return frame_of, int(total_snap_count), dv

def run_pair(cif_A, cif_B, label, chain_A="A", chain_B="A"):
    atoms_A = extract_chain_CA(SCRATCH / cif_A, chain_A)
    atoms_B = extract_chain_CA(SCRATCH / cif_B, chain_B)
    if len(atoms_A) < 20 or len(atoms_B) < 20:
        return {"id": label, "status": "PIPELINE_FAILURE", "reason": "too few atoms"}
    matched = match_within_window(atoms_A, atoms_B, window=2)
    if len(matched) < 20:
        return {"id": label, "status": "PIPELINE_FAILURE",
                "reason": f"only {len(matched)} matched residues"}
    Po = np.array([m[0][3] for m in matched])
    Pc = np.array([m[1][3] for m in matched])
    R, t = kabsch(Po, Pc); Pc_al = (R @ Pc.T).T + t
    disp = np.linalg.norm(Po - Pc_al, axis=1)

    L_A, n_edges_A = build_laplacian(Po, CUTOFF)
    L_B, n_edges_B = build_laplacian(Pc_al, CUTOFF)  # use aligned B coords for consistent graph
    if L_A is None or L_B is None:
        return {"id": label, "status": "PIPELINE_FAILURE",
                "reason": "empty graph at 5Å"}
    try:
        v_A, lam_A, ncomp_A = canonical_fiedler(L_A)
        v_B, lam_B, ncomp_B = canonical_fiedler(L_B)
    except Exception as e:
        return {"id": label, "status": "PIPELINE_FAILURE",
                "reason": f"Lanczos failed: {type(e).__name__}: {e}"}

    frame_of, total_snap_count, dv = snap_walk_locked(v_A, v_B)

    # DT amendment 2026-04-21: transform frame into Predicted Mobility Score (PMS).
    # PMS[r] = (N_snaps + 1 - t) if snapped at frame t, else 0.
    # Unsnapped residues tied at zero (bottom), snapped residues monotone-positive.
    # High PMS <=> early flip <=> predicted high mobility. ρ > 0 natural.
    n = len(frame_of)
    PMS = np.zeros(n, dtype=np.float64)
    snapped_mask = frame_of >= 1
    PMS[snapped_mask] = float(total_snap_count + 1) - frame_of[snapped_mask].astype(np.float64)

    # Hinge-Euclidean baseline: seed = argmax |v_A|
    hinge_seed = int(np.argmax(np.abs(v_A)))
    euclid = np.linalg.norm(Po - Po[hinge_seed], axis=1)

    # Correlations — both on full N residues, PMS naturally handles unsnapped via zero-tie block
    rho_snap, p_snap = spearmanr(PMS, disp)
    rho_euc, p_euc = spearmanr(euclid, disp)
    d_rho = rho_snap - rho_euc

    # TM-score
    ca_A = np.array([m[0][3] for m in matched])
    ca_B = np.array([m[1][3] for m in matched])
    seq_A = "".join(m[0][2] for m in matched)
    seq_B = "".join(m[1][2] for m in matched)
    try:
        tm = tm_align(ca_A, ca_B, seq_A, seq_B)
        tm_score = float(tm.tm_norm_chain1)
    except Exception as e:
        tm_score = None

    return {
        "id": label, "status": "OK",
        "n_matched": len(matched),
        "n_edges_A": n_edges_A, "n_edges_B": n_edges_B,
        "n_components_A": ncomp_A, "n_components_B": ncomp_B,
        "lambda_2_A": lam_A, "lambda_2_B": lam_B,
        "hinge_seed_index": hinge_seed,
        "total_snap_count": total_snap_count,
        "reached_count": int(snapped_mask.sum()),
        "rho_snap_vs_disp": float(rho_snap),
        "rho_euclid_vs_disp": float(rho_euc),
        "delta_rho": float(d_rho),
        "tm_score": tm_score,
        "rmsd_kabsch_A": float(np.sqrt((disp**2).mean())),
    }

if __name__ == "__main__":
    # Pilot pairs (declared, results-not-counted per DT lock)
    PILOTS = [
        ("Calmodulin-hinge", "1CLL.cif", "1CTR.cif", "A", "A"),
        ("TIM-shear",        "1TIM.cif", "7TIM.cif", "A", "A"),
    ]
    # Confirmatory 10 (locked)
    COHORT = [
        ("HIV-1 Protease",      "1HHP.cif", "1HTG.cif", "A", "A"),
        ("Klenow Pol I",        "1KFD.cif", "1KLN.cif", "A", "A"),
        ("Lactoferrin",         "1LFH.cif", "1LFG.cif", "A", "A"),
        ("Yeast Hexokinase",    "2YHX.cif", "1HKG.cif", "A", "A"),
        ("GlutamineBP",         "1GGG.cif", "1WDN.cif", "A", "A"),
        ("RiboseBP",            "2DRI.cif", "1URP.cif", "A", "A"),
        ("Citrate Synthase",    "1CTS.cif", "2CTS.cif", "A", "A"),
        ("T4 Lysozyme",         "2LZM.cif", "1CT0.cif", "A", "A"),
        ("Enolase",             "3ENL.cif", "7ENL.cif", "A", "A"),
        ("GroEL subunit",       "1OEL.cif", "1AON.cif", "A", "A"),
    ]

    which = sys.argv[1] if len(sys.argv) > 1 else "pilot"

    if which == "pilot":
        print("=== PILOTS (declared, results-not-counted per DT lock) ===\n")
        pilot_results = []
        for label, cif_A, cif_B, ch_A, ch_B in PILOTS:
            r = run_pair(cif_A, cif_B, label, ch_A, ch_B)
            pilot_results.append(r)
            if r["status"] == "OK":
                print(f"{r['id']:>20}  n={r['n_matched']:>4}  ncomp(A,B)=({r['n_components_A']},{r['n_components_B']})  "
                      f"ρ_snap={r['rho_snap_vs_disp']:+.4f}  ρ_euc={r['rho_euclid_vs_disp']:+.4f}  "
                      f"Δρ={r['delta_rho']:+.4f}  snaps={r['total_snap_count']}/{r['n_matched']}  "
                      f"TM={('%.3f' % r['tm_score']) if r['tm_score'] is not None else 'n/a'}")
            else:
                print(f"{r['id']:>20}  {r['status']}: {r.get('reason','?')}")
        out = Path(__file__).parent / "pilot_results.json"
        json.dump({"note": "pilot, results-not-counted", "rows": pilot_results}, open(out, "w"), indent=2)
    elif which == "cohort":
        print("=== CONFIRMATORY COHORT (locked, submitted to DT) ===\n")
        cohort_results = []
        for label, cif_A, cif_B, ch_A, ch_B in COHORT:
            r = run_pair(cif_A, cif_B, label, ch_A, ch_B)
            cohort_results.append(r)
            if r["status"] == "OK":
                print(f"{r['id']:>20}  n={r['n_matched']:>4}  "
                      f"ρ_snap={r['rho_snap_vs_disp']:+.4f}  Δρ={r['delta_rho']:+.4f}  "
                      f"snaps={r['total_snap_count']}/{r['n_matched']}  "
                      f"TM={('%.3f' % r['tm_score']) if r['tm_score'] is not None else 'n/a'}")
            else:
                print(f"{r['id']:>20}  {r['status']}: {r.get('reason','?')}")
        # H1a summary
        oks = [r for r in cohort_results if r["status"] == "OK"]
        if oks:
            rhos = [r["rho_snap_vs_disp"] for r in oks]
            deltas = [r["delta_rho"] for r in oks]
            wins = sum(1 for d in deltas if d > 0)
            print(f"\nH1a: median ρ_snap = {np.median(rhos):+.4f}, wins {wins}/{len(oks)} = {100*wins/len(oks):.0f}%")
            # H1b
            pairs_tm = [(r["total_snap_count"], 1 - r["tm_score"]) for r in oks if r["tm_score"] is not None]
            if len(pairs_tm) >= 5:
                xs, ys = zip(*pairs_tm)
                rho_H1b, p_H1b = spearmanr(xs, ys)
                print(f"H1b: Spearman ρ(snap_count, 1-TM) = {rho_H1b:+.4f}  p = {p_H1b:.2e}  n = {len(pairs_tm)}")
        out = Path(__file__).parent / "cohort_results.json"
        json.dump({"note": "confirmatory cohort per DT lock", "rows": cohort_results}, open(out, "w"), indent=2)
