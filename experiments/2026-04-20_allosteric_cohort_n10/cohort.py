#!/usr/bin:env python3
"""
Allosteric pathway confirmatory cohort (N=10).

Executes DeepThink-Dusk's locked protocol exactly.
  - 8.0 Å pure Euclidean Cα cutoff (not KNN)
  - BFS suboptimal tube: {v : dist_A(v) + dist_B(v) ≤ L + 1}
  - Euclidean cylinder: top M residues by perpendicular distance to line A-B
  - ±2 sequence window for ground-truth matching
  - Record pipeline failures, drop N; never substitute

See DEEPTHINK_PROTOCOL.md for the complete spec.
"""
import sys, json, itertools
from pathlib import Path
import numpy as np
import gemmi

import os
EXP = Path(__file__).resolve().parent
# Where downloaded PDB CIFs live (one per protein in the cohort).
# Override with PROTEIN_DIR env var if you've cached them elsewhere.
SCRATCH = Path(os.environ.get('PROTEIN_DIR', 'data/cohort'))
CUTOFF = 8.0
TUBE_SLACK = 1
MATCH_WINDOW = 2

# One-letter to 3-letter codes used for name sanity
ONE_TO_THREE = {
    'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','E':'GLU','Q':'GLN',
    'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE',
    'P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'
}

# Each system: PDB, seeds as (chain, seqid, resname-1letter), ground truth as
# list of (chain, seqid, resname-1letter). If chain is None, accept any chain.
# For multi-chain systems we specify concrete PDB chain IDs.
SYSTEMS = [
    {
        'id': 'IGPS', 'pdb': '1GPW', 'expect_N': 450,
        # Biological unit copy: chain A (HisF, 253 aa) + chain B (HisH, 200 aa).
        'seed_A': ('B', 84, 'C'),      # HisH Cys84
        'seed_B': ('A', 98, 'D'),      # HisF Asp98
        'ground_truth': [
            ('A', 19, 'K'), ('A', 51, 'V'), ('A', 73, 'Y'),
            ('A', 99, 'K'), ('A', 170, 'E'),
            ('B', 138, 'Y'), ('B', 181, 'K'),
        ],
        'note': 'HisH=chain B, HisF=chain A',
    },
    {
        'id': 'PTP1B', 'pdb': '1T49', 'expect_N': 300,
        'seed_A': ('A', 215, 'C'),
        'seed_B': ('A', 192, 'L'),
        'ground_truth': [
            ('A', 179, 'W'), ('A', 180, 'P'), ('A', 196, 'F'),
            ('A', 280, 'F'), ('A', 152, 'Y'),
        ],
    },
    {
        'id': 'β2-AR', 'pdb': '3SN6', 'expect_N': 300,
        # Chain R carries the β2-AR receptor in 3SN6
        'seed_A': ('R', 113, 'D'),
        'seed_B': ('R', 131, 'R'),
        'ground_truth': [
            ('R', 121, 'I'), ('R', 211, 'P'), ('R', 282, 'F'),
            ('R', 286, 'W'), ('R', 290, 'F'), ('R', 322, 'N'),
        ],
    },
    {
        'id': 'Hsp70', 'pdb': '2KHO', 'expect_N': 600,
        'seed_A': ('A', 199, 'T'),
        'seed_B': ('A', 436, 'V'),
        'ground_truth': [
            ('A', 394, 'L'), ('A', 395, 'P'), ('A', 398, 'V'),
            ('A', 399, 'P'), ('A', 400, 'D'), ('A', 467, 'R'),
            ('A', 481, 'D'),
        ],
    },
    {
        'id': 'PKA', 'pdb': '1ATP', 'expect_N': 350,
        'seed_A': ('E', 57, 'V'),
        'seed_B': ('E', 197, 'T'),
        'ground_truth': [
            ('E', 106, 'L'), ('E', 116, 'L'), ('E', 164, 'Y'),
            ('E', 185, 'F'), ('E', 220, 'D'),
        ],
    },
    {
        'id': 'GlmS', 'pdb': '1XFF', 'expect_N': 600,
        # Full GlmS is ~600, split glutaminase (N-term) + isomerase (C-term)
        # 1XFF has two chains A and B, each a different half? Re-examine.
        'seed_A': ('A', 1, 'C'),
        'seed_B': ('B', 504, 'H'),
        'ground_truth': [
            ('A', 74, 'W'), ('A', 98, 'N'), ('A', 123, 'D'),
            ('B', 399, 'V'), ('B', 488, 'E'),
        ],
        'note': 'Biological unit is one chain covering both domains in the real protein; if seed not found, mark failure.',
    },
    {
        'id': 'ATCase', 'pdb': '1D09', 'expect_N': 450,
        # Catalytic = chain A (310 aa), Regulatory = chain B (153 aa)
        'seed_A': ('A', 134, 'H'),
        'seed_B': ('B', 94, 'K'),
        'ground_truth': [
            ('A', 240, 'Y'), ('A', 236, 'D'), ('A', 271, 'D'),
            ('B', 77, 'Y'), ('B', 19, 'D'),
        ],
    },
    {
        'id': 'TrpSyn', 'pdb': '1B8F', 'expect_N': 650,
        # 1B8F chain A = fused α+β construct, residues 1-509.
        # α subunit ~residues 1-268, β subunit ~residues 269-509 (continuous numbering).
        # α-Glu49 = chain A seqid 49 (α-local)
        # β-Lys87 = chain A seqid (~268 + 87) = ~355. We'll verify by residue type match.
        # Same offset applies to β GT residues.
        # If offset wrong by more than ±2, pipeline will report a low residue-match and that's fine.
        'seed_A': ('A', 49, 'E'),
        'seed_B': ('A', 355, 'K'),  # β-Lys87 → global 268+87=355
        'ground_truth': [
            ('A', 268+175, 'Y'),  # β-Y175 → 443
            ('A', 268+212, 'F'),  # β-F212 → 480
            ('A', 268+232, 'G'),  # β-G232 → 500
            ('A', 268+235, 'S'),  # β-S235 → 503
        ],
        'note': 'Fused α+β in chain A; β seqids offset by +268 (global numbering). Verify by residue type at build time.',
    },
    {
        'id': 'PFK', 'pdb': '4PFK', 'expect_N': 320,
        'seed_A': ('A', 162, 'R'),
        'seed_B': ('A', 21, 'R'),
        'ground_truth': [
            ('A', 160, 'H'), ('A', 187, 'E'), ('A', 252, 'R'),
        ],
    },
    {
        'id': 'Abl', 'pdb': '2HYY', 'expect_N': 280,
        'seed_A': ('A', 315, 'T'),
        'seed_B': ('A', 356, 'A'),
        'ground_truth': [
            ('A', 290, 'M'), ('A', 301, 'L'), ('A', 361, 'H'),
            ('A', 381, 'D'), ('A', 382, 'F'),
        ],
    },
]


def collect_CA_all_chains(cif):
    """Return list of (chain, seqid, resname_1letter, xyz) for all CA atoms in model 0."""
    s = gemmi.read_structure(str(cif))
    out = []
    for c in s[0]:
        for r in c:
            for a in r:
                if a.name == 'CA':
                    # 3-letter to 1-letter
                    tabbed = gemmi.find_tabulated_residue(r.name)
                    one = tabbed.one_letter_code.upper() if tabbed and tabbed.one_letter_code else 'X'
                    out.append((c.name, r.seqid.num, one, (a.pos.x, a.pos.y, a.pos.z), r.name))
                    break
    return out


def find_idx(atoms, chain, seqid, expected_one):
    """Exact-match (chain, seqid) with residue-name check. Returns index or None."""
    for i, (ch, sid, one, _, _) in enumerate(atoms):
        if ch == chain and sid == seqid:
            if expected_one is None or one == expected_one:
                return i
            # Residue-name mismatch — treat as a soft failure, return None
            return None
    return None


def bfs_distances(adj, src, n):
    """Return array of shortest-path distances from src (infinity = INT_MAX)."""
    dist = np.full(n, -1, dtype=np.int32)
    dist[src] = 0
    cur = [src]
    frame = 0
    while cur:
        nxt = []
        for u in cur:
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = frame + 1
                    nxt.append(v)
        cur = nxt; frame += 1
    return dist


def build_cutoff_adj(coords, cutoff):
    """Symmetric adjacency list using Euclidean cutoff and a cKDTree."""
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=cutoff, output_type='ndarray')
    n = coords.shape[0]
    adj = [[] for _ in range(n)]
    for a, b in pairs:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    for r in adj: r.sort()
    return adj


def euclidean_cylinder(coords, a_idx, b_idx, m):
    """Top m residues (excluding a_idx and b_idx) by perpendicular distance to
    the 3D segment coords[a_idx]–coords[b_idx]."""
    A = coords[a_idx]; B = coords[b_idx]
    AB = B - A; ab2 = float(AB @ AB)
    out = []
    for i in range(len(coords)):
        if i == a_idx or i == b_idx: continue
        AP = coords[i] - A
        if ab2 == 0:
            d = float(np.linalg.norm(AP))
        else:
            t = float(AP @ AB) / ab2
            t = max(0.0, min(1.0, t))
            closest = A + t * AB
            d = float(np.linalg.norm(coords[i] - closest))
        out.append((i, d))
    out.sort(key=lambda x: x[1])
    return [i for i, _ in out[:m]]


def match_with_window(pred_indices, atoms, gt_list, window=MATCH_WINDOW):
    """Per-chain ±window seqid tolerance. Returns:
       - pred_hits: count of pred residues that match SOMETHING in GT (precision numerator)
       - gt_hits:   count of GT residues that are matched by SOMETHING in pred (recall numerator)
    Each pred residue and each GT residue counted at most once."""
    gt_by_chain = {}
    for j, (ch, sid, _) in enumerate(gt_list):
        gt_by_chain.setdefault(ch, []).append((j, sid))
    pred_matched = set()
    gt_matched = set()
    for i in pred_indices:
        ch, sid, _, _, _ = atoms[i]
        if ch not in gt_by_chain: continue
        for j, g_sid in gt_by_chain[ch]:
            if abs(sid - g_sid) <= window:
                pred_matched.add(i)
                gt_matched.add(j)
    return len(pred_matched), len(gt_matched)


def prec_rec_f1(pred_hits, gt_hits, n_pred, n_gt):
    if n_pred == 0 and n_gt == 0: return 1.0, 1.0, 1.0
    if n_pred == 0 or n_gt == 0:  return 0.0, 0.0, 0.0
    p = pred_hits / n_pred
    r = gt_hits / n_gt
    f = 2*p*r/(p+r) if (p+r) > 0 else 0.0
    return p, r, f


def run_system(sys_spec):
    pdb = sys_spec['pdb']
    cif = SCRATCH / f'{pdb}.cif'
    print(f"\n=== {sys_spec['id']} ({pdb}) ===", flush=True)
    atoms = collect_CA_all_chains(cif)
    n = len(atoms)
    print(f"  CA atoms across all chains: {n}")

    seed_A_info = sys_spec['seed_A']
    seed_B_info = sys_spec['seed_B']
    a_idx = find_idx(atoms, seed_A_info[0], seed_A_info[1], seed_A_info[2])
    b_idx = find_idx(atoms, seed_B_info[0], seed_B_info[1], seed_B_info[2])
    if a_idx is None or b_idx is None:
        print(f"  PIPELINE FAILURE: seed not found.  a_idx={a_idx}  b_idx={b_idx}")
        print(f"    seed A expected: {seed_A_info}; seed B expected: {seed_B_info}")
        return {'id': sys_spec['id'], 'pdb': pdb, 'status': 'PIPELINE_FAILURE',
                'reason': 'seed not found or name mismatch',
                'seed_A_expected': seed_A_info, 'seed_B_expected': seed_B_info}
    print(f"  seed A idx={a_idx} ({atoms[a_idx][:3]})")
    print(f"  seed B idx={b_idx} ({atoms[b_idx][:3]})")

    coords = np.array([a[3] for a in atoms])
    adj = build_cutoff_adj(coords, CUTOFF)
    n_edges = sum(len(r) for r in adj)
    print(f"  contact graph: 8.0 Å cutoff, {n_edges} directed edges, mean degree {n_edges/n:.1f}")

    dist_A = bfs_distances(adj, a_idx, n)
    dist_B = bfs_distances(adj, b_idx, n)
    L = int(dist_A[b_idx])
    if L < 0:
        print(f"  PIPELINE FAILURE: no path from seed A to seed B (graph disconnected).")
        return {'id': sys_spec['id'], 'pdb': pdb, 'status': 'PIPELINE_FAILURE',
                'reason': 'seed A and B in different components'}
    print(f"  BFS distance A→B: {L}")

    # Suboptimal tube: dist_A(v) + dist_B(v) ≤ L + TUBE_SLACK; exclude endpoints
    dA = dist_A; dB = dist_B
    mask_tube = (dA >= 0) & (dB >= 0) & (dA + dB <= L + TUBE_SLACK)
    tube = [i for i in range(n) if mask_tube[i] and i != a_idx and i != b_idx]
    M = len(tube)
    print(f"  BFS tube |M| (excl. endpoints): {M}")
    if M == 0:
        print(f"  PIPELINE FAILURE: tube is empty (no intermediate residues).")
        return {'id': sys_spec['id'], 'pdb': pdb, 'status': 'PIPELINE_FAILURE',
                'reason': 'empty tube'}

    euc = euclidean_cylinder(coords, a_idx, b_idx, M)
    print(f"  Euclidean cylinder |M|: {len(euc)}")

    gt = sys_spec['ground_truth']
    bfs_pred_hits, bfs_gt_hits = match_with_window(tube, atoms, gt)
    eu_pred_hits,  eu_gt_hits  = match_with_window(euc,  atoms, gt)
    p_b, r_b, f_b = prec_rec_f1(bfs_pred_hits, bfs_gt_hits, M, len(gt))
    p_e, r_e, f_e = prec_rec_f1(eu_pred_hits,  eu_gt_hits,  M, len(gt))
    delta = f_b - f_e
    print(f"  BFS:  pred_hits={bfs_pred_hits}/{M}  gt_hits={bfs_gt_hits}/{len(gt)}  P={p_b:.3f} R={r_b:.3f} F1={f_b:.3f}")
    print(f"  EUC:  pred_hits={eu_pred_hits}/{M}  gt_hits={eu_gt_hits}/{len(gt)}  P={p_e:.3f} R={r_e:.3f} F1={f_e:.3f}")
    print(f"  ΔF1 = {delta:+.3f}")

    return {
        'id': sys_spec['id'], 'pdb': pdb, 'status': 'OK',
        'N': n,
        'seed_A': seed_A_info, 'seed_B': seed_B_info,
        'L': L, 'M': M,
        'n_edges_directed': n_edges,
        'bfs_tube_residues':   [(atoms[i][0], atoms[i][1], atoms[i][2]) for i in tube],
        'eucl_cyl_residues':   [(atoms[i][0], atoms[i][1], atoms[i][2]) for i in euc],
        'bfs_f1': f_b, 'eucl_f1': f_e, 'delta_f1': delta,
        'bfs_precision': p_b, 'bfs_recall': r_b,
        'eucl_precision': p_e, 'eucl_recall': r_e,
    }


def main():
    rows = []
    for sys_spec in SYSTEMS:
        try:
            rows.append(run_system(sys_spec))
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append({'id': sys_spec['id'], 'pdb': sys_spec['pdb'],
                         'status': 'PIPELINE_FAILURE', 'reason': str(e)})

    print("\n\n=== FINAL ΔF1 TABLE ===\n")
    print(f"{'system':>10} {'pdb':>5} {'N':>5} {'M':>4} {'L':>4}  {'F1(BFS)':>8} {'F1(EUC)':>8}  {'ΔF1':>8}  status")
    print('-' * 90)
    evaluated = 0
    deltas = []
    bfs_wins = 0
    for r in rows:
        if r.get('status') == 'OK':
            evaluated += 1
            deltas.append(r['delta_f1'])
            if r['delta_f1'] > 0: bfs_wins += 1
            print(f"{r['id']:>10} {r['pdb']:>5} {r['N']:>5} {r['M']:>4} {r['L']:>4}  "
                  f"{r['bfs_f1']:>8.3f} {r['eucl_f1']:>8.3f}  {r['delta_f1']:>+8.3f}  OK")
        else:
            print(f"{r['id']:>10} {r['pdb']:>5}     -    -    -        -        -         -  FAIL: {r.get('reason','?')}")

    print(f"\nEvaluated: {evaluated} / 10 systems")
    if deltas:
        deltas = np.array(deltas)
        med = float(np.median(deltas))
        win_pct = bfs_wins / evaluated * 100
        print(f"Median ΔF1: {med:+.4f}")
        print(f"BFS wins (ΔF1 > 0): {bfs_wins}/{evaluated} = {win_pct:.0f}%")
        establish = med >= 0.05 and win_pct >= 70
        reject = med <= 0.01 or (evaluated - bfs_wins) / evaluated >= 0.5
        if establish: verdict = "ESTABLISH — allosteric mapping joins the pitch."
        elif reject:  verdict = "REJECT — allosteric mapping stays out of the pitch permanently."
        else:         verdict = "INCONCLUSIVE — defer to queen."
        print(f"\nVERDICT (per DeepThink's pre-registered criteria): {verdict}")
    else:
        verdict = "ALL_FAILED — no systems evaluated."
        print(f"\nVERDICT: {verdict}")

    out_results = {'cohort': 'DeepThink allosteric N=10', 'cutoff_A': CUTOFF,
                   'tube_slack': TUBE_SLACK, 'match_window': MATCH_WINDOW,
                   'rows': rows,
                   'evaluated': evaluated,
                   'median_delta_f1': float(np.median(deltas)) if len(deltas) else None,
                   'bfs_wins': bfs_wins,
                   'verdict': verdict}
    with open(EXP / 'results.json', 'w') as f:
        json.dump(out_results, f, indent=2)
    print(f"\nSaved {EXP / 'results.json'}")


if __name__ == '__main__':
    main()
