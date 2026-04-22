#!/usr/bin/env python3
"""
Run BFS wavefront on the rebuilt 3D graph, compute per-node biological
displacement against 4V9C, and report correlation between BFS frame-of-arrival
and displacement.

Same BFS algorithm as the Swift KowalskiCrush. Speed isn't the question here;
the question is whether the engine's traversal order has any biological meaning.

Outputs:
  bfs_frames.json     — node_idx -> frame, plus seed
  displacement.json   — node_idx -> Å displacement under translocation, plus stats
  correlation.json    — Spearman + Pearson between frame and displacement
  summary.md          — human-readable verdict
"""
import gemmi, numpy as np, json, struct
from pathlib import Path
from collections import defaultdict, Counter, deque
from scipy.stats import spearmanr, pearsonr

import os
ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('PROTEIN_DIR', 'data'))
D_PATH = str(DATA_DIR / '4V9D.cif')
C_PATH = str(DATA_DIR / '4V9C.cif')

# ---- 1. BFS over the CSR graph ----
def bfs_frames(offsets, indices, n, seed):
    visited = np.zeros(n, dtype=np.uint8)
    frame_of = np.full(n, -1, dtype=np.int32)
    visited[seed] = 1; frame_of[seed] = 0
    cur = [seed]; frame = 0
    while cur:
        nxt = []
        for u in cur:
            for k in range(offsets[u], offsets[u+1]):
                v = int(indices[k])
                if not visited[v]:
                    visited[v] = 1
                    frame_of[v] = frame + 1
                    nxt.append(v)
        cur = nxt; frame += 1
    return frame_of, frame

# ---- 2. Per-node displacement ----
def kabsch(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Qc.T @ Pc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = P.mean(0) - R @ Q.mean(0)
    return R, t

def collect_backbone(path):
    s = gemmi.read_structure(path)
    sub_to_ent = {sc: e.name for e in s.entities for sc in e.subchains}
    out = []
    for chain in s[0]:
        for res in chain:
            sc = res.subchain or chain.name
            ent = sub_to_ent.get(sc, '?')
            for atom in res:
                if atom.name in ('CA', 'P', "C1'"):
                    out.append((ent, sc, res.seqid.num, atom.name, res.name,
                                (atom.pos.x, atom.pos.y, atom.pos.z)))
                    break
    return out, s

def entity_seq(ent):
    out = []
    for r in ent.full_sequence:
        info = gemmi.find_tabulated_residue(r)
        out.append(info.one_letter_code.upper() if info and info.one_letter_code else 'X')
    return ''.join(out)

def match_entities(d_ents, c_ents):
    pairs = []; used_c = set()
    c_by_len = defaultdict(list)
    for e in c_ents: c_by_len[len(e.full_sequence)].append(e)
    for de in d_ents:
        L = len(de.full_sequence); d_seq = entity_seq(de)[:80]
        cands = []
        for delta in (0, -1, 1, -2, 2):
            cands += [ce for ce in c_by_len.get(L + delta, []) if ce.name not in used_c]
        best = None; best_dist = 999
        for ce in cands:
            c_seq = entity_seq(ce)[:80]
            n = min(len(d_seq), len(c_seq))
            if n == 0: continue
            mismatches = sum(1 for i in range(n) if d_seq[i] != c_seq[i])
            if mismatches < best_dist:
                best_dist = mismatches; best = ce
        if best is not None and best_dist <= max(8, len(d_seq) // 5):
            pairs.append((de, best, best_dist)); used_c.add(best.name)
    return pairs

def main():
    # --- load graph ---
    print("Loading rebuilt 3D graph...")
    with open(ROOT / 'residue_index.json') as f:
        ri = json.load(f)
    n = ri['n_nodes']; seed = ri['seed_idx']
    offsets = np.frombuffer(open(ROOT / 'offsets.bin', 'rb').read(), dtype=np.int32)
    indices = np.frombuffer(open(ROOT / 'indices.bin', 'rb').read(), dtype=np.int32)
    print(f"  {n} nodes, {len(indices)} CSR entries, seed = node {seed} (entity {ri['nodes'][seed]['entity']}, seq {ri['nodes'][seed]['seq_id']})")

    # --- BFS ---
    print("Running BFS wavefront...")
    frame_of, total_frames = bfs_frames(offsets, indices, n, seed)
    reached = int((frame_of >= 0).sum())
    print(f"  {reached}/{n} reached in {total_frames} frames")
    with open(ROOT / 'bfs_frames.json', 'w') as f:
        json.dump({'seed': int(seed), 'total_frames': int(total_frames),
                   'reached': reached,
                   'frame_of': [int(x) for x in frame_of]}, f)

    # --- biological displacement under translocation ---
    print("\nComputing per-node displacement vs 4V9C...")
    sd, _ = collect_backbone(D_PATH); sc_atoms, sc_struct = collect_backbone(C_PATH)
    # Build dict for 4V9C lookup: keyed by (entity_in_C, seq_id, atom)
    sd_struct_obj = gemmi.read_structure(D_PATH)
    sc_struct_obj = gemmi.read_structure(C_PATH)
    d_polys = [e for e in sd_struct_obj.entities if e.entity_type == gemmi.EntityType.Polymer]
    c_polys = [e for e in sc_struct_obj.entities if e.entity_type == gemmi.EntityType.Polymer]
    pairs = match_entities(d_polys, c_polys)
    ent_d_to_c = {de.name: ce.name for de, ce, _ in pairs}
    print(f"  Matched {len(pairs)} polymer entities")

    # Build (entity_C, seq_id, atom) -> xyz lookup for 4V9C, choosing the subchain whose
    # centroid is closest to working 23S in 4V9C.
    work_d_23s_subchain = ri['working_23s_subchain']  # 'VA' for 4V9D
    # Find which 4V9C 23S subchain it pairs with
    d_23s = max(d_polys, key=lambda e: len(e.full_sequence))
    c_23s = max(c_polys, key=lambda e: len(e.full_sequence))
    # All-pairs Kabsch on 23S subchains
    bd = defaultdict(list); bc = defaultdict(list)
    for a in sd: bd[a[1]].append(a)
    for a in sc_atoms: bc[a[1]].append(a)
    def pair_rmsd(d_atoms, c_atoms):
        dk = {(a[2], a[3]): a for a in d_atoms}
        ck = {(a[2], a[3]): a for a in c_atoms}
        shared = sorted(set(dk) & set(ck))
        if len(shared) < 100: return 999, None, None, None
        Pd = np.array([dk[k][5] for k in shared])
        Pc = np.array([ck[k][5] for k in shared])
        R, t = kabsch(Pd, Pc)
        Pc_al = (R @ Pc.T).T + t
        return float(np.sqrt(np.mean(np.sum((Pd-Pc_al)**2, 1)))), R, t, shared
    rmsd_grid = {}
    for ds in d_23s.subchains:
        for cs in c_23s.subchains:
            rmsd, R, t, shared = pair_rmsd(bd[ds], bc[cs])
            rmsd_grid[(ds, cs)] = (rmsd, R, t, shared)
    # Pick disjoint best pair, find work_d_23s_subchain partner
    ds_l = list(d_23s.subchains); cs_l = list(c_23s.subchains)
    if len(ds_l) == 2 and len(cs_l) == 2:
        a = rmsd_grid[(ds_l[0], cs_l[0])][0] + rmsd_grid[(ds_l[1], cs_l[1])][0]
        b = rmsd_grid[(ds_l[0], cs_l[1])][0] + rmsd_grid[(ds_l[1], cs_l[0])][0]
        assignment = [(ds_l[0], cs_l[0]), (ds_l[1], cs_l[1])] if a < b else [(ds_l[0], cs_l[1]), (ds_l[1], cs_l[0])]
    else:
        assignment = [(ds_l[0], cs_l[0])]
    work_pair = next(p for p in assignment if p[0] == work_d_23s_subchain)
    print(f"  Working 23S pair: 4V9D[{work_pair[0]}] ↔ 4V9C[{work_pair[1]}]  RMSD={rmsd_grid[work_pair][0]:.2f} Å")
    R, t = rmsd_grid[work_pair][1], rmsd_grid[work_pair][2]
    work_c_23s_subchain = work_pair[1]

    # 4V9C centroid for working copy
    Pc_23s = np.array([a[5] for a in bc[work_c_23s_subchain]])
    c_work_centroid = Pc_23s.mean(0)

    # For each entity pair, pick c-subchain with centroid closest to c_work_centroid
    cents_c = {sc: np.array([a[5] for a in lst]).mean(0) for sc, lst in bc.items() if lst}
    def chosen_c_sub(ce):
        subs = [s for s in ce.subchains if s in bc and bc[s]]
        if not subs: return None
        return min(subs, key=lambda s: np.linalg.norm(cents_c[s] - c_work_centroid))

    # Build (entity_d_name, seq_id, atom) -> aligned 4V9C coord
    c_lookup = {}
    for de, ce, _ in pairs:
        cs = chosen_c_sub(ce)
        if cs is None: continue
        for a in bc[cs]:
            xyz = np.array(a[5])
            c_lookup[(de.name, a[2], a[3])] = R @ xyz + t

    # Per-node displacement
    disp = np.full(n, np.nan, dtype=np.float64)
    matched = 0
    for node in ri['nodes']:
        key = (node['entity'], node['seq_id'], node['atom'])
        c_xyz = c_lookup.get(key)
        if c_xyz is None: continue
        d_xyz = np.array([node['x'], node['y'], node['z']])
        disp[node['idx']] = float(np.linalg.norm(d_xyz - c_xyz))
        matched += 1
    print(f"  Per-node displacement: {matched}/{n} nodes have biological displacement data")
    print(f"  disp stats: mean={np.nanmean(disp):.2f} median={np.nanmedian(disp):.2f} max={np.nanmax(disp):.2f}")
    with open(ROOT / 'displacement.json', 'w') as f:
        json.dump({'matched': matched, 'n_nodes': n,
                   'mean': float(np.nanmean(disp)), 'median': float(np.nanmedian(disp)),
                   'max': float(np.nanmax(disp)), 'p90': float(np.nanpercentile(disp, 90)),
                   'p95': float(np.nanpercentile(disp, 95)), 'p99': float(np.nanpercentile(disp, 99)),
                   'disp': [None if np.isnan(d) else float(d) for d in disp]}, f)

    # --- correlation ---
    mask = (~np.isnan(disp)) & (frame_of >= 0)
    f_arr = frame_of[mask].astype(float); d_arr = disp[mask]
    rho, p_rho = spearmanr(f_arr, d_arr)
    r, p_r = pearsonr(f_arr, d_arr)
    print(f"\nCorrelation between BFS frame-of-arrival and biological displacement:")
    print(f"  n = {mask.sum()}")
    print(f"  Spearman ρ = {rho:.4f}  (p = {p_rho:.2e})")
    print(f"  Pearson  r = {r:.4f}  (p = {p_r:.2e})")

    # By-entity slice — does the engine reach RRF / tRNA / S16 / L9 LATER than the bulk?
    # Group nodes by entity, compute median frame and median disp.
    by_ent = defaultdict(list)
    for i, node in enumerate(ri['nodes']):
        if not mask[i]: continue
        by_ent[node['entity']].append((float(frame_of[i]), float(disp[i])))
    print(f"\nPer-entity median (frame, disp):")
    rows = []
    for ent, vs in by_ent.items():
        if len(vs) < 5: continue
        fs = [v[0] for v in vs]; ds = [v[1] for v in vs]
        rows.append((ent, len(vs), float(np.median(fs)), float(np.median(ds)), float(np.mean(ds))))
    rows.sort(key=lambda r: -r[4])
    print(f"  {'ent':>5} {'n':>5} {'med_frame':>10} {'med_disp':>9} {'mean_disp':>10}")
    for ent, nn, mf, md, mn in rows[:15]:
        print(f"  {ent:>5} {nn:>5} {mf:>10.1f} {md:>9.2f} {mn:>10.2f}")

    out_corr = {
        'n_compared': int(mask.sum()),
        'spearman_rho': float(rho), 'spearman_p': float(p_rho),
        'pearson_r': float(r), 'pearson_p': float(p_r),
        'per_entity_rows': [{'entity': ent, 'n': nn, 'median_frame': mf,
                             'median_disp': md, 'mean_disp': mn}
                            for ent, nn, mf, md, mn in rows],
    }
    with open(ROOT / 'correlation.json', 'w') as f:
        json.dump(out_corr, f, indent=2)

    # ---- summary ----
    if abs(rho) >= 0.3:
        verdict = f"SIGNAL: Spearman ρ = {rho:+.3f} indicates BFS frame-of-arrival is correlated with biological displacement under translocation. Sign is {'positive' if rho>0 else 'negative'}: drivers are reached {'late' if rho>0 else 'early'} in the wavefront."
    elif abs(rho) >= 0.15:
        verdict = f"WEAK SIGNAL: Spearman ρ = {rho:+.3f}. Some correlation but not strong; the engine's BFS order has only modest biological meaning."
    else:
        verdict = f"NO SIGNAL: Spearman ρ = {rho:+.3f}. BFS frame-of-arrival is essentially uncorrelated with biological displacement. The engine is fast graph infrastructure; this BFS protocol does not encode the ratchet."
    sm = (
        f"# Engine vs biology — 3D-rebuild experiment\n\n"
        f"Date: 2026-04-19  Agent: Hari Fold Antigyan (4e4d0ac9)\n\n"
        f"## Graph\n- Nodes: {n}\n- Seed: node {seed} (entity 25 / 23S, seq 2451 — PTC catalytic A2451)\n"
        f"- BFS reached: {reached}/{n} in {total_frames} frames\n\n"
        f"## Biological displacement (4V9D → 4V9C, working copy, 23S-anchored)\n"
        f"- Nodes with displacement data: {matched}/{n}\n"
        f"- mean {np.nanmean(disp):.2f} Å, median {np.nanmedian(disp):.2f} Å, p90 {np.nanpercentile(disp,90):.2f}, p99 {np.nanpercentile(disp,99):.2f}, max {np.nanmax(disp):.2f}\n\n"
        f"## Correlation\n- n compared: {int(mask.sum())}\n"
        f"- Spearman ρ = **{rho:+.4f}** (p = {p_rho:.2e})\n"
        f"- Pearson  r = **{r:+.4f}** (p = {p_r:.2e})\n\n"
        f"## Verdict\n{verdict}\n\n"
        f"## Top entities (sorted by mean displacement)\n"
        f"| ent | n | median_frame | median_disp | mean_disp |\n"
        f"|---:|---:|---:|---:|---:|\n"
    )
    for ent, nn, mf, md, mn in rows[:15]:
        sm += f"| {ent} | {nn} | {mf:.1f} | {md:.2f} | {mn:.2f} |\n"
    with open(ROOT / 'summary.md', 'w') as f: f.write(sm)
    print(f"\nWrote summary.md, correlation.json, displacement.json, bfs_frames.json")
    print(f"\n=== VERDICT ===\n{verdict}")

if __name__ == '__main__':
    main()
