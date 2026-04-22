#!/usr/bin/env python3
"""
Build a 4V9D backbone graph from real 3D Å coordinates (NOT 2D-projected).

One biological copy only (the working copy from dag_v3: 23S subchain "VA").
For each polymer entity that has a subchain in this copy, extract one
representative backbone atom per residue (CA for protein, P for RNA where present
else C1'). KNN k=12 in Å distance. Output: states.bin + offsets.bin + indices.bin
+ residue_index.json that maps node id -> (entity, subchain, seq_id, residue_name,
xyz, biological-displacement-from-dag_v3-if-known).

The displacement field is the bridge to the 4V9D-vs-4V9C analysis: for every
matched residue we know its biological displacement under translocation. This
lets the engine's BFS frame-of-arrival be correlated with biology directly.
"""
import gemmi, numpy as np, json, struct, os
from collections import defaultdict
from pathlib import Path

import os
DATA_DIR = Path(os.environ.get('PROTEIN_DIR', 'data'))
D_PATH = str(DATA_DIR / '4V9D.cif')
DAG_V3 = str(DATA_DIR / 'dag_v3_result.json')
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# Working-copy 23S subchain (from dag_v3 verdict)
WORK_D_23S = 'VA'
K_NEIGHBORS = 12

# E. coli 23S PTC catalytic residues (canonical E. coli numbering): A2451, U2506, U2585, A2602
PTC_CANDIDATES = {2451, 2506, 2585, 2602}

def collect_backbone(path):
    """Return list of (entity_name, subchain, seq_id, atom_name, residue_name, xyz, b_iso)."""
    s = gemmi.read_structure(path)
    # Build subchain -> entity_name map
    sub_to_ent = {}
    for ent in s.entities:
        for sc in ent.subchains:
            sub_to_ent[sc] = ent.name
    out = []
    for chain in s[0]:
        for res in chain:
            sc = res.subchain or chain.name
            ent_name = sub_to_ent.get(sc, '?')
            for atom in res:
                if atom.name in ('CA', 'P', "C1'"):
                    out.append((ent_name, sc, res.seqid.num, atom.name, res.name,
                                (atom.pos.x, atom.pos.y, atom.pos.z), atom.b_iso))
                    break
    return out, s

def main():
    print("Loading 4V9D...")
    atoms, s = collect_backbone(D_PATH)
    print(f"  {len(atoms)} backbone atoms across {len(set(a[1] for a in atoms))} subchains")

    # Need to know which subchains belong to the working copy.
    # Strategy from dag_v3: subchain centroid closest to working 23S centroid.
    work_atoms_23s = [a for a in atoms if a[1] == WORK_D_23S]
    if not work_atoms_23s:
        raise SystemExit(f"No atoms for working 23S subchain {WORK_D_23S}")
    work_centroid = np.array([a[5] for a in work_atoms_23s]).mean(0)
    print(f"  Working 23S subchain {WORK_D_23S}: {len(work_atoms_23s)} atoms, centroid {work_centroid}")

    # For each entity, take whichever subchain has its centroid closer to work_centroid
    by_ent = defaultdict(lambda: defaultdict(list))
    for a in atoms:
        by_ent[a[0]][a[1]].append(a)
    chosen = []
    for ent_name, subs in by_ent.items():
        if len(subs) == 1:
            chosen.extend(next(iter(subs.values())))
        else:
            best_sub = None; best_d = float('inf')
            for sc, lst in subs.items():
                c = np.array([a[5] for a in lst]).mean(0)
                d = np.linalg.norm(c - work_centroid)
                if d < best_d:
                    best_d = d; best_sub = sc
            chosen.extend(subs[best_sub])
    print(f"  Working copy: {len(chosen)} atoms after centroid selection")

    # Sort to a canonical order: by entity then seq_id
    chosen.sort(key=lambda a: (a[0], a[1], a[2]))

    n = len(chosen)
    coords = np.array([a[5] for a in chosen], dtype=np.float64)

    # KNN k=12 in 3D Å
    print(f"Computing KNN k={K_NEIGHBORS} on {n} atoms...")
    # Pairwise distance is feasible: n ~ 5000, n^2 = 25M floats, 200MB. OK on M5.
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))
    np.fill_diagonal(dist, np.inf)
    nn_idx = np.argpartition(dist, K_NEIGHBORS, axis=1)[:, :K_NEIGHBORS]
    # Sort each row's neighbors by distance for reproducibility
    for i in range(n):
        nn_idx[i] = nn_idx[i][np.argsort(dist[i, nn_idx[i]])]

    # Build CSR (undirected: include reverse edges, deduped)
    edges_set = set()
    for i in range(n):
        for j in nn_idx[i]:
            if i == j: continue
            a_, b_ = (i, j) if i < j else (j, i)
            edges_set.add((a_, b_))
    print(f"  {len(edges_set)} unique undirected edges")

    # Directed CSR
    csr_rows = [[] for _ in range(n)]
    for i, j in edges_set:
        csr_rows[i].append(j); csr_rows[j].append(i)
    for r in csr_rows: r.sort()

    offsets = [0]
    indices = []
    for r in csr_rows:
        indices.extend(r); offsets.append(len(indices))

    print(f"  Directed CSR: {n} nodes, {len(indices)} entries (avg degree {len(indices)/n:.1f})")

    # Initial states (ternary): pick a PTC seed.
    # Find a 23S residue with seq_id in PTC_CANDIDATES, prefer A2451 if present.
    states = np.zeros(n, dtype=np.int8)
    seed_idx = None
    for i, a in enumerate(chosen):
        if a[1] == WORK_D_23S and a[2] == 2451:
            seed_idx = i; break
    if seed_idx is None:
        for i, a in enumerate(chosen):
            if a[1] == WORK_D_23S and a[2] in PTC_CANDIDATES:
                seed_idx = i; break
    if seed_idx is None:
        # Fallback: pick the 23S residue closest to overall 23S centroid (geometric center)
        centro = coords[[i for i, a in enumerate(chosen) if a[1] == WORK_D_23S]].mean(0)
        ents23 = [(i, np.linalg.norm(coords[i] - centro)) for i, a in enumerate(chosen) if a[1] == WORK_D_23S]
        seed_idx = min(ents23, key=lambda x: x[1])[0]
        print(f"  PTC residues not found; fallback seed = closest-to-centroid 23S: idx {seed_idx} (seq_id {chosen[seed_idx][2]})")
    else:
        print(f"  Seed = PTC residue: idx {seed_idx}, entity {chosen[seed_idx][0]}, seq_id {chosen[seed_idx][2]} ({chosen[seed_idx][4]})")
    states[seed_idx] = 1

    # Write flat binaries
    with open(OUT / 'states.bin', 'wb') as f: f.write(states.tobytes())
    with open(OUT / 'offsets.bin', 'wb') as f: f.write(np.array(offsets, dtype=np.int32).tobytes())
    with open(OUT / 'indices.bin', 'wb') as f: f.write(np.array(indices, dtype=np.int32).tobytes())
    print(f"  Wrote states.bin ({n} bytes), offsets.bin ({(n+1)*4} bytes), indices.bin ({len(indices)*4} bytes)")

    # Load dag_v3 displacement map for residues we have data for
    with open(DAG_V3) as f:
        v3 = json.load(f)
    # dag_v3 stores top-30 only in JSON; need full map. Re-run displacement computation here for
    # all matched entities so the residue_index.json carries biological ground truth.
    # Simpler: store node metadata, leave displacement enrichment to a follow-up script that
    # re-runs the matching against 4V9C with the same selection logic.
    residue_index = []
    for i, a in enumerate(chosen):
        residue_index.append({
            'idx': i, 'entity': a[0], 'subchain': a[1], 'seq_id': a[2],
            'atom': a[3], 'residue': a[4], 'x': a[5][0], 'y': a[5][1], 'z': a[5][2],
            'b_iso': a[6],
        })
    with open(OUT / 'residue_index.json', 'w') as f:
        json.dump({'n_nodes': n, 'k_neighbors': K_NEIGHBORS, 'seed_idx': int(seed_idx),
                   'working_23s_subchain': WORK_D_23S, 'nodes': residue_index}, f, indent=2)
    print(f"  Wrote residue_index.json ({n} nodes)")

    # Quick connectivity check
    print("\nSanity:")
    print(f"  Isolated nodes: {sum(1 for i in range(n) if offsets[i+1] == offsets[i])}")
    print(f"  Min degree: {min(offsets[i+1]-offsets[i] for i in range(n))}")
    print(f"  Max degree: {max(offsets[i+1]-offsets[i] for i in range(n))}")

if __name__ == '__main__':
    main()
