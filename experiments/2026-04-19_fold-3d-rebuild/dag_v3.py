#!/usr/bin/env python3
"""
Match 4V9D ↔ 4V9C with biological grounding.

Strategy:
  1. 23S rRNA is the largest molecule in both (entity 25 in 4V9D, entity 24 in 4V9C, ~2900 nt).
     Each structure has 2 copies (2 ribosomes per asymmetric unit).
  2. For each (4V9D 23S subchain) × (4V9C 23S subchain) pair, build a sequence-position
     map (P atoms keyed by seq_id, take intersection), Kabsch-align, score by RMSD.
  3. The two best disjoint pairs define the biological-copy mapping.
  4. For one copy, apply the 23S Kabsch transform to ALL backbone atoms in 4V9C and
     compute per-atom displacement vs the matched 4V9D atom.
  5. Match non-23S atoms by entity-sequence identity: walk every polymer entity in 4V9D,
     find an entity in 4V9C with the same length and same first 80 residues, then map
     by (matched_entity_pair, seq_id, atom). Use spatial proximity to the aligned 23S
     to pick which subchain belongs to which copy when an entity has 2 subchains.

This avoids the subchain-label collision (subchain "YA" being two different proteins).
"""
import gemmi, numpy as np, json
from collections import defaultdict, Counter

D_PATH = '/tmp/validate_isomorphic/4V9D.cif'
C_PATH = '/tmp/validate_isomorphic/4V9C.cif'

def load(path):
    return gemmi.read_structure(path)

def entity_seq(ent):
    """Return string of one-letter codes for the entity sequence."""
    out = []
    for r in ent.full_sequence:
        info = gemmi.find_tabulated_residue(r)
        out.append(info.one_letter_code.upper() if info and info.one_letter_code else 'X')
    return ''.join(out)

def get_polymer_entities(s):
    return [e for e in s.entities if e.entity_type == gemmi.EntityType.Polymer]

def collect_backbone_by_subchain(s):
    """Map subchain -> list of (seq_id, atom_name, x, y, z, residue_name)."""
    out = defaultdict(list)
    for chain in s[0]:
        for res in chain:
            sc = res.subchain or chain.name
            for atom in res:
                if atom.name in ('CA', 'P', "C1'"):
                    out[sc].append((res.seqid.num, atom.name, atom.pos.x, atom.pos.y, atom.pos.z, res.name))
                    break
    return out

def kabsch(P, Q):
    """Align Q onto P. Returns R, t such that R @ Q + t ≈ P."""
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Qc.T @ Pc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = P.mean(0) - R @ Q.mean(0)
    return R, t

def pair_score(d_atoms, c_atoms):
    """Both are lists of (seq_id, atom, x, y, z, res). Match by seq_id+atom, Kabsch."""
    dk = {(a[0], a[1]): a for a in d_atoms}
    ck = {(a[0], a[1]): a for a in c_atoms}
    shared = sorted(set(dk) & set(ck))
    if len(shared) < 100:
        return None, len(shared), 999.0
    Pd = np.array([[dk[k][2], dk[k][3], dk[k][4]] for k in shared])
    Pc = np.array([[ck[k][2], ck[k][3], ck[k][4]] for k in shared])
    R, t = kabsch(Pd, Pc)
    Pc_al = (R @ Pc.T).T + t
    rmsd = np.sqrt(np.mean(np.sum((Pd - Pc_al)**2, axis=1)))
    return (R, t, shared, Pd, Pc_al, rmsd), len(shared), rmsd

def match_polymer_entities(d_ents, c_ents):
    """Pair entities by (length, first-80 residues)."""
    pairs = []
    used_c = set()
    d_idx = {e.name: e for e in d_ents}
    c_idx = {e.name: e for e in c_ents}
    # group by length
    c_by_len = defaultdict(list)
    for e in c_ents: c_by_len[len(e.full_sequence)].append(e)
    for de in d_ents:
        L = len(de.full_sequence)
        d_seq = entity_seq(de)[:80]
        candidates = [ce for ce in c_by_len.get(L, []) if ce.name not in used_c]
        # also try L-1, L+1
        for delta in (-1, 1, -2, 2):
            candidates += [ce for ce in c_by_len.get(L + delta, []) if ce.name not in used_c]
        best = None; best_dist = 999
        for ce in candidates:
            c_seq = entity_seq(ce)[:80]
            n = min(len(d_seq), len(c_seq))
            if n == 0: continue
            mismatches = sum(1 for i in range(n) if d_seq[i] != c_seq[i])
            if mismatches < best_dist:
                best_dist = mismatches; best = ce
        if best is not None and best_dist <= max(8, len(d_seq) // 5):
            pairs.append((de, best, best_dist))
            used_c.add(best.name)
    return pairs

def main():
    sd = load(D_PATH); sc = load(C_PATH)
    print(f"Loaded 4V9D ({len(sd[0])} chains) and 4V9C ({len(sc[0])} chains)")

    bd = collect_backbone_by_subchain(sd)
    bc = collect_backbone_by_subchain(sc)
    print(f"Subchains: 4V9D={len(bd)}  4V9C={len(bc)}")

    # 1. Find 23S rRNA in both: longest polymer
    d_polys = get_polymer_entities(sd); c_polys = get_polymer_entities(sc)
    d_23s = max(d_polys, key=lambda e: len(e.full_sequence))
    c_23s = max(c_polys, key=lambda e: len(e.full_sequence))
    print(f"23S in 4V9D: entity {d_23s.name}, len {len(d_23s.full_sequence)}, subchains {d_23s.subchains}")
    print(f"23S in 4V9C: entity {c_23s.name}, len {len(c_23s.full_sequence)}, subchains {c_23s.subchains}")

    # 2. All-pairs Kabsch on 23S subchains
    print("\n23S all-pairs alignment:")
    rmsd_grid = {}
    for ds in d_23s.subchains:
        for cs in c_23s.subchains:
            r, n, rmsd = pair_score(bd[ds], bc[cs])
            rmsd_grid[(ds, cs)] = (r, n, rmsd)
            print(f"  4V9D[{ds}] ↔ 4V9C[{cs}]: shared={n} RMSD={rmsd:.2f}")

    # 3. Pick disjoint best pairing (Hungarian-lite for 2x2)
    ds_list = d_23s.subchains; cs_list = c_23s.subchains
    if len(ds_list) == 2 and len(cs_list) == 2:
        # two assignments: (ds0,cs0)+(ds1,cs1) or (ds0,cs1)+(ds1,cs0)
        a = rmsd_grid[(ds_list[0], cs_list[0])][2] + rmsd_grid[(ds_list[1], cs_list[1])][2]
        b = rmsd_grid[(ds_list[0], cs_list[1])][2] + rmsd_grid[(ds_list[1], cs_list[0])][2]
        if a < b:
            assignment = [(ds_list[0], cs_list[0]), (ds_list[1], cs_list[1])]
        else:
            assignment = [(ds_list[0], cs_list[1]), (ds_list[1], cs_list[0])]
    else:
        # fallback: greedy
        assignment = []; used_c = set()
        for ds in ds_list:
            best_cs = None; best_r = 999
            for cs in cs_list:
                if cs in used_c: continue
                if rmsd_grid[(ds, cs)][2] < best_r:
                    best_r = rmsd_grid[(ds, cs)][2]; best_cs = cs
            if best_cs:
                assignment.append((ds, best_cs)); used_c.add(best_cs)

    print(f"\nBiological-copy mapping (by 23S):")
    for ds, cs in assignment:
        print(f"  copy: 4V9D[{ds}] ↔ 4V9C[{cs}] (23S RMSD = {rmsd_grid[(ds,cs)][2]:.2f} Å)")

    # 4. Pick the better-aligned copy as the working ribosome
    assignment.sort(key=lambda p: rmsd_grid[p][2])
    work_ds, work_cs = assignment[0]
    work = rmsd_grid[(work_ds, work_cs)][0]
    R, t, shared_23s, _, _, rmsd_23s = work
    print(f"\nWorking copy: 4V9D[{work_ds}] ↔ 4V9C[{work_cs}], 23S anchor RMSD={rmsd_23s:.2f} Å on {len(shared_23s)} P atoms")

    # 5. Match all polymer entities by sequence
    pairs = match_polymer_entities(d_polys, c_polys)
    print(f"\nMatched {len(pairs)} polymer entities by length+sequence.")

    # 6. For each entity pair, decide which subchain in 4V9D belongs to the working copy.
    #    Heuristic: subchain whose residues' centroid is closest to the working 23S centroid.
    # Compute working-copy 23S centroid (in 4V9D coordinates).
    Pd_23s = np.array([[a[2], a[3], a[4]] for a in bd[work_ds]])
    d_centroid = Pd_23s.mean(0)

    # And in 4V9C coordinates (the matched copy)
    Pc_23s = np.array([[a[2], a[3], a[4]] for a in bc[work_cs]])
    c_centroid = Pc_23s.mean(0)

    # Map subchain -> centroid for each structure
    def subchain_centroids(struct_atoms):
        out = {}
        for sc, lst in struct_atoms.items():
            if not lst: continue
            arr = np.array([[a[2], a[3], a[4]] for a in lst])
            out[sc] = arr.mean(0)
        return out
    d_cents = subchain_centroids(bd)
    c_cents = subchain_centroids(bc)

    # 7. Walk pairs, assign to working copy by centroid proximity, compute displacements.
    all_disp = []  # (entity_d, entity_c, residue_name, seq_id, atom, displacement, x_d, y_d, z_d)
    for de, ce, mm in pairs:
        # Filter subchains of each entity to those present in our backbone collection
        ds_subs = [s for s in de.subchains if s in bd and bd[s]]
        cs_subs = [s for s in ce.subchains if s in bc and bc[s]]
        if not ds_subs or not cs_subs: continue
        # Pick the d subchain closest to working d-centroid
        d_sub = min(ds_subs, key=lambda s: np.linalg.norm(d_cents[s] - d_centroid))
        c_sub = min(cs_subs, key=lambda s: np.linalg.norm(c_cents[s] - c_centroid))
        # Build seq_id+atom -> coords map for the chosen subchain pair
        dk = {(a[0], a[1]): a for a in bd[d_sub]}
        ck = {(a[0], a[1]): a for a in bc[c_sub]}
        shared = sorted(set(dk) & set(ck))
        for k in shared:
            d_atom = dk[k]; c_atom = ck[k]
            d_xyz = np.array([d_atom[2], d_atom[3], d_atom[4]])
            c_xyz = np.array([c_atom[2], c_atom[3], c_atom[4]])
            c_aligned = R @ c_xyz + t
            disp = np.linalg.norm(d_xyz - c_aligned)
            all_disp.append((de.name, ce.name, d_atom[5], k[0], k[1], disp, d_sub, c_sub, *d_xyz))

    print(f"\nTotal matched residues across all entities: {len(all_disp)}")
    if not all_disp:
        print("No matches found."); return

    disps = np.array([row[5] for row in all_disp])
    print(f"Displacement: mean={disps.mean():.2f}  median={np.median(disps):.2f}  max={disps.max():.2f}  p90={np.percentile(disps, 90):.2f}  p99={np.percentile(disps, 99):.2f}")

    # Top-N movers
    order = np.argsort(-disps)
    print(f"\nTop 30 moving residues:")
    print(f"  {'ent_D':>5} {'ent_C':>5} {'sub_D':>5} {'sub_C':>5} {'res':>4} {'seq':>5} {'atom':>4}  {'disp_Å':>8}")
    for i in order[:30]:
        row = all_disp[i]
        print(f"  {row[0]:>5} {row[1]:>5} {row[6]:>5} {row[7]:>5} {row[2]:>4} {row[3]:>5} {row[4]:>4}  {row[5]:8.2f}")

    # Distribution by entity (which entities house the top 5%?)
    cutoff = np.percentile(disps, 95)
    top_idx = [i for i, d in enumerate(disps) if d >= cutoff]
    by_ent = Counter((all_disp[i][0], all_disp[i][1]) for i in top_idx)
    print(f"\nTop 5% movers (>= {cutoff:.2f} Å) by entity:")
    for (ed, ec), cnt in by_ent.most_common(15):
        # Lookup sequence-length for context
        de_obj = next(e for e in d_polys if e.name == ed)
        print(f"  ent_D={ed} (len {len(de_obj.full_sequence)}, type {de_obj.polymer_type})  ent_C={ec}  count={cnt}")

    # Save full results
    out = {
        'best_copy': {'d_subchain': work_ds, 'c_subchain': work_cs, 'rmsd_23s': float(rmsd_23s), 'n_anchor': len(shared_23s)},
        'all_assignments': [{'d': ds, 'c': cs, 'rmsd': float(rmsd_grid[(ds,cs)][2])} for ds, cs in assignment],
        'matched_entities': len(pairs),
        'matched_residues': len(all_disp),
        'displacement_stats': {
            'mean': float(disps.mean()), 'median': float(np.median(disps)),
            'max': float(disps.max()), 'p90': float(np.percentile(disps, 90)),
            'p95': float(np.percentile(disps, 95)), 'p99': float(np.percentile(disps, 99)),
        },
        'top_30': [{'ent_d': r[0], 'ent_c': r[1], 'sub_d': r[6], 'sub_c': r[7],
                    'res': r[2], 'seq': r[3], 'atom': r[4], 'disp': float(r[5])}
                   for i in order[:30] for r in [all_disp[i]]],
    }
    with open('/tmp/validate_isomorphic/dag_v3_result.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved /tmp/validate_isomorphic/dag_v3_result.json")

if __name__ == '__main__':
    main()
