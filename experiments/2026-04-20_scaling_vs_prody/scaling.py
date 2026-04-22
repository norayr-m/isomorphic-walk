#!/usr/bin/env python3
"""
Bench C v2 — Asymptotic Scaling vs ProDy.

v2 changes:
  - cKDTree for KNN (so HIV-1 capsid at 200k+ atoms is tractable)
  - ProDy run only on systems where it's expected to complete (≤ ~20k atoms)
  - For 3J3Q: only Swift + SciPy BFS, ProDy reported as "predicted intractable"
  - Per-system wall-time cap so nothing hangs
"""
import os, sys, time, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'engine' / 'python'))
import bfslib

import gemmi
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order
from scipy.spatial import cKDTree
import prody
prody.confProDy(verbosity='warning')

SCRATCH = Path(os.environ.get('PROTEIN_DIR', 'data'))
RESULTS = HERE / 'results.json'

K_NN = 12

def extract_CA_P(cif_path: str):
    s = gemmi.read_structure(cif_path)
    coords = []
    for chain in s[0]:
        for res in chain:
            for atom in res:
                if atom.name in ('CA', 'P', "C1'"):
                    coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
                    break
    return np.array(coords, dtype=np.float64)

def build_knn_csr_kdtree(coords: np.ndarray, k: int = K_NN):
    """KNN via cKDTree — O(N log N), handles 100k+ atoms."""
    n = coords.shape[0]
    tree = cKDTree(coords)
    # query k+1 to skip self
    _, nn = tree.query(coords, k=k+1)
    edges = set()
    for i in range(n):
        for j in nn[i, 1:]:  # skip self
            jj = int(j)
            if i == jj: continue
            a, b = (i, jj) if i < jj else (jj, i)
            edges.add((a, b))
    rows = [[] for _ in range(n)]
    for a, b in edges:
        rows[a].append(b); rows[b].append(a)
    for r in rows: r.sort()
    off = [0]; idx = []
    for r in rows:
        idx.extend(r); off.append(len(idx))
    return np.array(off, dtype=np.int32), np.array(idx, dtype=np.int32), n

def time_swift_bfs(offsets, indices, seed, runs=5):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        bfslib.bfs_traverse(offsets, indices, seed)
        times.append(time.perf_counter_ns() - t0)
    return min(times) / 1000.0  # µs

def time_scipy_bfs(scsr, seed, runs=5):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        breadth_first_order(scsr, seed, return_predecessors=False)
        times.append(time.perf_counter_ns() - t0)
    return min(times) / 1000.0

def time_prody_gnm(cif_path):
    """Single run, return ms (or None on OOM/error)."""
    try:
        t0 = time.perf_counter()
        atoms = prody.parseMMCIF(cif_path)
        ca = atoms.select('protein and name CA') if atoms else None
        if ca is None:
            ca = atoms.select('name CA P "C1\'"') if atoms else None
        if ca is None or ca.numAtoms() < 50:
            return None, 0
        n = ca.numAtoms()
        g = prody.GNM('benchmark')
        g.buildKirchhoff(ca, cutoff=10.0)
        g.calcModes(n_modes=20, zeros=False)
        return (time.perf_counter() - t0) * 1000, n
    except (MemoryError, ValueError, Exception) as e:
        return None, str(e)

def main():
    pairs = [
        ('AK',           SCRATCH / '4AKE.cif',  True),
        ('MBP',          SCRATCH / '1OMP.cif',  True),
        ('Abl',          SCRATCH / '2HZ4.cif',  True),
        ('ribosome',     SCRATCH / '4V9D.cif',  True),
        ('HIV-1 capsid', SCRATCH / '3J3Q.cif',  False),  # ProDy skipped, predicted intractable
    ]
    rows = []
    print(f"{'system':>16}  {'n':>8}  {'edges':>10}  {'KNN_s':>7}  {'Swift_µs':>10} {'SciPy_µs':>10} {'ProDy_ms':>14}")
    print('-' * 100)
    for name, cif_path, run_prody in pairs:
        if not cif_path.exists():
            print(f"{name:>16}: skipped — {cif_path} not present"); continue
        try:
            t_parse = time.perf_counter()
            coords = extract_CA_P(str(cif_path))
            n = len(coords)
            print(f"{name:>16}: parsed n={n} CA/P/C1' in {time.perf_counter()-t_parse:.1f}s, building KNN…", flush=True)

            t_knn = time.perf_counter()
            offsets, indices, _ = build_knn_csr_kdtree(coords, K_NN)
            knn_s = time.perf_counter() - t_knn

            data = np.ones(len(indices), dtype=np.int32)
            scsr = sp.csr_matrix((data, indices, offsets), shape=(n, n))

            sw = time_swift_bfs(offsets, indices, seed=0, runs=5)
            sc = time_scipy_bfs(scsr, seed=0, runs=5)

            if run_prody:
                pr_ms, pr_n = time_prody_gnm(str(cif_path))
                pr_str = f"{pr_ms:>9.1f} ms" if pr_ms is not None else f"FAIL: {str(pr_n)[:30]}"
            else:
                pr_ms, pr_n = None, "skipped (predicted intractable)"
                pr_str = "skipped"

            print(f"{name:>16}  {n:>8}  {len(indices):>10}  {knn_s:>6.1f}s  {sw:>9.1f}µ {sc:>9.1f}µ  {pr_str:>14}", flush=True)
            rows.append({
                'system': name, 'n': int(n), 'edges': int(len(indices)),
                'knn_build_s': float(knn_s),
                'swift_us_best': float(sw),
                'scipy_us_best': float(sc),
                'prody_gnm_ms': None if pr_ms is None else float(pr_ms),
                'prody_status': 'ok' if pr_ms is not None else (pr_n if isinstance(pr_n, str) else f'fail: {pr_n}'),
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append({'system': name, 'error': str(e)})

    with open(RESULTS, 'w') as f:
        json.dump({'rows': rows, 'k_nn': K_NN}, f, indent=2)
    print(f"\nSaved {RESULTS}")

if __name__ == '__main__':
    main()
