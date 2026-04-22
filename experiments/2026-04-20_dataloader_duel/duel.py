#!/usr/bin/env python3
"""
Benchmark A — Dataloader Duel.

Compare wall-clock time for single-source BFS on biomolecular contact
graphs across:
  - Pure Python (baseline floor)
  - SciPy compiled C (scipy.sparse.csgraph.breadth_first_order)
  - Our Swift BFSLib via ctypes
  - Optionally NetworkX (slowest, sanity check)

Per DeepThink: 'A >10× speedup over SciPy's compiled C-backend is a hard
infrastructure claim.' That's the gate.
"""
import os
import sys
import time
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
EXP_ROOT = REPO_ROOT / 'experiments'
DATA_DIR = Path(os.environ.get('PROTEIN_DIR', 'data'))

sys.path.insert(0, str(REPO_ROOT / 'engine' / 'python'))
import bfslib

import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order

def load_csr_from_bin(graph_dir: Path):
    """Load offsets+indices from KowalskiCrush flat-binary CSR."""
    offsets = np.frombuffer((graph_dir / 'offsets.bin').read_bytes(), dtype=np.int32).copy()
    indices = np.frombuffer((graph_dir / 'indices.bin').read_bytes(), dtype=np.int32).copy()
    n = len(offsets) - 1
    return offsets, indices, n

def py_bfs(offsets, indices, n, seed):
    visited = np.zeros(n, dtype=np.uint8)
    fo = np.full(n, -1, dtype=np.int32)
    visited[seed] = 1; fo[seed] = 0
    cur = [seed]; frame = 0
    while cur:
        nxt = []
        for u in cur:
            for k in range(offsets[u], offsets[u+1]):
                v = int(indices[k])
                if not visited[v]:
                    visited[v] = 1; fo[v] = frame + 1
                    nxt.append(v)
        cur = nxt; frame += 1
    return fo

def time_func(fn, runs=10):
    times_us = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        times_us.append((time.perf_counter_ns() - t0) / 1000.0)
    return min(times_us), float(np.median(times_us))

def main():
    # Each graph dir contains states.bin / offsets.bin / indices.bin.
    # Default: ribosome experiment from this repo. Add additional graphs
    # by setting GRAPH_DIRS env var (comma-separated) or editing here.
    graphs = [
        ('ribosome', EXP_ROOT / '2026-04-19_fold-3d-rebuild', 5542),
    ]
    extra = os.environ.get('GRAPH_DIRS', '')
    for entry in (e.strip() for e in extra.split(',') if e.strip()):
        graphs.append((Path(entry).name, Path(entry), 0))
    print(f"# Dataloader Duel — single-source BFS")
    print(f"# Library: {bfslib.version()}")
    print()
    print(f"{'system':>10}  {'n':>6} {'edges':>8}  {'Swift':>10} {'SciPy':>10} {'Python':>10}  {'Sw/SciPy':>9} {'Sw/Py':>8}")
    print("-" * 90)
    rows = []
    for name, gdir, seed in graphs:
        if not (gdir / 'offsets.bin').exists():
            print(f"{name}: skipped (no CSR binary at {gdir})")
            continue
        offsets, indices, n = load_csr_from_bin(gdir)
        # Build scipy CSR (data is just 1s; structure-only)
        data = np.ones(len(indices), dtype=np.int32)
        scsr = sp.csr_matrix((data, indices, offsets), shape=(n, n))

        # Time Swift
        swift_us_min, swift_us_med = time_func(lambda: bfslib.bfs_traverse(offsets, indices, seed))
        # Time SciPy
        scipy_us_min, scipy_us_med = time_func(lambda: breadth_first_order(scsr, seed, return_predecessors=False))
        # Time Python (slow — fewer runs)
        py_us_min, py_us_med = time_func(lambda: py_bfs(offsets, indices, n, seed), runs=3)

        sw_vs_scipy = scipy_us_min / swift_us_min if swift_us_min > 0 else float('inf')
        sw_vs_py = py_us_min / swift_us_min if swift_us_min > 0 else float('inf')
        rows.append({
            'system': name, 'n': int(n), 'edges': int(len(indices)),
            'swift_us_best': float(swift_us_min),
            'scipy_us_best': float(scipy_us_min),
            'python_us_best': float(py_us_min),
            'swift_vs_scipy': float(sw_vs_scipy),
            'swift_vs_python': float(sw_vs_py),
        })
        print(f"{name:>10}  {n:>6} {len(indices):>8}  {swift_us_min:>9.1f}µ {scipy_us_min:>9.1f}µ {py_us_min:>9.1f}µ  {sw_vs_scipy:>8.1f}× {sw_vs_py:>7.1f}×")

    out = HERE / 'results.json'
    with open(out, 'w') as f:
        json.dump({'rows': rows, 'note': 'best of 10 runs (Swift, SciPy); best of 3 (Python).'}, f, indent=2)
    print(f"\nSaved {out}")

    # Verdict
    if rows:
        sw_vs_scipy_med = float(np.median([r['swift_vs_scipy'] for r in rows]))
        print(f"\nMedian Swift / SciPy: {sw_vs_scipy_med:.1f}×")
        if sw_vs_scipy_med >= 10.0:
            print("VERDICT: PASSES the >10× SciPy gate. Hard infrastructure claim.")
        else:
            print("VERDICT: FAILS the >10× SciPy gate. SciPy's C-backend is competitive.")

if __name__ == '__main__':
    main()
