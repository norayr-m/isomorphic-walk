#!/usr/bin/env python3
"""
Bench B — GAPBS-style standardized graph BFS.

Read a SuiteSparse / SNAP graph (Matrix Market format), build CSR,
run BFS in Swift and SciPy. Report TEPS (Traversed Edges Per Second).

Per DeepThink: 'Swift on Apple Silicon sustaining >200 MTEPS on a
standard GAPBS graph = undeniable infrastructure credibility.'
"""
import sys, time, json
from pathlib import Path
import numpy as np
import scipy.io as spio
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order

import os
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'engine' / 'python'))
import bfslib

SCRATCH = Path(os.environ.get('PROTEIN_DIR', 'data'))
RESULTS = HERE / 'results.json'

def load_mtx_to_csr(mtx_path):
    """Read Matrix Market, convert to symmetric int32 CSR."""
    print(f"Reading {mtx_path}…", flush=True)
    M = spio.mmread(str(mtx_path)).tocsr()
    print(f"  shape={M.shape}, nnz={M.nnz}")
    # Symmetrize: A + A.T (logical OR)
    Ms = M + M.T
    Ms.data[:] = 1
    Ms = Ms.tocsr()
    n = Ms.shape[0]
    offsets = Ms.indptr.astype(np.int32)
    indices = Ms.indices.astype(np.int32)
    return offsets, indices, n

def load_snap_to_csr(snap_path):
    """Read SNAP edge list (gzipped, lines: '<from>\t<to>', # comments)."""
    import gzip
    print(f"Reading SNAP edge list {snap_path}…", flush=True)
    src = []; dst = []
    with gzip.open(str(snap_path), 'rt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            a, b = line.split()
            src.append(int(a)); dst.append(int(b))
    src = np.array(src, dtype=np.int32); dst = np.array(dst, dtype=np.int32)
    n = int(max(src.max(), dst.max()) + 1)
    print(f"  edges (one direction in source): {len(src)}, max node id: {n-1}")
    # Symmetrize via scipy.sparse
    data = np.ones(len(src), dtype=np.int32)
    M = sp.coo_matrix((data, (src, dst)), shape=(n, n)).tocsr()
    Ms = M + M.T
    Ms.data[:] = 1
    Ms = Ms.tocsr()
    offsets = Ms.indptr.astype(np.int32)
    indices = Ms.indices.astype(np.int32)
    return offsets, indices, n

def time_swift(offsets, indices, seed, runs=5):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fo = bfslib.bfs_traverse(offsets, indices, seed)
        times.append(time.perf_counter_ns() - t0)
    reached = (fo >= 0).sum()
    return min(times), reached

def time_scipy(scsr, seed, runs=5):
    times = []
    reached = 0
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        order = breadth_first_order(scsr, seed, return_predecessors=False)
        times.append(time.perf_counter_ns() - t0)
        reached = len(order)
    return min(times), reached

def main():
    snap_path = SCRATCH / 'roadNet-CA.txt.gz'
    mtx_candidates = list(SCRATCH.glob('roadNet-CA*/roadNet-CA.mtx'))
    if mtx_candidates:
        offsets, indices, n = load_mtx_to_csr(mtx_candidates[0])
    elif snap_path.exists():
        offsets, indices, n = load_snap_to_csr(snap_path)
    else:
        print(f"ERROR: no roadNet-CA file under {SCRATCH}", file=sys.stderr); sys.exit(1)
    edges_directed = len(indices)
    edges_undirected = edges_directed // 2
    print(f"Graph: n={n}, edges (directed)={edges_directed}, edges (undirected)={edges_undirected}")

    # Pick a seed with non-trivial degree
    degrees = np.diff(offsets)
    high_deg_seeds = np.where(degrees >= np.percentile(degrees, 99))[0]
    seed = int(high_deg_seeds[0]) if len(high_deg_seeds) else 0
    print(f"Seed: node {seed}, degree {int(degrees[seed])}")

    # Swift
    print("\nRunning Swift BFS (best of 5)…", flush=True)
    sw_ns, sw_reached = time_swift(offsets, indices, seed, runs=5)
    sw_us = sw_ns / 1000.0
    sw_teps = (edges_directed / (sw_ns * 1e-9)) if sw_ns > 0 else 0
    print(f"  best wall: {sw_us:.0f} µs, reached {sw_reached}/{n} nodes")
    print(f"  TEPS: {sw_teps:.2e} = {sw_teps/1e6:.1f} MTEPS")

    # SciPy
    print("\nRunning SciPy BFS (best of 5)…", flush=True)
    data = np.ones(edges_directed, dtype=np.int32)
    scsr = sp.csr_matrix((data, indices, offsets), shape=(n, n))
    sc_ns, sc_reached = time_scipy(scsr, seed, runs=5)
    sc_us = sc_ns / 1000.0
    sc_teps = (edges_directed / (sc_ns * 1e-9)) if sc_ns > 0 else 0
    print(f"  best wall: {sc_us:.0f} µs, reached {sc_reached}/{n} nodes")
    print(f"  TEPS: {sc_teps:.2e} = {sc_teps/1e6:.1f} MTEPS")

    multiplier = sc_us / sw_us if sw_us > 0 else float('inf')
    print(f"\nSwift / SciPy speedup: {multiplier:.1f}×")

    out = {
        'graph': 'roadNet-CA (SuiteSparse SNAP)',
        'n': int(n),
        'edges_directed': int(edges_directed),
        'edges_undirected': int(edges_undirected),
        'seed': int(seed),
        'seed_degree': int(degrees[seed]),
        'swift_us_best': float(sw_us),
        'swift_reached': int(sw_reached),
        'swift_mteps': float(sw_teps / 1e6),
        'scipy_us_best': float(sc_us),
        'scipy_reached': int(sc_reached),
        'scipy_mteps': float(sc_teps / 1e6),
        'swift_over_scipy': float(multiplier),
    }
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {RESULTS}")

    print("\n=== VERDICT ===")
    if sw_teps / 1e6 >= 200:
        print(f"PASSES the 200 MTEPS gate ({sw_teps/1e6:.1f} MTEPS).")
    else:
        print(f"FAILS the 200 MTEPS gate ({sw_teps/1e6:.1f} MTEPS).")

if __name__ == '__main__':
    main()
