#!/usr/bin/env python3
"""
bfslib quickstart.

Build a small random graph, BFS from node 0, inspect the frame array.
"""
import sys
from pathlib import Path

# Ensure the wrapper is importable. In a real install this would be
# pip-installed or PYTHONPATH'd; for the in-repo example we just point at it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'engine' / 'python'))
import bfslib

import numpy as np
import scipy.sparse as sp

print(bfslib.version())
print()

# --- Build a tiny graph ---
# Path graph: 0 — 1 — 2 — 3 — 4
offsets = np.array([0, 1, 3, 5, 7, 8], dtype=np.int32)
indices = np.array([1, 0, 2, 1, 3, 2, 4, 3], dtype=np.int32)
print("Graph: 0 — 1 — 2 — 3 — 4")
frames = bfslib.bfs_traverse(offsets, indices, seed=0)
print(f"  BFS frames from node 0: {frames.tolist()}")
assert frames.tolist() == [0, 1, 2, 3, 4], "path graph BFS failed"

# --- From a scipy CSR matrix ---
print("\nRandom Erdős–Rényi (n=1000, p=0.012):")
rng = np.random.default_rng(0)
G = sp.random(1000, 1000, density=0.012, format='csr', random_state=rng) > 0
G = G + G.T          # symmetrise
G = G.astype(bool).astype(np.int32)
G = G.tocsr()
offsets = G.indptr.astype(np.int32)
indices = G.indices.astype(np.int32)
print(f"  n={G.shape[0]}, edges (directed)={G.nnz}, mean degree={G.nnz / G.shape[0]:.1f}")

import time
t0 = time.perf_counter_ns()
frames = bfslib.bfs_traverse(offsets, indices, seed=0)
t1 = time.perf_counter_ns()
reached = int((frames >= 0).sum())
max_frame = int(frames.max())
print(f"  reached {reached}/{G.shape[0]} nodes in {max_frame} frames")
print(f"  wall: {(t1-t0)/1000:.1f} µs")
