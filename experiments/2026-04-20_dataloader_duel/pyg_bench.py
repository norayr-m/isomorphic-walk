#!/usr/bin/env python3
"""
Dataloader Duel addendum — PyTorch Geometric k_hop_subgraph vs Swift BFS.

Per DeepThink's pivot doc, this is the specific AI-dataloader scenario.
GNN training pipelines extract k-hop neighborhoods around target nodes
many millions of times per epoch. The slowest step is often this CPU
graph traversal, not the GPU forward pass.

Compare wall-clock for k=2, 3, 4 on the rebuilt ribosome graph
(10 739 nodes, 152 946 edges) using:
  - torch_geometric.utils.k_hop_subgraph
  - Swift BFS truncated to depth k (post-filter our frame_of array)
  - SciPy breadth_first_order, post-filter

For PyG we measure with the seed as a single source node.
"""
import sys, time, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
EXP_ROOT = REPO_ROOT / 'experiments'
RESULTS = HERE / 'pyg_results.json'

sys.path.insert(0, str(REPO_ROOT / 'engine' / 'python'))
import bfslib

import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order

import torch
from torch_geometric.utils import k_hop_subgraph

def load_csr_from_bin(graph_dir: Path):
    offsets = np.frombuffer((graph_dir / 'offsets.bin').read_bytes(), dtype=np.int32).copy()
    indices = np.frombuffer((graph_dir / 'indices.bin').read_bytes(), dtype=np.int32).copy()
    n = len(offsets) - 1
    return offsets, indices, n

def csr_to_pyg_edge_index(offsets, indices):
    """Build (2, E) torch.long edge_index from CSR — what PyG expects."""
    src = []; dst = []
    for i in range(len(offsets) - 1):
        for k in range(offsets[i], offsets[i+1]):
            src.append(i); dst.append(int(indices[k]))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index

def time_k(fn, runs=10):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)
    return min(times) / 1000.0  # µs

def main():
    gdir = EXP_ROOT / '2026-04-19_fold-3d-rebuild'
    offsets, indices, n = load_csr_from_bin(gdir)
    seed = 5542  # PTC

    print(f"Graph: n={n}, edges (directed)={len(indices)}, seed={seed}")
    print(f"Building PyG edge_index from CSR…", flush=True)
    t0 = time.perf_counter()
    edge_index = csr_to_pyg_edge_index(offsets, indices)
    print(f"  built in {time.perf_counter()-t0:.1f}s, shape={edge_index.shape}")

    data = np.ones(len(indices), dtype=np.int32)
    scsr = sp.csr_matrix((data, indices, offsets), shape=(n, n))
    seed_t = torch.tensor([seed], dtype=torch.long)

    print(f"\n{'k':>3}  {'PyG':>10} {'Swift_full':>12} {'Swift_clip':>12} {'SciPy':>10}  {'Sw/PyG':>9} {'Sw/SciPy':>10}")
    print('-' * 90)
    rows = []

    for k in (2, 3, 4):
        # PyG k_hop_subgraph: returns (subset, edge_index, mapping, edge_mask)
        pyg_us = time_k(lambda: k_hop_subgraph(seed_t, k, edge_index, relabel_nodes=False), runs=10)

        # Swift: full BFS then post-filter to nodes with frame ≤ k
        def swift_clip():
            fo = bfslib.bfs_traverse(offsets, indices, seed)
            return np.where((fo >= 0) & (fo <= k))[0]
        swift_clip_us = time_k(swift_clip, runs=10)

        # Swift full BFS only (no post-filter cost)
        swift_full_us = time_k(lambda: bfslib.bfs_traverse(offsets, indices, seed), runs=10)

        # SciPy full BFS — there is no native depth-limited BFS in scipy
        scipy_us = time_k(lambda: breadth_first_order(scsr, seed, return_predecessors=False), runs=10)

        sw_vs_pyg = pyg_us / swift_clip_us if swift_clip_us > 0 else float('inf')
        sw_vs_sc  = scipy_us / swift_full_us if swift_full_us > 0 else float('inf')
        rows.append({
            'k': k,
            'pyg_us_best': float(pyg_us),
            'swift_full_us_best': float(swift_full_us),
            'swift_clip_us_best': float(swift_clip_us),
            'scipy_us_best': float(scipy_us),
            'swift_vs_pyg': float(sw_vs_pyg),
            'swift_vs_scipy': float(sw_vs_sc),
        })
        print(f"{k:>3}  {pyg_us:>9.1f}µ {swift_full_us:>11.1f}µ {swift_clip_us:>11.1f}µ {scipy_us:>9.1f}µ  {sw_vs_pyg:>8.1f}× {sw_vs_sc:>9.1f}×")

    out = {'graph': 'ribosome rebuilt', 'n': int(n), 'edges': int(len(indices)),
           'seed': int(seed), 'rows': rows}
    with open(RESULTS, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {RESULTS}")

    sw_vs_pyg_med = float(np.median([r['swift_vs_pyg'] for r in rows]))
    print(f"\nMedian Swift / PyG: {sw_vs_pyg_med:.1f}×")
    if sw_vs_pyg_med >= 10:
        print("PASSES the >10× PyG gate. Recovers the dataloader claim.")
    else:
        print("Does NOT pass >10× PyG. Honest report.")

if __name__ == '__main__':
    main()
