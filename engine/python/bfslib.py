"""
bfslib — Python ctypes wrapper around BFSLib.swift

Zero-copy on Apple Silicon: numpy int32 arrays are passed by pointer to
the Swift dylib. The Swift code reads them directly out of unified
memory; no marshalling, no copy, no serialisation.

Usage:
    from bfslib import bfs_traverse, bfs_traverse_multi
    import scipy.sparse as sp

    # From a scipy CSR (must be int32 indices/indptr)
    G = sp.random(1000, 1000, density=0.01, format='csr')
    G = G.astype(bool).astype(np.int32)
    offsets = G.indptr.astype(np.int32)
    indices = G.indices.astype(np.int32)

    frame_of = bfs_traverse(offsets, indices, seed=0)
    # frame_of[i] = BFS frame at which node i was first reached, -1 if unreached
"""
import ctypes
import os
import numpy as np
from pathlib import Path

# Locate the dylib. Search order:
#   1. BFSLIB_PATH env var (if set)
#   2. Relative to this module (after `swift build -c release --product BFSLib`)
#   3. System library path (for installed copies)
_HERE = Path(__file__).resolve().parent
_DYLIB_CANDIDATES = []
if os.environ.get("BFSLIB_PATH"):
    _DYLIB_CANDIDATES.append(Path(os.environ["BFSLIB_PATH"]))
_DYLIB_CANDIDATES.extend([
    _HERE.parent / 'swift' / '.build' / 'arm64-apple-macosx' / 'release' / 'libBFSLib.dylib',
    _HERE.parent / 'swift' / '.build' / 'release' / 'libBFSLib.dylib',
    Path('/usr/local/lib/libBFSLib.dylib'),
])

_lib = None
for _p in _DYLIB_CANDIDATES:
    if _p.exists():
        _lib = ctypes.CDLL(str(_p))
        break
if _lib is None:
    raise FileNotFoundError(
        f"libBFSLib.dylib not found. Tried: {[str(p) for p in _DYLIB_CANDIDATES]}\n"
        "Build with: cd ../swift && swift build -c release --product BFSLib"
    )

# Function signatures
_lib.bfs_traverse.argtypes = [
    ctypes.c_int32,                                            # nodeCount
    ctypes.POINTER(ctypes.c_int32),                            # offsets
    ctypes.POINTER(ctypes.c_int32),                            # indices
    ctypes.c_int32,                                            # seed
    ctypes.POINTER(ctypes.c_int32),                            # frame_of_out
]
_lib.bfs_traverse.restype = ctypes.c_int32

_lib.bfs_traverse_multi.argtypes = [
    ctypes.c_int32,                                            # nodeCount
    ctypes.POINTER(ctypes.c_int32),                            # offsets
    ctypes.POINTER(ctypes.c_int32),                            # indices
    ctypes.POINTER(ctypes.c_int32),                            # seeds
    ctypes.c_int32,                                            # nSeeds
    ctypes.POINTER(ctypes.c_int32),                            # frame_of_out (nSeeds × nodeCount)
]
_lib.bfs_traverse_multi.restype = ctypes.c_uint64               # wall-clock ns

_lib.bfs_lib_version.argtypes = []
_lib.bfs_lib_version.restype = ctypes.c_char_p


def version() -> str:
    """Library identification string."""
    return _lib.bfs_lib_version().decode('utf-8')


def _i32_ptr(arr: np.ndarray) -> ctypes._Pointer:
    """Get an int32 pointer into a numpy array (in-place, zero-copy)."""
    if arr.dtype != np.int32:
        raise TypeError(f"expected int32, got {arr.dtype}")
    if not arr.flags['C_CONTIGUOUS']:
        raise ValueError("array must be C-contiguous")
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def bfs_traverse(offsets: np.ndarray, indices: np.ndarray, seed: int) -> np.ndarray:
    """
    Single-source BFS.

    Parameters
    ----------
    offsets : numpy int32 array, length n+1 — CSR row pointers
    indices : numpy int32 array, length offsets[-1] — CSR column indices
    seed    : int — starting node

    Returns
    -------
    frame_of : numpy int32 array, length n — frame at which each node was reached
               (-1 for unreached)
    """
    n = len(offsets) - 1
    frame_of = np.empty(n, dtype=np.int32)
    rc = _lib.bfs_traverse(
        ctypes.c_int32(n),
        _i32_ptr(offsets),
        _i32_ptr(indices),
        ctypes.c_int32(seed),
        _i32_ptr(frame_of),
    )
    if rc < 0:
        raise RuntimeError(f"bfs_traverse failed (rc={rc})")
    return frame_of


def bfs_traverse_multi(offsets: np.ndarray, indices: np.ndarray,
                       seeds: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Multi-source BFS — runs one BFS per seed sequentially in the dylib.

    Parameters
    ----------
    offsets : numpy int32 array, CSR row pointers (length n+1)
    indices : numpy int32 array, CSR column indices
    seeds   : numpy int32 array, length s — starting nodes

    Returns
    -------
    frames  : numpy int32 array of shape (s, n) — frame_of arrays per seed
    wall_ns : int — total wall-clock nanoseconds (measured inside Swift)
    """
    n = len(offsets) - 1
    s = len(seeds)
    frames = np.empty((s, n), dtype=np.int32)
    wall_ns = _lib.bfs_traverse_multi(
        ctypes.c_int32(n),
        _i32_ptr(offsets),
        _i32_ptr(indices),
        _i32_ptr(seeds),
        ctypes.c_int32(s),
        _i32_ptr(frames),
    )
    return frames, int(wall_ns)


if __name__ == '__main__':
    print(version())
    # Smoke test on a tiny graph: 0 — 1 — 2 — 3 — 4
    offsets = np.array([0, 1, 3, 5, 7, 8], dtype=np.int32)
    indices = np.array([1, 0, 2, 1, 3, 2, 4, 3], dtype=np.int32)
    frames = bfs_traverse(offsets, indices, seed=0)
    print(f"path graph BFS from 0: {frames.tolist()}")
    assert frames.tolist() == [0, 1, 2, 3, 4]
    print("smoke test PASS")
