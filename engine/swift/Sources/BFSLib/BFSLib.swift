// BFSLib — C-callable BFS over a CSR contact graph.
//
// Designed to be loaded from Python via ctypes against a numpy-backed
// scipy.sparse.csr_matrix. Zero-copy on Apple Silicon unified memory:
// Python passes raw int32 pointers from numpy arrays, Swift reads them
// directly without copying.
//
// Same algorithm as KowalskiCrush. Same hot-loop discipline (no ARC, no
// heap allocation, ping-pong frontier queues). Output is per-node frame-
// of-arrival written into a caller-supplied Int32 buffer.
//
// Build: swift build -c release --product BFSLib
// Output: .build/arm64-apple-macosx/release/libBFSLib.dylib

import Foundation

/// Single-source BFS over a CSR graph.
///
/// - Parameters:
///   - nodeCount: number of nodes
///   - offsets: CSR row pointers, length nodeCount+1, Int32
///   - indices: CSR column indices, length offsets[nodeCount], Int32
///   - seed: starting node (0..<nodeCount)
///   - frame_of_out: caller-allocated Int32 buffer of length nodeCount,
///     filled with frame-of-arrival per node (-1 for unreached)
/// - Returns: total frame count (BFS depth)
@_cdecl("bfs_traverse")
public func bfs_traverse(
    _ nodeCount: Int32,
    _ offsets: UnsafePointer<Int32>,
    _ indices: UnsafePointer<Int32>,
    _ seed: Int32,
    _ frame_of_out: UnsafeMutablePointer<Int32>
) -> Int32 {
    let n = Int(nodeCount)
    if n <= 0 || seed < 0 || Int(seed) >= n { return -1 }

    // Initialise frame_of_out to -1
    for i in 0..<n { frame_of_out[i] = -1 }
    frame_of_out[Int(seed)] = 0

    // Visited flag (separate from frame_of so we don't need to compare against -1 in hot loop)
    let visited = UnsafeMutablePointer<UInt8>.allocate(capacity: n)
    defer { visited.deallocate() }
    for i in 0..<n { visited[i] = 0 }
    visited[Int(seed)] = 1

    let frontA = UnsafeMutablePointer<Int32>.allocate(capacity: n)
    let frontB = UnsafeMutablePointer<Int32>.allocate(capacity: n)
    defer { frontA.deallocate(); frontB.deallocate() }

    var current = frontA
    var next = frontB
    var currentCount = 1
    var nextCount = 0
    current[0] = seed
    var frame: Int32 = 0

    while currentCount > 0 {
        nextCount = 0
        for qi in 0..<currentCount {
            let i = Int(current[qi])
            let start = Int(offsets[i])
            let end   = Int(offsets[i + 1])
            for j in start..<end {
                let nb = Int(indices[j])
                if visited[nb] == 0 {
                    visited[nb] = 1
                    frame_of_out[nb] = frame + 1
                    next[nextCount] = Int32(nb)
                    nextCount += 1
                }
            }
        }
        frame += 1
        let tmp = current; current = next; next = tmp
        currentCount = nextCount
    }
    return frame
}

/// Multi-source BFS: run a BFS from each of `nSeeds` start nodes
/// sequentially, write each frame-of-arrival into a separate column of
/// `frame_of_out` (row-major: source index s, target node t →
/// frame_of_out[s * nodeCount + t]).
///
/// Caller-allocated buffer: nSeeds × nodeCount Int32 entries.
/// Returns: total wall-clock nanoseconds across all BFS runs.
@_cdecl("bfs_traverse_multi")
public func bfs_traverse_multi(
    _ nodeCount: Int32,
    _ offsets: UnsafePointer<Int32>,
    _ indices: UnsafePointer<Int32>,
    _ seeds: UnsafePointer<Int32>,
    _ nSeeds: Int32,
    _ frame_of_out: UnsafeMutablePointer<Int32>
) -> UInt64 {
    let n = Int(nodeCount)
    let s = Int(nSeeds)
    if n <= 0 || s <= 0 { return 0 }

    var timebaseInfo = mach_timebase_info_data_t()
    mach_timebase_info(&timebaseInfo)
    let t0 = mach_absolute_time()
    for src in 0..<s {
        let seed = seeds[src]
        let outOff = src * n
        _ = bfs_traverse(nodeCount, offsets, indices, seed, frame_of_out.advanced(by: outOff))
    }
    let t1 = mach_absolute_time()
    return (t1 - t0) * UInt64(timebaseInfo.numer) / UInt64(timebaseInfo.denom)
}

/// Returns library identification — useful for verifying ctypes loaded the right .dylib.
@_cdecl("bfs_lib_version")
public func bfs_lib_version() -> UnsafePointer<CChar> {
    let s = "BFSLib v1 — isowalk2 — Apple Silicon UMA CSR BFS"
    return s.withCString { ptr in
        // Leak a copy: caller doesn't free, this is a static identification string.
        let copy = strdup(ptr)
        return UnsafePointer(copy!)
    }
}
