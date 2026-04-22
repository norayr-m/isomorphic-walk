// KowalskiCrushGPU — Metal compute version of the BFS frontier propagator.
//
// CAVEAT (measured 2026-04-19, M5 Max, 10 739-node ribosome graph):
// On single-source BFS at this scale, the GPU is ~13× SLOWER than the
// unified-memory CPU version (8 002 µs vs 633 µs). Per-frame command-buffer
// commit + waitUntilCompleted round-trip (~380 µs) dominates the wall time;
// the actual kernel work is small relative to dispatch overhead, and the
// median frontier (a few hundred nodes) underfills the GPU's 5 120 threads.
// This target wins under multi-source dispatch (every residue as seed in
// one kernel) or on graphs ≥ 10⁶ nodes with frontiers ≥ 100 k. For
// production single-source BFS at biomolecule scale, use KowalskiCrush
// (CPU) instead.
//
// Same CSR-on-disk format as the CPU KowalskiCrush. Same observable result
// (frame_of[] is bit-exact identical). The kernel runs on the GPU; the CPU
// owns the per-frame loop, frontier swap, and termination check.
//
// Frontier expansion strategy: one thread per current-frontier node. Each
// thread iterates that node's neighbours, atomically claims unvisited ones,
// appends to the next-frontier buffer using an atomic counter, and writes
// the frame index. CAS on visited[] guarantees each node is enqueued exactly
// once.
//
// Storage: MTLResourceStorageModeShared for everything (Apple Silicon
// unified memory; no copy across the CPU/GPU boundary).
//
// CLI: KowalskiCrushGPU <graphDir> [framesOutPath]

import Foundation
import Metal

// ── POSIX file helpers ──
func openOrDie(_ path: String, _ flags: Int32, _ mode: mode_t = 0) -> Int32 {
    let fd = open(path, flags, mode)
    precondition(fd >= 0, "open(\(path)) failed: \(String(cString: strerror(errno)))")
    return fd
}

func fileSizeOf(_ path: String) -> Int {
    var st = stat()
    guard stat(path, &st) == 0 else { return -1 }
    return Int(st.st_size)
}

func readAll<T>(path: String, count: Int) -> [T] {
    let fd = openOrDie(path, O_RDONLY)
    defer { close(fd) }
    let buf = UnsafeMutableBufferPointer<T>.allocate(capacity: count)
    let expected = count * MemoryLayout<T>.stride
    let got = read(fd, buf.baseAddress!, expected)
    precondition(got == expected, "read \(got)/\(expected) bytes from \(path)")
    let arr = Array(buf)
    buf.deallocate()
    return arr
}

// ── Args ──
let args = CommandLine.arguments
let graphDir: String
let framesOut: String
if args.count >= 2 {
    graphDir = args[1]
    framesOut = args.count >= 3 ? args[2] : "\(graphDir)/frames_gpu.bin"
} else {
    graphDir = ProcessInfo.processInfo.environment["PROTEIN_DIR"] ?? "data/4V9D"
    framesOut = "\(graphDir)/frames_gpu.bin"
}

print("=== KowalskiCrush GPU (Metal) ===")
print("Graph dir: \(graphDir)")

// ── Load CSR ──
let offsetsBytes = fileSizeOf("\(graphDir)/offsets.bin")
precondition(offsetsBytes > 4, "offsets.bin missing")
let nodeCount = (offsetsBytes / 4) - 1
let states: [Int8]   = readAll(path: "\(graphDir)/states.bin",  count: nodeCount)
let offsets: [Int32] = readAll(path: "\(graphDir)/offsets.bin", count: nodeCount + 1)
let edgeCount = Int(offsets[nodeCount])
let indices: [Int32] = readAll(path: "\(graphDir)/indices.bin", count: edgeCount)
print("Nodes: \(nodeCount)  Edges: \(edgeCount)")

// Find seed (first node with state == +1, else first with non-zero degree)
var seedNode = -1
for i in 0..<nodeCount where states[i] == 1 { seedNode = i; break }
if seedNode < 0 {
    for i in 0..<nodeCount where offsets[i+1] > offsets[i] { seedNode = i; break }
}
precondition(seedNode >= 0, "no seed found")
print("Seed: node \(seedNode)")

// ── Metal setup ──
guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("No Metal device available")
}
print("Device: \(device.name)")
print("Unified memory: \(device.hasUnifiedMemory)")

let kernelSource = """
#include <metal_stdlib>
using namespace metal;

kernel void bfs_expand(
    device const int*           current_frontier  [[buffer(0)]],
    device const uint&          current_count     [[buffer(1)]],
    device const int*           offsets           [[buffer(2)]],
    device const int*           indices           [[buffer(3)]],
    device atomic_uint*         visited           [[buffer(4)]],
    device int*                 next_frontier     [[buffer(5)]],
    device atomic_uint*         next_count        [[buffer(6)]],
    device int*                 frame_of          [[buffer(7)]],
    constant int&               frame_index       [[buffer(8)]],
    uint                        tid               [[thread_position_in_grid]])
{
    if (tid >= current_count) return;
    int u = current_frontier[tid];
    int start = offsets[u];
    int end   = offsets[u + 1];
    for (int j = start; j < end; ++j) {
        int v = indices[j];
        uint expected = 0u;
        // CAS visited[v]: 0 -> 1. Only the winning thread enqueues.
        if (atomic_compare_exchange_weak_explicit(&visited[v], &expected, 1u,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) {
            uint slot = atomic_fetch_add_explicit(next_count, 1u, memory_order_relaxed);
            next_frontier[slot] = v;
            frame_of[v] = frame_index + 1;
        }
    }
}
"""

let compileOpts = MTLCompileOptions()
let library: MTLLibrary
do {
    library = try device.makeLibrary(source: kernelSource, options: compileOpts)
} catch {
    fatalError("Metal kernel compile failed: \(error)")
}
guard let kernel = library.makeFunction(name: "bfs_expand") else {
    fatalError("kernel function not found")
}
let pso = try device.makeComputePipelineState(function: kernel)
guard let queue = device.makeCommandQueue() else { fatalError("no queue") }

// ── Buffers (shared storage = unified memory, no copy) ──
let opt: MTLResourceOptions = [.storageModeShared]

let bufOffsets = device.makeBuffer(bytes: offsets, length: (nodeCount + 1) * 4, options: opt)!
let bufIndices = device.makeBuffer(bytes: indices, length: edgeCount * 4, options: opt)!

// visited as atomic_uint (4 bytes per node)
let bufVisited = device.makeBuffer(length: nodeCount * 4, options: opt)!
memset(bufVisited.contents(), 0, nodeCount * 4)
let visitedPtr = bufVisited.contents().bindMemory(to: UInt32.self, capacity: nodeCount)
visitedPtr[seedNode] = 1

// frontier ping-pong, sized to nodeCount upper bound
let bufFrontA = device.makeBuffer(length: nodeCount * 4, options: opt)!
let bufFrontB = device.makeBuffer(length: nodeCount * 4, options: opt)!
let frontAPtr = bufFrontA.contents().bindMemory(to: Int32.self, capacity: nodeCount)
frontAPtr[0] = Int32(seedNode)

// counters
var currentCount: UInt32 = 1
let bufCurrentCount = device.makeBuffer(length: 4, options: opt)!
let bufNextCount    = device.makeBuffer(length: 4, options: opt)!

// frame_of (-1 sentinel, then 0 at seed)
let bufFrameOf = device.makeBuffer(length: nodeCount * 4, options: opt)!
let frameOfPtr = bufFrameOf.contents().bindMemory(to: Int32.self, capacity: nodeCount)
for i in 0..<nodeCount { frameOfPtr[i] = -1 }
frameOfPtr[seedNode] = 0

// frame index (for kernel's frame_of write)
let bufFrameIndex = device.makeBuffer(length: 4, options: opt)!

// ── BFS loop on GPU ──
var current = bufFrontA
var next    = bufFrontB
var frame: Int32 = 0
var totalReached = 1

let threadsPerThreadgroup = MTLSize(width: pso.maxTotalThreadsPerThreadgroup, height: 1, depth: 1)

var timebaseInfo = mach_timebase_info_data_t()
mach_timebase_info(&timebaseInfo)
let clockStart = mach_absolute_time()

while currentCount > 0 {
    // Set counters
    bufCurrentCount.contents().bindMemory(to: UInt32.self, capacity: 1)[0] = currentCount
    bufNextCount.contents().bindMemory(to: UInt32.self, capacity: 1)[0] = 0
    bufFrameIndex.contents().bindMemory(to: Int32.self, capacity: 1)[0] = frame

    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(current,         offset: 0, index: 0)
    enc.setBuffer(bufCurrentCount, offset: 0, index: 1)
    enc.setBuffer(bufOffsets,      offset: 0, index: 2)
    enc.setBuffer(bufIndices,      offset: 0, index: 3)
    enc.setBuffer(bufVisited,      offset: 0, index: 4)
    enc.setBuffer(next,            offset: 0, index: 5)
    enc.setBuffer(bufNextCount,    offset: 0, index: 6)
    enc.setBuffer(bufFrameOf,      offset: 0, index: 7)
    enc.setBuffer(bufFrameIndex,   offset: 0, index: 8)

    let totalThreads = MTLSize(width: Int(currentCount), height: 1, depth: 1)
    enc.dispatchThreads(totalThreads, threadsPerThreadgroup: threadsPerThreadgroup)
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    let nextCount = bufNextCount.contents().bindMemory(to: UInt32.self, capacity: 1)[0]
    frame += 1
    totalReached += Int(nextCount)
    // swap
    let tmp = current; current = next; next = tmp
    currentCount = nextCount
}

let clockEnd = mach_absolute_time()
let elapsedNs = (clockEnd - clockStart) * UInt64(timebaseInfo.numer) / UInt64(timebaseInfo.denom)
let elapsedUs = elapsedNs / 1000

// ── Receipt ──
print("\n=== KowalskiCrushGPU — RECEIPT ===")
print("Nodes:           \(nodeCount)")
print("Edges:           \(edgeCount)")
print("Seed:            \(seedNode)")
print("BFS frames:      \(frame)")
print("Nodes reached:   \(totalReached) / \(nodeCount)")
print("Wall time:       \(elapsedNs) ns (\(elapsedUs) µs)")
let nsPerEdge = edgeCount > 0 ? elapsedNs / UInt64(edgeCount) : 0
print("ns/edge:         \(nsPerEdge)")
print("Note: 'wall time' includes per-frame command-buffer commit + GPU sync.")
print("==================================")

// ── Write frames.bin ──
let outFd = openOrDie(framesOut, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
let bytes = nodeCount * 4
let wrote = write(outFd, bufFrameOf.contents(), bytes)
close(outFd)
precondition(wrote == bytes, "wrote \(wrote)/\(bytes) bytes to \(framesOut)")
print("Frames written: \(framesOut) (\(bytes) bytes, Int32×\(nodeCount))")
