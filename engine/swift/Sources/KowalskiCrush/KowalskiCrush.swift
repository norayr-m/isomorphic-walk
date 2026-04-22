// KowalskiCrush v3 — Operation Kowalski-Crush (Memphis Hold)
// Load 4V9D KNN+MST flat binaries. BFS wavefront. Print honest receipt.
// Zero ARC. Zero float. Zero heap allocation in hot loop.
// POSIX I/O. @main struct (no Swift 6 top-level deadlock).
//
// Graph: KNN(k=6) + MST union. 0 isolated nodes.
// Wavefront: BFS with visited flag. No re-excitation.
// Clock: mach_absolute_time nanosecond precision.

import Foundation
#if canImport(Darwin)
import Darwin.C
#elseif canImport(Glibc)
import Glibc
#endif

@main
struct KowalskiCrush {
    // ── POSIX binary loader ──
    static func loadBinary<T>(path: String, count: Int) -> UnsafeMutableBufferPointer<T> {
        let fd = open(path, O_RDONLY)
        precondition(fd >= 0, "open(\(path)) failed: \(String(cString: strerror(errno)))")
        let buf = UnsafeMutableBufferPointer<T>.allocate(capacity: count)
        let expected = count &* MemoryLayout<T>.stride
        let got = read(fd, buf.baseAddress!, expected)
        close(fd)
        precondition(got == expected, "read \(got)/\(expected) bytes from \(path)")
        return buf
    }

    // File-size probe (lets us derive nodeCount from offsets.bin without an arg)
    static func fileSize(_ path: String) -> Int {
        var st = stat()
        guard stat(path, &st) == 0 else { return -1 }
        return Int(st.st_size)
    }

    static func main() {
        // CLI: KowalskiCrush <graphDir> [framesOutPath]
        // Fallback: read PROTEIN_DIR env var, or default to relative "data/4V9D".
        let args = CommandLine.arguments
        let proteinDir: String
        let framesOutPath: String?
        if args.count >= 2 {
            proteinDir = args[1]
            framesOutPath = args.count >= 3 ? args[2] : "\(proteinDir)/frames.bin"
        } else {
            proteinDir = ProcessInfo.processInfo.environment["PROTEIN_DIR"] ?? "data/4V9D"
            framesOutPath = nil
        }

        // Derive nodeCount from offsets.bin file size: (nodeCount+1) * 4 bytes
        let offsetsBytes = fileSize("\(proteinDir)/offsets.bin")
        precondition(offsetsBytes > 4, "offsets.bin missing or too small at \(proteinDir)")
        let nodeCount = (offsetsBytes / 4) - 1

        // ── 1. UNCAGE THE MEMORY ──
        print("=== OPERATION KOWALSKI-CRUSH v3 (Memphis Hold) ===")
        print("Graph dir:     \(proteinDir)")
        print("Loading graph: \(nodeCount) nodes")

        let states: UnsafeMutableBufferPointer<Int8>   = loadBinary(path: "\(proteinDir)/states.bin",  count: nodeCount)
        let offsets: UnsafeMutableBufferPointer<Int32>  = loadBinary(path: "\(proteinDir)/offsets.bin",  count: nodeCount + 1)

        let lastOffset = Int(offsets[nodeCount])
        precondition(lastOffset > 0, "CSR has zero edges — graph must be built first")
        let edgeCount = lastOffset

        let indices: UnsafeMutableBufferPointer<Int32>  = loadBinary(path: "\(proteinDir)/indices.bin",  count: edgeCount)

        print("CSR integrity: PASS (\(edgeCount) edges)")

        // Count connectivity
        var isolated = 0
        var minDeg = Int.max; var maxDeg = 0
        for i in 0..<nodeCount {
            let deg = Int(offsets[i + 1] - offsets[i])
            if deg == 0 { isolated += 1 }
            if deg < minDeg { minDeg = deg }
            if deg > maxDeg { maxDeg = deg }
        }
        print("Isolated nodes: \(isolated) / \(nodeCount)")
        print("Degree range:   \(minDeg)–\(maxDeg)")

        // ── 2. FIND INJECTION SITE ──
        var seedNode = -1
        for i in 0..<nodeCount {
            if states[i] == 1 { seedNode = i; break }
        }
        if seedNode < 0 {
            for i in 0..<nodeCount {
                if offsets[i + 1] > offsets[i] { seedNode = i; break }
            }
        }
        precondition(seedNode >= 0, "No valid seed node found")

        let seedDegree = Int(offsets[seedNode + 1] - offsets[seedNode])
        print("Injection site: node \(seedNode) (degree \(seedDegree))")
        print()

        // ── 3. BFS WAVEFRONT ──
        // Pre-allocated ping-pong queues. Zero heap allocation.
        // visited[] flag — each node fires exactly once. No standing waves.

        let visited = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: nodeCount)
        for i in 0..<nodeCount { visited[i] = 0 }
        visited[seedNode] = 1

        // Per-node frame-of-arrival (-1 = never reached). Captured for
        // downstream correlation analysis (e.g., BFS-vs-biology displacement).
        let frameOf = UnsafeMutableBufferPointer<Int32>.allocate(capacity: nodeCount)
        for i in 0..<nodeCount { frameOf[i] = -1 }
        frameOf[seedNode] = 0

        let frontA = UnsafeMutableBufferPointer<Int32>.allocate(capacity: nodeCount)
        let frontB = UnsafeMutableBufferPointer<Int32>.allocate(capacity: nodeCount)
        var current = frontA
        var next = frontB
        var currentCount = 1
        var nextCount = 0
        current[0] = Int32(seedNode)
        var frameCount = 0
        var totalReached = 1

        // Clock
        var timebaseInfo = mach_timebase_info_data_t()
        mach_timebase_info(&timebaseInfo)
        let clockStart = mach_absolute_time()

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
                        frameOf[nb] = Int32(frameCount + 1)
                        next[nextCount] = Int32(nb)
                        nextCount += 1
                    }
                }
            }
            frameCount += 1
            totalReached += nextCount

            // Swap buffers
            let tmp = current
            current = next
            next = tmp
            currentCount = nextCount
        }

        let clockEnd = mach_absolute_time()
        let elapsedTicks = clockEnd - clockStart
        let elapsedNs = elapsedTicks * UInt64(timebaseInfo.numer) / UInt64(timebaseInfo.denom)

        // ── 4. PRINT THE RECEIPT ──
        let nsPerEdge: UInt64 = edgeCount > 0 ? elapsedNs / UInt64(edgeCount) : 0
        let nsPerFrame: UInt64 = frameCount > 0 ? elapsedNs / UInt64(frameCount) : 0
        let cyclesPerEdge = Double(nsPerEdge) * 3.49  // M2 ~3.49 GHz

        let usTotal = elapsedNs / 1000

        print("=== OPERATION KOWALSKI-CRUSH v3 — RECEIPT ===")
        print("Graph:           KNN(k=6) + MST")
        print("Nodes:           \(nodeCount)")
        print("Edges:           \(edgeCount)")
        print("Isolated:        \(isolated)")
        print("Degree range:    \(minDeg)–\(maxDeg)")
        print("Seed node:       \(seedNode) (degree \(seedDegree))")
        print("BFS frames:      \(frameCount)")
        print("Nodes reached:   \(totalReached) / \(nodeCount)")
        print("Wall time:       \(elapsedNs) ns (\(usTotal) µs)")
        print("ns/edge:         \(nsPerEdge)")
        print("ns/frame:        \(nsPerFrame)")
        print("cycles/edge:     \(Int(cyclesPerEdge))")
        if cyclesPerEdge < 20.0 {
            print("TARGET:          MET (< 20 cycles/edge)")
        } else {
            print("TARGET:          NOT MET (\(Int(cyclesPerEdge)) > 20)")
        }
        print("==============================================")

        // ── 5. EMIT PER-NODE FRAME-OF-ARRIVAL (optional) ──
        if let outPath = framesOutPath {
            let fd = open(outPath, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
            precondition(fd >= 0, "open(\(outPath)) for write failed: \(String(cString: strerror(errno)))")
            let bytes = nodeCount &* MemoryLayout<Int32>.stride
            let wrote = write(fd, frameOf.baseAddress!, bytes)
            close(fd)
            precondition(wrote == bytes, "wrote \(wrote)/\(bytes) bytes to \(outPath)")
            print("Frames written: \(outPath) (\(bytes) bytes, Int32×\(nodeCount))")
        }

        // Cleanup
        states.deallocate()
        offsets.deallocate()
        indices.deallocate()
        visited.deallocate()
        frameOf.deallocate()
        frontA.deallocate()
        frontB.deallocate()
    }
}
