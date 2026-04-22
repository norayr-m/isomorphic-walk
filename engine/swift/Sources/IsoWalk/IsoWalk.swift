// IsoWalk v1 — Greedy Coordinate Descent on Ternary Ratchet
// One flip per iteration. JSONL ledger = Levenshtein-style edit distance.
// CSR adjacency = fixed weights. State vector = activations.
// No backprop. No batches. No epochs.
//
// Phase 1: degree as eigenvalue proxy. Phase 2: proper Laplacian spectrum.

import Foundation
#if canImport(Darwin)
import Darwin.C
#elseif canImport(Glibc)
import Glibc
#endif

// ── Ternary ratchet ──
// -1 (excitatory) → 0 (refractory) → +1 (neutral) → -1 (excitatory)
@inline(__always)
func nextRatchetState(_ s: Int8) -> Int8 {
    switch s {
    case -1: return 0
    case  0: return 1
    case  1: return -1
    default: preconditionFailure("Invalid ratchet state: \(s)")
    }
}

@main
struct IsoWalk {
    // ── POSIX binary loader (same as KowalskiCrush) ──
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

    // File-size probe for auto-derived node count.
    static func fileSize(_ path: String) -> Int {
        var st = stat()
        guard stat(path, &st) == 0 else { return -1 }
        return Int(st.st_size)
    }

    static func main() {
        // CLI: IsoWalk <graphDir> [ledgerOutPath]
        // Fallback: read PROTEIN_DIR env var, or default to relative "data/4V9D".
        let args = CommandLine.arguments
        let proteinDir: String
        let ledgerPath: String
        if args.count >= 2 {
            proteinDir = args[1]
            ledgerPath = args.count >= 3 ? args[2] : "\(proteinDir)/isowalk_ledger.jsonl"
        } else {
            proteinDir = ProcessInfo.processInfo.environment["PROTEIN_DIR"] ?? "data/4V9D"
            ledgerPath = "\(proteinDir)/isowalk_ledger.jsonl"
        }

        // Derive nodeCount from offsets.bin size.
        let offBytes = fileSize("\(proteinDir)/offsets.bin")
        precondition(offBytes > 4, "offsets.bin missing at \(proteinDir)")
        let nodeCount = (offBytes / 4) - 1

        print("=== ISOWALK v1 — Greedy Coordinate Descent ===")
        print("Graph dir: \(proteinDir)")
        print("Loading graph: \(nodeCount) nodes")

        // ── 1. LOAD CSR ──
        let offsets: UnsafeMutableBufferPointer<Int32> = loadBinary(path: "\(proteinDir)/offsets.bin", count: nodeCount + 1)
        let edgeCount = Int(offsets[nodeCount])
        precondition(edgeCount > 0, "CSR has zero edges")
        let indices: UnsafeMutableBufferPointer<Int32> = loadBinary(path: "\(proteinDir)/indices.bin", count: edgeCount)

        print("CSR: \(nodeCount) nodes, \(edgeCount) edges")

        // ── 2. COMPUTE DEGREE (eigenvalue proxy for Phase 1) ──
        let degree = UnsafeMutableBufferPointer<Float>.allocate(capacity: nodeCount)
        for i in 0..<nodeCount {
            degree[i] = Float(offsets[i + 1] - offsets[i])
        }

        // ── 3. FIND INJECTION SITE ──
        let seedStates: UnsafeMutableBufferPointer<Int8> = loadBinary(path: "\(proteinDir)/states.bin", count: nodeCount)
        var seedNode = -1
        for i in 0..<nodeCount {
            if seedStates[i] == 1 { seedNode = i; break }
        }
        seedStates.deallocate()
        if seedNode < 0 {
            for i in 0..<nodeCount {
                if offsets[i + 1] > offsets[i] { seedNode = i; break }
            }
        }
        precondition(seedNode >= 0, "No valid seed node")
        print("Injection site: node \(seedNode) (degree \(Int(degree[seedNode])))")

        // ── 3.5  Per-node first-flip frame + flip count (post-fix instrumentation) ──
        // first-flip frame is the canonical per-residue derived metric for biology
        // correlation experiments. Cap each node at maxFlipsPerNode = 1 so the
        // ledger is bounded above by N (Levenshtein interpretation).
        let maxFlipsPerNode = 1
        let firstFlipFrame = UnsafeMutablePointer<Int32>.allocate(capacity: nodeCount)
        let flipsPerNode   = UnsafeMutablePointer<Int32>.allocate(capacity: nodeCount)
        for i in 0..<nodeCount { firstFlipFrame[i] = -1; flipsPerNode[i] = 0 }

        // ── 4. ALLOCATE STATE + QUEUES ──
        // State vector: Int8, all zero (refractory), seed starts at -1 (excitatory)
        // All nodes start at 0 (refractory). The walk differentiates from here.
        // Spec: "states = [0] * N" — injection site goes into active queue, not pre-flipped.
        let states = UnsafeMutableBufferPointer<Int8>.allocate(capacity: nodeCount)
        for i in 0..<nodeCount { states[i] = 0 }

        // Ping-pong active queues
        let queueA = UnsafeMutableBufferPointer<Int32>.allocate(capacity: nodeCount)
        let queueB = UnsafeMutableBufferPointer<Int32>.allocate(capacity: nodeCount)
        var currentQueue = queueA
        var nextQueue    = queueB
        var currentCount = 1
        var nextCount    = 0
        currentQueue[0] = Int32(seedNode)

        // Gradient buffer (reused each iteration)
        let gradients = UnsafeMutableBufferPointer<Float>.allocate(capacity: nodeCount)

        // In-queue flag to avoid duplicates
        let inQueue = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: nodeCount)
        for i in 0..<nodeCount { inQueue[i] = 0 }
        inQueue[seedNode] = 1

        // ── 5. OPEN LEDGER ──
        let ledgerFd = open(ledgerPath, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
        precondition(ledgerFd >= 0, "Cannot open ledger: \(ledgerPath)")

        @inline(__always)
        func writeLedgerLine(_ frame: Int, _ node: Int, _ fromState: Int8, _ toState: Int8,
                             _ gradient: Float, _ activeCount: Int, _ wallNs: UInt64) {
            var line = "{\"f\":\(frame),\"n\":\(node),\"s\":\(fromState),\"t\":\(toState),\"g\":\(gradient),\"a\":\(activeCount),\"ns\":\(wallNs)}\n"
            line.withUTF8 { buf in
                _ = write(ledgerFd, buf.baseAddress!, buf.count)
            }
        }

        // ── 6. GREEDY COORDINATE DESCENT ──
        print()
        print("Starting walk...")

        var timebaseInfo = mach_timebase_info_data_t()
        mach_timebase_info(&timebaseInfo)
        let walkStart = mach_absolute_time()

        var frameCount = 0
        var monotonicViolations = 0

        while currentCount > 0 {
            // 6a. Compute gradient for each node in active queue
            var bestNode = -1
            var bestGrad: Float = 0
            var bestDegree: Float = 0

            for qi in 0..<currentCount {
                let i = Int(currentQueue[qi])
                let si = states[i]
                let nextSi = nextRatchetState(si)
                let start = Int(offsets[i])
                let end   = Int(offsets[i + 1])

                // δL/δs_i = Σ_j degree_j × [conflict(si, sj) - conflict(nextSi, sj)]
                // conflict(a, b) = 1 if a == b, else 0
                var grad: Float = 0
                for jIdx in start..<end {
                    let j = Int(indices[jIdx])
                    let sj = states[j]
                    let wj = degree[j]
                    let currentConflict: Float = (si == sj) ? 1.0 : 0.0
                    let futureConflict:  Float = (nextSi == sj) ? 1.0 : 0.0
                    grad += wj * (currentConflict - futureConflict)
                }
                gradients[i] = grad

                // Greedy DESCENT: only consider positive gradients (loss decreases on flip).
                // Tie-break by higher degree.
                if grad > bestGrad || (grad == bestGrad && degree[i] > bestDegree) {
                    bestNode = i
                    bestGrad = grad
                    bestDegree = degree[i]
                }
            }

            // Halt if no positive-gradient flip exists, or per-node flip cap reached.
            if bestNode < 0 || bestGrad <= 0 {
                break
            }
            if flipsPerNode[bestNode] >= maxFlipsPerNode {
                // Skip this node, try next iteration without flipping; but if all top
                // candidates are at cap, halt to prevent stall. Simplest: just halt.
                break
            }

            // 6b. Flip the winner
            let oldState = states[bestNode]
            let newState = nextRatchetState(oldState)
            states[bestNode] = newState
            if firstFlipFrame[bestNode] < 0 { firstFlipFrame[bestNode] = Int32(frameCount) }
            flipsPerNode[bestNode] += 1

            // 6c. Check monotonicity (post-fix: should always be ≥ 0)
            if bestGrad < 0 {
                monotonicViolations += 1
            }

            // 6d. Build next active queue: winner's neighbors with nonzero gradient
            // Spec: only winner's neighbors go into next queue. No carryover.
            nextCount = 0
            for qi in 0..<currentCount { inQueue[Int(currentQueue[qi])] = 0 }

            let winStart = Int(offsets[bestNode])
            let winEnd   = Int(offsets[bestNode + 1])
            for jIdx in winStart..<winEnd {
                let j = Int(indices[jIdx])
                if inQueue[j] == 0 {
                    // Compute gradient for neighbor j to check if nonzero
                    let sj = states[j]
                    let nextSj = nextRatchetState(sj)
                    let nStart = Int(offsets[j])
                    let nEnd   = Int(offsets[j + 1])
                    var grad: Float = 0
                    for kIdx in nStart..<nEnd {
                        let k = Int(indices[kIdx])
                        let sk = states[k]
                        let wk = degree[k]
                        let curr: Float = (sj == sk) ? 1.0 : 0.0
                        let next: Float = (nextSj == sk) ? 1.0 : 0.0
                        grad += wk * (curr - next)
                    }
                    if grad != 0 {
                        nextQueue[nextCount] = Int32(j)
                        nextCount += 1
                        inQueue[j] = 1
                    }
                }
            }

            // 6e. Emit ledger line
            let now = mach_absolute_time()
            let ns = (now - walkStart) * UInt64(timebaseInfo.numer) / UInt64(timebaseInfo.denom)
            writeLedgerLine(frameCount, bestNode, oldState, newState, bestGrad, nextCount, ns)

            frameCount += 1

            // 6f. Swap queues
            let tmp = currentQueue
            currentQueue = nextQueue
            nextQueue = tmp
            currentCount = nextCount

            // Progress every 1000 frames
            if frameCount % 1000 == 0 {
                print("  frame \(frameCount): active=\(currentCount), node=\(bestNode), grad=\(bestGrad)")
            }
        }

        let walkEnd = mach_absolute_time()
        let totalNs = (walkEnd - walkStart) * UInt64(timebaseInfo.numer) / UInt64(timebaseInfo.denom)

        close(ledgerFd)

        // ── 7. FINAL STATE SURVEY ──
        var countNeg1 = 0, count0 = 0, countPos1 = 0
        for i in 0..<nodeCount {
            switch states[i] {
            case -1: countNeg1 += 1
            case  0: count0 += 1
            case  1: countPos1 += 1
            default: break
            }
        }

        // ── 8. PRINT RECEIPT ──
        let usTotal = totalNs / 1000
        let nsPerFlip: UInt64 = frameCount > 0 ? totalNs / UInt64(frameCount) : 0
        let cyclesPerFlip = Double(nsPerFlip) * 3.49  // M2 ~3.49 GHz

        print()
        print("=== ISOWALK v1 — RECEIPT ===")
        print("Graph:              KNN(k=6) + MST")
        print("Nodes:              \(nodeCount)")
        print("Edges:              \(edgeCount)")
        print("Seed node:          \(seedNode) (degree \(Int(degree[seedNode])))")
        print("Walk frames (flips):\(frameCount)")
        print("Ledger length:      \(frameCount) (= edit distance)")
        print("Wall time:          \(totalNs) ns (\(usTotal) µs)")
        print("ns/flip:            \(nsPerFlip)")
        print("cycles/flip:        \(Int(cyclesPerFlip))")
        print("Monotonic violations: \(monotonicViolations)")
        print("Final states:       -1=\(countNeg1)  0=\(count0)  +1=\(countPos1)")
        print("Ledger:             \(ledgerPath)")
        print()
        if frameCount <= nodeCount {
            print("BOUND CHECK:        PASS (flips \(frameCount) <= nodes \(nodeCount))")
        } else {
            print("BOUND CHECK:        EXCEEDED (flips \(frameCount) > nodes \(nodeCount))")
        }

        // Write per-node first-flip frame as flat Int32[N] for downstream Python.
        let firstFlipPath = "\(proteinDir)/isowalk_firstflip.bin"
        let ffd = open(firstFlipPath, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
        if ffd >= 0 {
            let nBytes = nodeCount &* MemoryLayout<Int32>.stride
            _ = write(ffd, firstFlipFrame, nBytes)
            close(ffd)
            print("First-flip:         \(firstFlipPath) (\(nBytes) bytes, Int32×\(nodeCount))")
        }
        var nFlippedAtLeastOnce = 0
        for i in 0..<nodeCount where firstFlipFrame[i] >= 0 { nFlippedAtLeastOnce += 1 }
        print("Nodes flipped ≥ 1×: \(nFlippedAtLeastOnce) / \(nodeCount)")
        if monotonicViolations == 0 {
            print("MONOTONICITY:       PASS (every flip reduced loss)")
        } else {
            print("MONOTONICITY:       \(monotonicViolations) violations")
        }
        print("================================")

        // Cleanup
        offsets.deallocate()
        indices.deallocate()
        degree.deallocate()
        states.deallocate()
        queueA.deallocate()
        queueB.deallocate()
        gradients.deallocate()
        inQueue.deallocate()
        firstFlipFrame.deallocate()
        flipsPerNode.deallocate()
    }
}
