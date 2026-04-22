import Foundation

#if canImport(Darwin)
import Darwin.C
#elseif canImport(Glibc)
import Glibc
#endif

// Represents a geometric point inside the Swift ARC loosely bounds
struct Node {
    let x: Float
    let y: Float
    let z: Float
}

// Fixed voxel structure for spatial hashing
struct Voxel: Hashable {
    let i: Int
    let j: Int
    let k: Int
}

@main
struct Crusher {
    static let voxelSize: Float = 8.5  // was 5.0 — captures RNA P-P backbone (~5.8-6.5Å)
    static let voxelSizeSq: Float = voxelSize * voxelSize

    static func main() {
        var nodes = [Node]()
        var grid: [Voxel: [Int32]] = [:]
        // CLI arg, env-var override, or default to relative data path.
        let args = CommandLine.arguments
        let dataDir = args.count >= 2
            ? args[1]
            : (ProcessInfo.processInfo.environment["PROTEIN_DIR"] ?? "data/4V9D")
        let filePath = "\(dataDir)/4V9D.cif"

        print("Initializing stream parsing for \(filePath)...")

        guard let fp = fopen(filePath, "r") else {
            fatalError("Failed to open file: \(filePath). Please ensure it is uncompressed.")
        }
        
        var linePtr: UnsafeMutablePointer<CChar>? = nil
        var lineCap: Int = 0
        var isAtomSite = false
        var headers = [String: Int]()
        var colIndex = 0
        var headerParsing = false
        
        // Fast line reading avoiding massive ARC heap
        while getline(&linePtr, &lineCap, fp) > 0 {
            guard let lineStr = String(validatingCString: linePtr!) else { continue }
            let line = lineStr.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty { continue }

            if line == "loop_" {
                headerParsing = true
                continue
            }
            
            if line.hasPrefix("_atom_site.") {
                isAtomSite = true
                if headerParsing {
                    let parts = line.split(separator: ".")
                    if parts.count > 1 {
                        headers[String(parts[1])] = colIndex
                        colIndex += 1
                    }
                }
            } else if line.hasPrefix("ATOM") || line.hasPrefix("HETATM") {
                headerParsing = false
                let tokens = line.split(whereSeparator: { $0.isWhitespace }).map(String.init)
                
                let typeIdx = headers["group_PDB"] ?? 0
                let idIdx = headers["label_atom_id"] ?? 3
                let xIdx = headers["Cartn_x"] ?? 10
                let yIdx = headers["Cartn_y"] ?? 11
                let zIdx = headers["Cartn_z"] ?? 12
                
                guard typeIdx < tokens.count, idIdx < tokens.count, zIdx < tokens.count else { continue }
                
                let group = tokens[typeIdx]
                let atomId = tokens[idIdx]
                
                if group == "ATOM" && (atomId == "CA" || atomId == "P") {
                    if let x = Float(tokens[xIdx]), let y = Float(tokens[yIdx]), let z = Float(tokens[zIdx]) {
                        nodes.append(Node(x: x, y: y, z: z))
                    }
                }
            } else if isAtomSite && line.hasPrefix("#") {
                // End of ATOM block
                break
            }
        }
        
        free(linePtr)
        fclose(fp)
        
        let nodeCount = nodes.count
        print("Payload streamed successfully. Extracted \(nodeCount) tension-bearing nodes.")
        
        // Compute Center of Mass
        var sumX: Float = 0
        var sumY: Float = 0
        var sumZ: Float = 0
        
        for i in 0..<nodeCount {
            let node = nodes[i]
            sumX += node.x
            sumY += node.y
            sumZ += node.z
            
            // Populate spatial grid
            let voxel = Voxel(
                i: Int(floor(node.x / voxelSize)),
                j: Int(floor(node.y / voxelSize)),
                k: Int(floor(node.z / voxelSize))
            )
            grid[voxel, default: []].append(Int32(i))
        }
        
        let comX = sumX / Float(nodeCount)
        let comY = sumY / Float(nodeCount)
        let comZ = sumZ / Float(nodeCount)
        
        print("Center of Mass computed at: (\(comX), \(comY), \(comZ))")
        
        var excitatoryIndex: Int32 = 0
        var minDistSq: Float = .greatestFiniteMagnitude
        
        print("Building spatial adjacencies via O(N) Spatial Hashing...")
        var offsets = [Int32]()
        var indices = [Int32]()
        
        for i in 0..<nodeCount {
            let node = nodes[i]
            
            // Find COM node
            let dx = node.x - comX
            let dy = node.y - comY
            let dz = node.z - comZ
            let distSq = dx*dx + dy*dy + dz*dz
            if distSq < minDistSq {
                minDistSq = distSq
                excitatoryIndex = Int32(i)
            }
            
            let voxel = Voxel(
                i: Int(floor(node.x / voxelSize)),
                j: Int(floor(node.y / voxelSize)),
                k: Int(floor(node.z / voxelSize))
            )
            
            var neighbors = [Int32]()
            
            for di in -1...1 {
                for dj in -1...1 {
                    for dk in -1...1 {
                        let filterVoxel = Voxel(i: voxel.i + di, j: voxel.j + dj, k: voxel.k + dk)
                        if let potentialNeighbors = grid[filterVoxel] {
                            for neighborIndex in potentialNeighbors {
                                if neighborIndex == i { continue }
                                let nNode = nodes[Int(neighborIndex)]
                                let dx = nNode.x - node.x
                                let dy = nNode.y - node.y
                                let dz = nNode.z - node.z
                                if (dx*dx + dy*dy + dz*dz) <= voxelSizeSq {
                                    neighbors.append(neighborIndex)
                                }
                            }
                        }
                    }
                }
            }
            // Populate CSR
            offsets.append(Int32(indices.count))
            indices.append(contentsOf: neighbors)
        }
        offsets.append(Int32(indices.count)) // Final offset representing total edges
        
        // Initialize States array
        var states = [Int8](repeating: 0, count: nodeCount)
        states[Int(excitatoryIndex)] = 1
        
        print("Source node \(excitatoryIndex) designated Phase 1 Excitatory.")
        print("Total Edges Identified: \(indices.count)")

        // Writing Binary Flat-files (output to same directory as input)
        let dirPath = dataDir

        let stateData = Data(bytes: states, count: states.count * MemoryLayout<Int8>.stride)
        try! stateData.write(to: URL(fileURLWithPath: "\(dirPath)/states.bin"))

        let offsetData = Data(bytes: offsets, count: offsets.count * MemoryLayout<Int32>.stride)
        try! offsetData.write(to: URL(fileURLWithPath: "\(dirPath)/offsets.bin"))

        let indicesData = Data(bytes: indices, count: indices.count * MemoryLayout<Int32>.stride)
        try! indicesData.write(to: URL(fileURLWithPath: "\(dirPath)/indices.bin"))

        print("Binary CSR serialized to \(dirPath).")
        
        // ====== O(1) WAVEFRONT BENCHMARK ======
        // Zero heap allocation. Pre-allocated ping-pong queues.
        // Only iterates active wavefront, never 0..<nodeCount.

        // Count isolated nodes
        var isolatedCount = 0
        for i in 0..<nodeCount {
            if offsets[i + 1] == offsets[i] { isolatedCount += 1 }
        }

        print("\n====== BENCHMARK: O(1) WAVEFRONT ======")
        print("Nodes: \(nodeCount), Edges: \(indices.count), Isolated: \(isolatedCount)")

        // Age array: 2=excited, 1=refractory, 0=resting
        var age = [Int8](repeating: 0, count: nodeCount)
        age[Int(excitatoryIndex)] = 2

        // Pre-allocated wavefront buffers (no heap alloc in hot loop)
        var frontA = [Int32](repeating: 0, count: nodeCount)
        var frontB = [Int32](repeating: 0, count: nodeCount)
        var refracBuf = [Int32](repeating: 0, count: nodeCount)

        frontA[0] = excitatoryIndex
        var currentCount = 1
        var nextCount = 0
        var refracCount = 0
        var frameCount = 0
        var useA = true

        print("Injection site: node \(excitatoryIndex) (degree \(offsets[Int(excitatoryIndex) + 1] - offsets[Int(excitatoryIndex)]))")

        let clock = ContinuousClock()
        let elapsed = clock.measure {
            while currentCount > 0 {
                nextCount = 0
                let prevRefracCount = refracCount
                refracCount = 0

                // Phase A: Recover previous refractory → resting
                for ri in 0..<prevRefracCount {
                    age[Int(refracBuf[ri])] = 0
                }

                // Phase B: Process excited wavefront only
                for qi in 0..<currentCount {
                    let i = Int(useA ? frontA[qi] : frontB[qi])
                    age[i] = 1  // excited → refractory
                    refracBuf[refracCount] = Int32(i)
                    refracCount += 1

                    let start = Int(offsets[i])
                    let end   = Int(offsets[i + 1])
                    for j in start..<end {
                        let nb = Int(indices[j])
                        if age[nb] == 0 {
                            age[nb] = 2
                            if useA {
                                frontB[nextCount] = Int32(nb)
                            } else {
                                frontA[nextCount] = Int32(nb)
                            }
                            nextCount += 1
                        }
                    }
                }

                frameCount += 1
                useA = !useA
                currentCount = nextCount
            }
        }

        // Final state distribution
        var cntExcited = 0; var cntRefrac = 0; var cntResting = 0
        for i in 0..<nodeCount {
            switch age[i] {
            case 2:  cntExcited += 1
            case 1:  cntRefrac += 1
            default: cntResting += 1
            }
        }
        let nodesReached = nodeCount - cntResting

        let (sec, atto) = elapsed.components
        let totalNs = Double(sec) * 1_000_000_000.0 + Double(atto) / 1_000_000_000.0
        let edgeCountD = Double(indices.count)
        let nsPerEdge = totalNs / edgeCountD
        let cyclesPerEdge = nsPerEdge * 3.49

        print("De Giorgi frames:       \(frameCount)")
        print("Nodes reached:          \(nodesReached) / \(nodeCount)")
        print("Total wall time:        \(String(format: "%.0f", totalNs)) ns (\(String(format: "%.3f", totalNs / 1_000_000.0)) ms)")
        print("ns per edge:            \(String(format: "%.2f", nsPerEdge))")
        print("Estimated cycles/edge:  \(String(format: "%.2f", cyclesPerEdge))")
        if cyclesPerEdge < 20.0 {
            print("Status: [VERIFIED] Target < 20 cycles achieved.")
        } else {
            print("Status: [WARNING] cycles/edge > 20. Target not met.")
        }
        print("Final: excited=\(cntExcited) refractory=\(cntRefrac) resting=\(cntResting)")
        print("=======================================")
    }
}
