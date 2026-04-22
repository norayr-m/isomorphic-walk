// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "engine",
    platforms: [.macOS(.v14)],
    products: [
        .library(
            name: "BFSLib",
            type: .dynamic,
            targets: ["BFSLib"]
        ),
    ],
    targets: [
        .executableTarget(name: "engine"),
        .executableTarget(name: "KowalskiCrush"),
        .executableTarget(name: "KowalskiCrushGPU"),
        .executableTarget(name: "IsoWalk"),
        .target(name: "BFSLib"),
        .testTarget(name: "engineTests", dependencies: ["engine"]),
    ],
    swiftLanguageModes: [.v6]
)
