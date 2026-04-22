.PHONY: build test bench clean

DYLIB := engine/swift/.build/arm64-apple-macosx/release/libBFSLib.dylib

build: $(DYLIB)

$(DYLIB): engine/swift/Sources/BFSLib/BFSLib.swift engine/swift/Package.swift
	cd engine/swift && swift build -c release --product BFSLib

test: build
	python3 engine/python/bfslib.py
	python3 examples/quickstart.py

bench: build
	@echo "==> Bench A: Dataloader Duel (SciPy + Python)"
	python3 experiments/2026-04-20_dataloader_duel/duel.py
	@echo
	@echo "==> Bench A: PyTorch Geometric"
	python3 experiments/2026-04-20_dataloader_duel/pyg_bench.py
	@echo
	@echo "==> Bench B: GAPBS roadNet-CA (requires download from snap.stanford.edu)"
	python3 experiments/2026-04-20_gapbs/gapbs_bench.py
	@echo
	@echo "==> Bench C: Asymptotic vs ProDy GNM (requires PDB CIFs in /tmp/validate_isomorphic/)"
	python3 experiments/2026-04-20_scaling_vs_prody/scaling_v2.py

clean:
	rm -rf engine/swift/.build engine/swift/.swiftpm engine/swift/Package.resolved
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
