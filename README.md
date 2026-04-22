# Isomorphic Walk

This is work in progress. Some of what we tried worked. Some of what we
tried did not. We document both because the point is honest signal under
pre-registered acceptance gates.

*This is an amateur engineering project. We are not HPC professionals and make no competitive claims. Errors likely.*

---

A zero-allocation Swift CSR breadth-first-search primitive on Apple
Silicon, exposed to Python via `ctypes`, applied to biomolecular
contact graphs. Two findings cleared pre-registered acceptance gates
with Gemini DeepThink as neutral arbiter; one algorithmic variant was
formally rejected under the same protocol.

## What works

### 1. Asymptotic scaling — ESTABLISHED

On a 313,236-atom HIV-1 capsid (PDB 3J3Q), the Swift BFS finishes a
full traversal in **6 milliseconds** on an M5 Max laptop, sweeping
4.27 million contact edges. The standard biomolecular normal-mode
library (ProDy GNM) cannot run on this system: the dense Kirchhoff
matrix requires 313,236² × 8 bytes ≈ **784 GB** of memory, well above
any commodity hardware. This is a mathematical-exclusion claim, not a
speed comparison. At scales where the incumbent tool fits, it produces
more biologically informative output. At scales where it does not fit,
the engine still does.

Measured at 4 nanoseconds per edge, ~250 MTEPS sustained on dense
biological contact graphs.

Reproduce: `experiments/2026-04-20_scaling_vs_prody/scaling_v2.py`.

### 2. Allosteric pathway recovery — ESTABLISHED

On a pre-registered cohort of 10 well-studied allosteric systems
(IGPS, PTP1B, β₂-adrenergic receptor, Hsp70, PKA, GlmS, ATCase,
tryptophan synthase, PFK, Abl kinase), BFS suboptimal-tube on an 8 Å
Cα Euclidean contact graph identifies published allosteric-pathway
residues **better than a Euclidean-cylinder baseline** in 6 of 7
evaluated systems. Three systems were pipeline-failures from
seed-residue parsing issues and were dropped without substitution per
the locked protocol. **Median ΔF1 = +0.0707.** Acceptance gate was
ΔF1 ≥ +0.05 AND ≥ 70 % wins; both cleared.

Ground-truth residue lists from Rivalta 2012 (IGPS), Wiesmann 2004
(PTP1B), Venkatakrishnan 2013 (β₂-AR), Zhuravleva 2012 (Hsp70),
Taylor 2012 (PKA), Lipscomb 2008 (ATCase), Schirmer & Evans 1990
(PFK), Azam 2008 (Abl).

Reproduce: `experiments/2026-04-20_allosteric_cohort_n10/cohort.py`.

## What does not work

### Spectral snap walk against a forced target — REJECTED

A second pre-registered hypothesis tested whether a discrete,
deterministic walk along the Fiedler eigenvector (λ₂) of the contact
graph could predict per-residue mobility and a whole-system
edit-distance proxy when given two known conformers as start and end
states. The protocol was locked with DeepThink before any data were
collected. The N = 10 confirmatory cohort spanned classical paired
conformers (HIV-1 protease, Klenow polymerase, lactoferrin, yeast
hexokinase, glutamine binding protein, ribose binding protein,
citrate synthase, T4 lysozyme, enolase, GroEL subunit).

**Both hypotheses rejected:**

| | Result | Threshold | Verdict |
|---|---:|---:|---|
| H₁a: per-residue ρ(PMS, displacement), median across cohort | +0.146 | ≥ +0.35 ESTABLISH; < +0.20 REJECT | REJECT |
| H₁b: whole-system Spearman ρ(snap count, 1 − TM-score) | −0.358 | ≥ +0.65 ESTABLISH; ≤ +0.35 REJECT | REJECT |

Two systems went strongly negative on per-residue ρ: yeast hexokinase
(−0.49) and citrate synthase (−0.28). Both involve rotation/shear
rather than pure hinge motion. The arbiter's post-verdict diagnostic
called the algorithm a *rigid-body hinge detector* — the Fiedler
vector computes a single 1D minimum cut, which aligns cleanly with
bimodal hinge geometry but does not capture rotation or shear modes.

Per the pre-committed null clause, no retry with adjusted parameters,
no parameter sweeps, no cohort splits to rescue the claim. The result
is reported as the locked criteria say it is.

Reproduce: `experiments/2026-04-21_isowalk_spectral_snap/snap_walk.py`.

## What we tried (the gated experimental arc)

The chronology of pre-registered experiments documented above:

1. **Asymptotic scaling claim** — engineering benchmark against ProDy
   GNM across protein sizes from 200 atoms (adenylate kinase) up to
   313,236 atoms (HIV-1 capsid). ESTABLISHED.
2. **Allosteric pathway cohort N=10** — DT-locked protocol, BFS
   suboptimal tube vs Euclidean cylinder against published allosteric
   residue sets. ESTABLISHED.
3. **Spectral snap walk against a forced target** — DT-locked protocol,
   second pre-registered hypothesis class (per-residue PMS + whole-
   system snap count vs TM-score). REJECTED on both H₁a and H₁b.

Each result has a script in the corresponding `experiments/` directory
that reproduces the published numbers from the cited PDB inputs.

## What we are going to try again

Two open directions named for transparency. Neither is yet
pre-registered; if pursued, each will pass the same DeepThink-locked
protocol gates as above.

### Allosteric Fiedler zero-crossing

The arbiter's post-verdict analysis named the rejected snap walk a
hinge detector. That framing suggests a complementary signal to the
ESTABLISHED allosteric-pathway result: residues lying on a
communication path between two functional sites might cluster at the
**zero-crossing of the Fiedler vector** — the structural axis along
which the protein bends. The BFS-tube signal captures connectivity;
the Fiedler-axis signal would capture the mechanical bend along which
allosteric communication physically travels. If both signals
correlate independently with published allosteric residues on the
N=10 cohort, their intersection should be a sharper predictor than
either alone. A literature check on whether this specific spectral
framing is already established is the precondition before any
fresh pre-registration.

### Multi-mode spectral subspace

The arbiter's post-verdict analysis also identified that
1-dimensional Fiedler projection cannot capture rotation or shear,
but a subspace projection spanning λ₂ through λ₅ — still O(E) per
sparse Lanczos iteration, still within the engine's performance
envelope — might. The infrastructure (sparse iteration on Apple
Silicon unified memory) is ready; the question is whether a
subspace variant can recover the regime that the 1D variant lost.
Open hypothesis, not committed work.

## Repository layout

```
engine/
  swift/                  Swift package: BFSLib (C-callable dylib),
                          KowalskiCrush (CLI), KowalskiCrushGPU (Metal),
                          IsoWalk (experimental ternary-ratchet walk)
  python/bfslib.py        ~60-line ctypes wrapper, zero-copy via UMA
experiments/              Reproducible pipelines for each gated result
viewers/                  Three web-based NGL 3D viewers (HIV-1 capsid
                          scaling, allosteric pathway on PTP1B,
                          IsoWalk variant comparison)
examples/quickstart.py    Minimal usage example
LICENSE                   GPLv3
```

## Reproducing

Requirements: macOS 14+ on Apple Silicon, Swift 6+, Python 3.10+ with
`numpy`, `scipy`, `gemmi`, optionally `prody` (for the GNM scaling
comparison) and `tmtools` (for TM-score in the spectral snap walk
benchmark).

```
cd engine/swift
swift build -c release --product BFSLib
python3 examples/quickstart.py
```

Each experiment in `experiments/` is a self-contained script plus its
JSON outputs. Inputs are downloaded from RCSB at runtime; no large
binaries are committed.

## Pre-registration discipline

All acceptance-gated claims in this repository were pre-registered
with an external arbiter (Gemini 2.5 DeepThink) before the final
data were collected. Criteria, cohorts, and
thresholds were locked in advance. Numbers were returned verbatim
under a pre-committed null clause: if the locked criteria returned
REJECT, no retry with tweaked parameters, no reframing. The Spectral
Snap Walk REJECT documented above was accepted under this rule.

## License

GPLv3. See `LICENSE`.

## Acknowledgements

Pre-registration design, locked protocols, and post-verdict analysis
by Gemini 2.5 DeepThink. Architectural direction by the project's
lead. Engineering implementation by an AI agent under strict honesty
discipline (no retries, no reframing, raw numbers or rejection).
