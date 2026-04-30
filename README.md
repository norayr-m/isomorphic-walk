# Isomorphic Walk

A method for finding **allosteric pathways** in proteins — the chains of
residues that carry signals across a folded protein from one site to
another. Built from a fast graph-search engine on Apple Silicon, with
every claim pre-registered against a neutral arbiter before any data
were collected.

*This is an amateur engineering project. We are not HPC professionals
and make no competitive claims. Errors are likely.*

---

## What this is — for the reader who hasn't seen it before

Proteins are long molecular chains that fold into specific 3D shapes.
Many of the most important proteins in biology are **allosteric**: a
signal arrives at one place on the molecule (a hormone binding, a
nucleotide attaching, a covalent modification) and somehow propagates
to a second, distant place on the same molecule, where the actual
functional change happens. Most enzymes work this way. Most cell-surface
receptors work this way. Most molecular machines that consume ATP work
this way. The chain of residues that carries the signal between the
two sites is called the **allosteric pathway**, and identifying it
matters because those residues are the ones a biologist would mutate
to test mechanism, and a chemist would target with a small molecule
to modulate the protein's function.

This repository contains a method for predicting allosteric pathways
from a single static protein structure (the kind that crystallographers
deposit publicly). It works by treating the residues as nodes in a
graph and the close 3D contacts between them as edges, then running a
fast graph search to find the corridor of residues that lies between
two designated functional sites. The novelty is twofold: a graph-search
engine written natively for Apple Silicon laptops that handles
biological structures up to the size of a complete viral capsid (over
a quarter-million atoms), and an honesty discipline — every quantitative
claim was pre-registered with an external arbiter before the data were
collected, with the option to reject the method left explicitly open.

## Why allosteric pathways matter

Allostery is one of the central mechanisms of biological regulation.
Hemoglobin's oxygen affinity changes by a factor of 100 depending on
whether one of its four subunits is bound to oxygen — that's allostery.
Hexokinase changes shape after glucose binds, so that downstream
enzymes know substrate is available — that's allostery. The G-protein-
coupled receptors that mediate sight, smell, mood, and the action of
roughly a third of all approved drugs all work by allosteric coupling
between an external binding pocket and an internal G-protein-binding
surface — that's allostery. Identifying which residues actually
mediate the signal between the two sites is hard, because the residues
that *physically* lie between two functional sites are usually a much
larger set than the residues that *biologically* matter, and isolating
the latter from the former requires either decades of mutagenesis
work per protein or a computational method that can predict the right
residues from structure alone.

The standard biophysical computational tools for this kind of question
fall into two categories. **Molecular dynamics** simulates the protein
atom-by-atom under realistic physics; it's the gold standard but it's
expensive — a microsecond of simulation on a million-atom system takes
days on a GPU cluster. **Elastic network models** like the Gaussian
Network Model (GNM) treat the residue contacts as springs and solve a
matrix eigenvalue problem; they're cheap but they don't scale, because
the matrix is dense and grows quadratically with system size. At the
scale of a viral capsid (a few hundred thousand atoms), the dense
matrix doesn't fit in any commodity computer. This repository explores
a third path: discrete graph algorithms (breadth-first search and
sparse spectral methods) that scale linearly in the number of contacts
and run on a laptop. Whether the method actually predicts biology
correctly is the question; the engine just makes the question
testable at scales where the question couldn't previously be asked.

---

## Results — the gated experimental arc

All quantitative claims below were pre-registered with **Gemini 2.5
DeepThink** as a neutral arbiter before any data were collected. The
arbiter locked the protocol — graph construction parameters,
acceptance gates for ESTABLISH and REJECT, cohort composition,
dropped-system rules — and the numerical result was returned verbatim
under a pre-committed null clause: if the locked criteria returned
REJECT, no retry, no parameter sweeps, no reframing.

### What works

#### 1. Asymptotic scaling — ESTABLISHED

On a 313,236-atom HIV-1 capsid (PDB 3J3Q), the Swift breadth-first
search finishes a full traversal in **6 milliseconds** on an M5 Max
laptop, sweeping 4.27 million contact edges. The standard biomolecular
normal-mode library (ProDy GNM) cannot run on this system: the dense
Kirchhoff matrix requires 313,236² × 8 bytes ≈ **784 GB** of memory,
well above any commodity hardware. This is a *memory-exclusion* claim,
not a wall-clock comparison. At scales where the incumbent tool fits,
it produces more biologically informative output (vibrational modes,
not graph distances). At scales where it does not fit, the engine still
does. Sustained throughput is approximately 250 million traversed edges
per second, or 4 nanoseconds per edge.

Reproduce: `experiments/2026-04-20_scaling_vs_prody/scaling_v2.py`.

#### 2. Allosteric pathway recovery — ESTABLISHED

On a pre-registered cohort of 10 well-studied allosteric systems —
IGPS, PTP1B, β₂-adrenergic receptor, Hsp70, PKA, GlmS, ATCase,
tryptophan synthase, PFK, Abl kinase — a **BFS suboptimal-tube** on an
8 Å Cα Euclidean contact graph identifies published allosteric-pathway
residues **better than a Euclidean-cylinder baseline** in **6 of 7**
evaluated systems. Three systems were pipeline failures from
seed-residue parsing issues and were dropped without substitution per
the locked protocol. **Median ΔF1 = +0.0707**. The pre-registered
acceptance gate was ΔF1 ≥ +0.05 AND ≥ 70 % wins; both cleared.

The mechanism in plain words: the residues identified by the BFS-tube
are those that lie on a graph-shortest-path corridor between the two
seeds — the algorithm finds residues that follow the protein chain and
its physical contacts, not residues that fall on a straight line in
3D space. The 6-of-7 result says the contact network carries
information about the biological signal pathway that the geometric
straight-line baseline does not. The effect is consistent across
protein families (kinases, GPCRs, transporters, chaperones, synthases),
so it is unlikely to be a coincidence on any single system.

Ground-truth allosteric residue lists were taken from the primary
literature: Rivalta 2012 (IGPS), Wiesmann 2004 (PTP1B), Venkatakrishnan
2013 (β₂-AR), Zhuravleva 2012 (Hsp70), Taylor 2012 (PKA), Lipscomb
2008 (ATCase), Schirmer & Evans 1990 (PFK), Azam 2008 (Abl).

Reproduce: `experiments/2026-04-20_allosteric_cohort_n10/cohort.py`.

### What does not work

#### Spectral snap walk against a forced target — REJECTED

A second pre-registered hypothesis tested whether a discrete walk
along the **Fiedler eigenvector** (λ₂) of the contact-graph Laplacian
could predict per-residue mobility and a whole-system structural-
similarity score when given two known conformers as start and end
states. The protocol was locked with the arbiter before any data
were collected. The N = 10 confirmatory cohort spanned classical
paired conformers (HIV-1 protease, Klenow polymerase, lactoferrin,
yeast hexokinase, glutamine binding protein, ribose binding protein,
citrate synthase, T4 lysozyme, enolase, GroEL subunit).

Both pre-registered hypotheses rejected:

| | Result | Threshold | Verdict |
|---|---:|---:|---|
| H₁a: per-residue ρ(PMS, displacement), median across cohort | +0.146 | ≥ +0.35 ESTABLISH; < +0.20 REJECT | **REJECT** |
| H₁b: whole-system Spearman ρ(snap count, 1 − TM-score) | −0.358 | ≥ +0.65 ESTABLISH; ≤ +0.35 REJECT | **REJECT** |

Two systems went strongly negative on per-residue correlation: yeast
hexokinase (ρ = −0.49) and citrate synthase (ρ = −0.28). Both involve
rotation or shear motion, not pure hinge motion. The arbiter's
post-verdict diagnostic called the algorithm a *rigid-body hinge
detector* — the Fiedler vector cleanly identifies the axis between
two graph-disjoint subdomains, but rigid-body rotation has near-zero
internal Cα displacement, so the per-residue correlation vanishes
exactly where the algorithm has nothing to say.

Per the pre-committed null clause, no retry with adjusted parameters,
no parameter sweeps, no cohort splits to rescue the claim. The result
is reported as the locked criteria say it is.

Reproduce: `experiments/2026-04-21_isowalk_spectral_snap/snap_walk.py`.

### What we explored next

After the spectral snap walk REJECT, the arbiter named two open
directions. Both were tested as informal pilots (declared not counted,
no formal pre-registration, no public ESTABLISH/REJECT claim) to
inform whether either direction was worth committing a formal
pre-registration round. Both pilots used the same N = 10 cohort and
the same ground-truth allosteric residue lists as the established
BFS-tube result, so the comparison is apples-to-apples.

#### Fiedler zero-crossing — pilot did not warrant pre-registration

The arbiter proposed that residues at the *zero-crossing* of the
Fiedler vector (where the eigenvector flips sign across an edge) might
identify the mechanical bend axis along which allosteric signal
travels. The pilot tested both standalone Fiedler zero-crossings and
their intersection with the established BFS-tube. Standalone
zero-crossings beat the BFS-tube on only 3 of 9 evaluated systems
(median ΔF1 = −0.020). The intersection variant was borderline —
4 of 9 wins, median ΔF1 = −0.012, with two genuine large wins (PTP1B
+0.165, PFK +0.101) offset by one catastrophic structural failure
(Abl kinase, where the bend axis and the BFS path between the
A-loop and the αC-helix are graph-disjoint, producing a zero
intersection). Conclusion: the standalone zero-crossing is not worth
formal pre-registration; the intersection variant is precision-rich
but recall-poor and would require a precision-at-fixed-recall metric
rather than F1 to evaluate honestly.

Pilot script: `experiments/2026-04-29_fiedler_zero_crossing_pilot/`.

#### Multi-mode spectral subspace (λ₂ + λ₃) — pilot rejects the direction

The arbiter also proposed that a subspace spanning λ₂ through λ₅
might capture the rotation and shear modes that the 1D Fiedler
projection misses. The pilot tested the smallest extension toward
that idea — adding λ₃ alongside λ₂ — across three independent
formulations: zero-crossings on v₃ alone, the union of zero-crossings
on v₂ and v₃, and a 2D-subspace axis defined by the smallest
v₂² + v₃². All three produced median ΔF1 ≈ −0.17 versus the
BFS-tube, with only 2 of 9 wins each. Adding v₃ to the borderline
intersection variant *dilutes* it (median ΔF1 dropped from −0.012 to
−0.031). Conclusion: the multi-mode subspace direction does not
recover the regime that the 1D variant lost; walking out further to
λ₄ or λ₅ would be a fishing expedition. This direction is closed
without consuming a formal pre-registration round.

Pilot script: `experiments/2026-04-29_fiedler_lambda3_subspace_pilot/`.

---

## Repository layout

```
engine/
  swift/                  Swift package: BFSLib (C-callable dylib),
                          KowalskiCrush (CLI), KowalskiCrushGPU (Metal),
                          IsoWalk (experimental ternary-ratchet walk)
  python/bfslib.py        ~60-line ctypes wrapper, zero-copy via UMA
experiments/              Reproducible pipelines for each result
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
data were collected. Criteria, cohorts, and thresholds were locked
in advance. Numbers were returned verbatim under a pre-committed null
clause: if the locked criteria returned REJECT, no retry with tweaked
parameters, no reframing. The Spectral Snap Walk REJECT documented
above was accepted under this rule.

The two follow-up pilots (Fiedler zero-crossing and λ₃ multi-mode
subspace) were *not* formally pre-registered — they were exploratory
runs intended to inform whether the arbiter's named open directions
were worth a formal pre-registration round. They are reported here
with explicit pilot framing so the reader can see what was tested,
what was learned, and why no further formal commitment was made on
these specific variants.

## License

GPLv3. See `LICENSE`.

## Acknowledgements

Pre-registration design, locked protocols, and post-verdict analysis
by Gemini 2.5 DeepThink. Architectural direction by the project's
lead. Engineering implementation by an AI agent under strict honesty
discipline (no retries, no reframing, raw numbers or rejection).
