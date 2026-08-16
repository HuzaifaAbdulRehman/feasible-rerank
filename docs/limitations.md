# Known limitations

Everything here is a weakness a reader should know about before trusting a number in this
repository. Several were found by an independent audit rather than by me, which is
recorded because the provenance is part of the evidence: the ones I found myself say
little about what I would have missed.

Items already fixed are in the commit history and the docs; this file is what is
**outstanding or intrinsic**.

---

## Fixed, but worth knowing the shape of

**The headline claim was wrong for 41 commits.** Until the apportionment baseline was
written, this project reported that no classical reranker could meet a group-exposure
budget of `tau <= 0.25` at any setting of its own hyperparameters. That measured a missing
remainder rule in `QuotaMMR`, not a property of classical reranking. `BalancedQuota` ties
the QUBO's reach on 8 of 8 benchmarks. The mechanism was described in `docs/findings.md`
the whole time and the baseline that exploits it was never implemented.

**The interaction matrix was not binary.** `scipy.sparse.csr_matrix` sums duplicate
coordinates, so repeat purchases inflated cells to a maximum of 122.0 in a matrix
documented as implicit feedback. This affected the similarity matrix, the popularity
tiers used as fairness groups, and the reported density (0.0067 -> 0.0049). Every number
predating the fix was computed on that matrix.

**Feasibility was certified on tuning data.** `reach` was computed from whether the
*tuning* half met the budget. Across the suite, 8 configurations certified feasible
violated the budget on held-out users and 7 flagged infeasible actually met it. Now
certified on the evaluation half (`feasible_eval`).

---

## Outstanding

**`qubo_tabu` stops on a wall clock, so its quality is hardware-dependent.**
`dwave-samplers`' TabuSampler defaults to a 20 ms timeout and every published number here
used that default. On a faster machine it searches more and may score better; on a slower
one, worse. `TabuSearch(num_restarts=50, timeout=None)` exists and is work-bounded, but
the results were not produced with it. Two identical protocol runs on this machine
differed by 0.0003 NDCG from this cause alone.

**`codecarbon` is not a power meter.** It estimates from CPU TDP and utilisation and its
readings are noise below ~0.1 s. For a project that reports energy, this is the central
measurement weakness, and no physical meter was available.

**The QPU experiment was never run.** D-Wave Leap is export-restricted in the region this
was developed in. `qubo_rerank/solvers/quantum.py` is written and marked clearly as
never executed. No number in this repository comes from quantum hardware.

**Catalogue-level metrics are not comparable across splits.** Gini and catalogue coverage
inside `protocol.py` are computed over half the users, so they cannot be read against
whole-benchmark figures. Documented in `split_users`, not flagged at each table.

**`exposure_parity` is both the optimisation target and the headline evaluation metric.**
That is close to tautological. The independent measures (Gini, IIF, AIF, IBO) are reported
alongside and do **not** show a QUBO advantage; the honest reading is that the QUBO wins
the axis it optimises and ties or loses elsewhere.

**`FeasibleAnnealing` starts warm, the other QUBO solvers start cold.** `warm_start=True`
seeds one restart from greedy top-k, so it cannot do worse than greedy on the search
objective. This is an asymmetry *inside* the QUBO family. It is conservative with respect
to the conclusions drawn -- `qubo_feasible` still loses to `qubo_tabu` -- but it is not a
like-for-like comparison between those two.

**The budget-sensitivity experiment is synthetic-only.** `experiments/sensitivity.py` runs
on `barrier_instance`, a constructed matrix, not on Amazon data. Section 2.1b of the
report presents its conclusion without restating that.

**The paired test compares a tuned QUBO against baselines at a default.** `paired.py` sets
`lam`/`mu` from the command line while the classical methods keep the config's
`mmr_lam=0.5`, a value the protocol never selects. A `--mmr-lam` flag now exists, but the
committed `amazon_lb_paired.csv` predates it. The parity result there is robust to this;
the +0.0012 NDCG result is not, and should not be quoted without rerunning symmetrically.

**Three benchmarks have no documented reproduction command.** `amazon_music`, `movielens`
and `amazon_lb_mf` are reported but their exact invocations are not written down in the
README the way the others are.

---

## Intrinsic

**Reranking, not retrieval.** The QUBO selects k=10 from n=200 candidates that a
classical model produced. It cannot recover an item the retrieval stage missed;
`candidate_hit_rate` reports that ceiling explicitly.

**O(n^2) in candidates.** The diversity term couples every pair, so the model is dense.
n=200 is tractable; a full catalogue is not, and Digital Music's 11,269 items already
force the sparse similarity path.

**k=10 over 4 groups throughout.** The advantage is known to depend on whether `k`
divides evenly by the number of groups -- at k=20 over 4 groups it is +0.0001. The
headline configuration is one of the favourable cases, deliberately and with the boundary
documented, but it is not a general result over k.

**Eight benchmarks, five catalogues.** `amazon_software`, `_vendor` and `_category` are
one catalogue under three group definitions; `amazon_lb` and `amazon_lb_mf` are one
catalogue under two retrieval models. "Eight benchmarks" means eight configurations, not
eight independent datasets.
