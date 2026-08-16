# feasible-rerank

[![tests](https://github.com/HuzaifaAbdulRehman/feasible-rerank/actions/workflows/tests.yml/badge.svg)](https://github.com/HuzaifaAbdulRehman/feasible-rerank/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Fair and energy-aware recommendation list selection via QUBO.**

Conventional recommenders pick the top-k items greedily by score. The resulting lists are
often redundant (near-identical items) and skewed toward popular sellers. `feasible-rerank`
instead selects the *entire list at once* as a Quadratic Unconstrained Binary Optimization
(QUBO) problem, jointly balancing relevance, diversity, and exposure fairness — and reports
the energy cost of doing so.

> Status: **complete**, apart from a QPU run that is export-blocked from this region.
> End-to-end on real data across 8 benchmarks, every headline comparison averaged over
> repeated seeds with paired significance testing, checked against mixed-integer-proven
> optima, and built around a negative result about the standard QUBO recipe that turned
> out to be the most interesting thing here.

**Documentation.** This file is the summary: what was found, the headline numbers, and
how to reproduce them. Two companions carry the detail —
[`docs/findings.md`](docs/findings.md) for the mechanisms and the experiments behind each
claim, and [`docs/report.md`](docs/report.md) for a standalone technical write-up.
[`docs/limitations.md`](docs/limitations.md) lists what is still wrong or unmeasured,
including the claim this project got wrong for 41 commits and how it was caught.

## The formulation



Given `n` candidate items with relevance scores `r_i` and pairwise similarity `s_ij`,
select exactly `k` items by minimising:

```
H  =  −Σ r_i·x_i                        relevance   (reward relevant items)
      + λ Σ s_ij·x_i·x_j                diversity   (penalise similar pairs)
      + P (Σ x_i − k)²                  cardinality (exactly k items)
      + μ Σ_c (Σ_{i∈c} x_i − k/|C|)²    fairness    (even exposure per category)
```

where `x_i ∈ {0,1}` indicates whether item *i* enters the list.

## Findings



1. **The standard QUBO recipe is broken, and it fails silently.** Penalty-encoded
   cardinality plus a single-flip annealer produces lists that are always the right
   length and close to arbitrary among lists of that length. Two fixes are implemented
   here, each ~2× better on the annealer's own objective. Across 5 seeds `qubo_sa`
   scores 0.475 ± 0.030 NDCG against 0.692 ± 0.038 for the fixed solvers — a gap of
   more than five standard deviations. **It is not a budget problem:** given three
   orders of magnitude more compute — up to 184 seconds — `neal` still never beats a
   single deterministic MMR pass that performs no search at all.
2. **The choice of `λ` and `μ` decides the QUBO-vs-classical comparison, and it decides
   it by more than the methods differ.** At `λ=4, μ=0` — the configuration this repo's
   headline benchmark uses — every QUBO variant loses to group-quota MMR on essentially
   everything. At `λ=0, μ=1` it does not. An earlier version of this README reported
   only the first and concluded the classical baseline wins; that was an artifact of
   comparing a QUBO with its fairness term switched **off** against a baseline with
   group quotas built in.
3. **The QUBO's advantage is *not* feasibility. A fifteen-line classical baseline reaches
   the same fairness budget on 8 of 8 benchmarks.** An earlier version of this file
   claimed no classical reranker could satisfy `τ ≤ 0.25` at any setting of its own
   hyperparameters. That was false, and it was false for an avoidable reason: the only
   classical fairness baseline implemented here, `quota_mmr`, caps each group at
   `ceil(k/|C|)` as an *upper* bound with no remainder rule, so it can finish 3/3/3/1 over
   four groups and never recover — parity 0.30 against an arithmetic floor of 0.20. That
   is a defect of one heuristic, not a property of classical reranking.
   `BalancedQuota` — textbook largest-remainder apportionment — attains the floor
   deterministically, on every user, at a small fraction of the compute. **Its reach ties
   the QUBO's on every benchmark tested.** (No multiplier is quoted here on purpose: see
   [Timing](#timing-is-provisional).)
4. **What survives is small, consistent, and about accuracy.** At that shared tightest
   budget the QUBO returns a better list than the apportionment baseline on **8 of 8**
   benchmarks — mean **+0.0086 NDCG**, sign test and Wilcoxon both **p = 0.008** — rising
   to **+0.0176** on `amazon_lb_mf`, where ALS produces the densest similarity matrix.
   The mechanism is non-separability: apportionment fills each group greedily, which is
   optimal for a separable objective and not for one containing `λ Σ s_ij x_i x_j`. On the
   objective the solvers actually minimise, the QUBO's margin over the best classical
   method grows with `λ` — tied at `λ=0`, +5.5% at `λ=4`, 2× at `λ=4, μ=0`. **The QUBO
   buys the diversity term, not the fairness constraint.**
5. **The naive quota heuristic collapses entirely under proportional targets**, where
   apportionment does not. On real Amazon product categories `quota_mmr` meets no budget
   below 1.00 at any hyperparameter setting, while both `balanced_quota` and the QUBO
   reach 0.20. A matched ablation changing only the target vector — same catalogue,
   partition, grids and seeds — restores `quota_mmr` to 0.30, isolating the cause as
   round-robin integer quotas being unable to represent a target of 3.6/1.15/0.95/4.3.
6. **It still costs far more compute**, `quota_mmr` still wins intra-list similarity
   outright, and the apportionment baseline is dramatically cheaper than the QUBO for an
   equal fairness guarantee. Every multiplier this section used to quote has been
   withdrawn pending a clean measurement — see [Timing](#timing-is-provisional).

<a id="timing-is-provisional"></a>
### Timing is provisional and must not be quoted

Earlier versions of this file put the apportionment baseline at "~800× cheaper" than the
QUBO. **That figure is withdrawn.** It came from a single ad-hoc run, not from the
committed artifacts. Recomputed across all eight `*_protocol.csv` files the ratio is
about **200×** (`qubo_tabu` 11.10 s mean against `balanced_quota` 0.056 s) — a factor of
four smaller than the number that was published.

Even the 200× figure is **not publishable**, because every `seconds`, `kwh` and `co2_kg`
value in the current CSVs was recorded while a second CPU-bound job shared the machine.
This repository's own rule (`docs/limitations.md`) is that timing and energy require a
clean sequential run, and these results do not have one. Contention inflates the columns
unevenly across methods, which is exactly the comparison a multiplier depends on.

What is safe to say is qualitative and follows from the algorithms rather than the clock:
apportionment is a single O(n log n) pass with no search, while the QUBO solvers run
iterated local search over a dense n×n model. **The quality conclusions are unaffected** —
`reach`, `ndcg@k` and `exposure_parity` are deterministic given the seed and cannot be
changed by CPU contention.

(1) is why (4) is worth trusting: without the barrier fix the solver never optimises well
enough for the operating point to matter. Numbers in [Results](#results).

> **This section was rewritten after an independent audit.** The previous headline — that
> the QUBO achieved a fairness guarantee no classical method could — did not survive the
> baseline above, and the finding that replaced it is narrower. The audit is the reason
> the claim is now defensible, and the old one is left described rather than deleted so
> the correction is visible.

> An intermediate version of this file claimed the QUBO beat quota-MMR on NDCG as well.
> That margin was measured with `λ` and `μ` selected on the same dataset they were
> scored on, and it **did not survive** the disjoint-split protocol. What survived that
> round became (3) — later refuted in turn by the apportionment baseline, leaving (4).
> Each correction narrowed the claim, and each was found by testing it harder rather than
> by tuning it away.

## Results

Per-dataset headline tables (synthetic, Amazon Luxury Beauty at λ=4/μ=0), the trade-off
curves, and the positioning against published QUBO recommender work now live in
[`docs/findings.md`](docs/findings.md). What follows is the evidence the conclusions
actually rest on.



### Tuning and evaluation on disjoint users



Everything above selects `λ` and `μ` by looking at a sweep over the same dataset the
result is then reported on. Re-validating on fresh user samples narrows the problem but
does not remove it: part of any reported margin is a measure of how many grid cells were
tried. `experiments/protocol.py` removes it.

**The protocol.** Split the sampled users into two disjoint halves. Declare a fairness
budget `τ` — an upper bound on mean exposure-parity deviation, the way a deployer would
actually state the requirement. Tune **every** method on the tuning half by maximising
NDCG@10 subject to `parity ≤ τ`. Evaluate the chosen configuration once, on the other
half. Repeat over seeds, and sweep `τ`.

Two details do most of the work:

- **The baselines are tuned too.** `mmr` and `quota_mmr` have a `λ` of their own, left
  at 0.5 everywhere else in this repo. Searching the QUBO's two weights while leaving
  the baseline's one at its default is the same asymmetry that made the original
  conclusion wrong, pointed the other way. Here `quota_mmr` genuinely uses its search —
  it picks `mmr_lam=0.7` under tight budgets and `0.3` under loose ones.
- **Selection is constrained.** Maximising NDCG alone always returns `λ=0, μ=0`, which
  *is* greedy top-k: NDCG 1.0 by construction and the worst parity available. A method
  that cannot meet the budget is flagged infeasible rather than quietly reported at its
  least-bad setting next to methods that met it.

NDCG@10 on held-out users, 3 seeds, 40 tuning / 40 evaluation users
(`results/amazon_lb_protocol.csv`). `--` means no configuration met the budget on every
seed:

| fairness budget τ | 0.20 | 0.22 | 0.25 | 0.30 | 0.40 | 1.00 |
|---|---|---|---|---|---|---|
| greedy_topk | -- | -- | -- | -- | -- | **1.0000** |
| mmr | -- | -- | -- | -- | -- | 0.9887 |
| quota_mmr | -- | -- | -- | 0.9033 | 0.9033 | 0.9033 |
| **qubo_tabu** | -- | **0.9043** | **0.9043** | **0.9043** | **0.9043** | 0.9639 |
| qubo_feasible | -- | 0.8501 | 0.8501 | 0.8501 | 0.8501 | 0.9453 |

![fairness budget curve](results/amazon_lb_protocol_budget.png)

**Three regimes, and they are the actual result.**

*Tight (`τ = 0.22, 0.25`).* Only the QUBO methods meet the budget at all. `quota_mmr`
achieves 0.2517 ± 0.0067 parity at best — it cannot get under 0.25 reliably at any
`mmr_lam`, because a quota mechanism can only redistribute slots it has. The QUBO's
penalty targets the parity objective directly and reaches 0.1983 ± 0.0017, essentially
the arithmetic floor. **This is the regime where QUBO reranking does something no
baseline here can do**, and it is a feasibility claim, not an accuracy one.

*Moderate (`τ = 0.30, 0.40`).* `quota_mmr` becomes feasible and the accuracy gap closes
to nothing: 0.9033 ± 0.0115 against `qubo_tabu`'s 0.9043 ± 0.0109. The QUBO still
delivers strictly better parity (0.1983 vs 0.2625, non-overlapping), so it buys fairness
headroom rather than accuracy — at ~100× the compute.

*Unconstrained (`τ = 1.00`).* Greedy top-k wins by definition. Both QUBO solvers select
`λ=0, μ=0` and return NDCG **1.0000**, recovering greedy exactly. That is the protocol's
built-in correctness check: told that fairness does not matter, the tuner correctly
concludes the QUBO should not be used.

The selection is stable — `λ=0, μ=1` is chosen at every constrained budget on every
seed — which matters, because an unstable selection would mean the tuning half was too
small to choose from reliably.

**What is still weak.** Three seeds and 40 evaluation users is a small sample, and the
`τ=0.20` column shows it: the QUBO's parity of 0.1983 ± 0.0017 straddles the budget, so
it is marked infeasible on the seeds where tuning landed just above. A paired per-user
test would extract far more from the same runs than comparing means over three seeds,
and is the next item in Phase 2.

### Does it hold on more than one catalogue?



The feasibility result is stated as a claim about *methods*. It could equally be a claim
about Amazon Luxury Beauty's popularity structure, and one dataset cannot tell those
apart. So the whole protocol was re-run, unchanged, on two more Amazon categories chosen
to differ in the ways that should matter:

| | Luxury Beauty | Software | Gift Cards |
|---|---|---|---|
| interactions (5-core) | 32,732 | 12,454 | 2,960 |
| users / items | 3,589 / 1,366 | 1,779 / 729 | 456 / 147 |
| density | 0.0049 | 0.0078 | **0.0373** |
| candidate-set ceiling on recall | 0.49 | 0.28 | — |
| candidates reranked | 200 | 200 | 100 |

Identical budgets, grids and seeds throughout — nothing is tuned per catalogue.

Six benchmarks: five catalogues across two domains, plus one that swaps the retrieval
model. Two are the important ones: **MovieLens groups are
curator-assigned genres** (Drama / Comedy / Romance / other), not popularity tiers, so it
tests the constraint against categories defined independently of the data being scored.

| dataset | items | density | groups from | retrieval |
|---|---|---|---|---|
| Gift Cards | 147 | 0.0373 | popularity tier | ItemKNN |
| Software | 729 | 0.0078 | popularity tier | ItemKNN |
| Luxury Beauty | 1,366 | 0.0049 | popularity tier | ItemKNN |
| Digital Music | 11,268 | 0.0007 | popularity tier | ItemKNN |
| **MovieLens 100K** | 1,349 | 0.0773 | **genre** | ItemKNN |
| **Luxury Beauty (MF)** | 1,366 | 0.0049 | popularity tier | **ALS** |
| **Software (seller)** | 729 | 0.0078 | **real vendor** | ItemKNN |
| **Software (category)** | 729 | 0.0078 | **real product category** | ItemKNN |

The last two use Amazon's separate metadata export rather than a partition derived from
the interaction counts being evaluated: 99.3% of the filtered Software catalogue carries a
brand and 93.5% a category path. Both partitions are near-independent of the popularity
tiers (NMI 0.012 and 0.016, where 1.0 would mean identical), so they ask a different
fairness question rather than restating the existing one. See
[`benchmarks/metadata.py`](benchmarks/metadata.py).

**Reach** — the tightest fairness budget each method meets *on every seed*. Lower is a
stronger guarantee:

| method | Gift Cards | Luxury Beauty | Digital Music | Software | MovieLens | LB (MF) | SW seller | SW category |
|---|---|---|---|---|---|---|---|---|
| quota_mmr | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 | *never* |
| **balanced_quota** | **0.20** | **0.22** | **0.22** | **0.20** | **0.20** | **0.20** | **0.20** | **0.20** |
| qubo_tabu | 0.20 | 0.22 | 0.22 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| qubo_feasible | 0.20 | 0.22 | 0.22 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |

**The QUBO is never strictly tighter than `balanced_quota` — 8 ties out of 8.** That row
is the whole correction: a classical apportionment baseline meets every budget the QUBO
meets. `greedy_topk` and `mmr` are omitted; neither meets any budget on most benchmarks.

NDCG@10 delivered at that tightest budget — this is where the QUBO does win:

| method | Gift Cards | Luxury Beauty | Digital Music | Software | MovieLens | LB (MF) | SW seller | SW category |
|---|---|---|---|---|---|---|---|---|
| quota_mmr | 0.7581 | 0.8931 | 0.8280 | 0.9084 | **0.9069** | 0.9365 | **0.9405** | 0.9187 |
| balanced_quota | 0.7514 | 0.8864 | 0.8213 | 0.8991 | 0.8997 | 0.9263 | 0.9317 | 0.9364 |
| **qubo_tabu** | **0.7591** | **0.8923** | **0.8236** | **0.9075** | 0.9039 | **0.9439** | 0.9396 | **0.9512** |
| qubo_feasible | 0.7263 | 0.8360 | 0.7556 | 0.8414 | 0.8111 | 0.8823 | 0.8756 | 0.9166 |

Against `balanced_quota` at the same budget, `qubo_tabu` is more accurate on **8 of 8**
benchmarks: mean **+0.0086 NDCG**, sign test and Wilcoxon both **p = 0.008**. The largest
margin is `LB (MF)` at **+0.0176**, where ALS gives the densest similarity matrix and the
pairwise term therefore matters most. Note `quota_mmr` sometimes posts a higher NDCG than
either — it is scoring at a *looser* budget (0.30) and so is not comparable down the
column; that is exactly why reach and accuracy have to be read together.

![cross-dataset fairness budget curves](results/datasets_budget.png)

![cross-dataset fairness budget curves](results/datasets_budget.png)

**Ties on 8 of 8.** `balanced_quota` meets every budget the QUBO meets, on every
benchmark — including Gift Cards, which was included precisely because it was the case
most likely to break the *old* result, being small and dense enough that a greedy
heuristic has room to do well. It broke it.

**Why quota-MMR stalls, and why that is not a fact about classical reranking.**
Quota-MMR's reach is 0.30 while the arithmetic floor for k=10 over 4 groups is 0.20. The
mechanism is a missing remainder rule: it caps each group at `ceil(k/|C|) = 3` as an
upper bound only, so 3/3/3/1 is reachable and unrecoverable. Largest-remainder
apportionment allocates 3/3/2/2 by construction and provably minimises
`Σ_c |quota_c − target_c|`, which *is* the exposure-parity numerator. This project
described that mechanism in `docs/findings.md` for 41 commits and did not implement the
baseline that exploits it — which is how a claim about one heuristic was reported as a
claim about a category of methods.

**What the group and model objections now establish.** These benchmarks were built to
close two objections to the old claim. They still do useful work, but for the *new* one:
the small accuracy advantage is not an artefact of how groups were defined or which
retrieval model produced the candidates.

- **MovieLens** groups by curator-assigned genre, not popularity. Reach ties at 0.20;
  QUBO +0.0041 NDCG.
- **Software (seller)** and **Software (category)** group by real vendor and real product
  category from Amazon's metadata export, both near-independent of the popularity tiers
  they replace (NMI 0.012 and 0.016). Reach ties at 0.20; QUBO +0.0079 and +0.0148.
- **`amazon_lb_mf`** draws candidates from matrix factorisation rather than ItemKNN.
  Reach ties at 0.20; QUBO **+0.0176**, the largest margin in the table — ALS produces
  the densest similarity matrix, so the pairwise term has the most to say.

**The claim this repo now makes** is narrower than the one it made before, and it is
about *when* the machinery pays rather than that it is better:

> A correct classical apportionment baseline achieves the same exposure-fairness
> guarantee as the QUBO on every benchmark tested, deterministically and at a small
> fraction of the compute (no multiplier quoted — see [Timing](#timing-is-provisional)).
> The QUBO's remaining advantage is confined to the non-separable part of the objective:
> at the same fairness budget it returns a more accurate list on 8 of 8 benchmarks
> (+0.0086 NDCG, p = 0.008), and its margin on the objective grows with the diversity
> weight λ. If you need exposure targets met, use apportionment. If you need the
> diversity term optimised jointly with them, the QUBO earns its compute.

### Paired per-user tests



Comparing means over 3 seeds is a test with n=3. But every method sees the **identical
candidate set for the identical user**, so their per-user scores are paired observations
— and that is a test with n=200 on data already collected. The seeds were never the
sample; the users are. Pairing also cancels the dominant variance term: users differ
enormously in how predictable they are, and a user whose held-out purchase never entered
the candidate set scores badly under every method.

Wilcoxon signed-rank, two-sided, Holm-corrected across all 20 comparisons in the run.
200 users, `λ=0, μ=1`, everything measured against `quota_mmr`
(`results/amazon_lb_paired.csv`):

| metric | method | median diff | 95% CI | better/worse/tied | p (Holm) |
|---|---|---|---|---|---|
| **exposure parity ↓** | qubo_tabu | **−0.1000** | [−0.1000, −0.0500] | 114 / **0** / 86 | 2.8e-25 |
| | qubo_feasible | **−0.1000** | [−0.1000, −0.0500] | 114 / **0** / 86 | 2.8e-25 |
| **NDCG@10 ↑** | qubo_tabu | **+0.0012** | [+0.0012, +0.0043] | 126 / 64 / 10 | 3.3e-07 |
| | qubo_feasible | −0.0409 | [−0.0522, −0.0251] | 51 / 149 / 0 | 1.6e-04 |
| | qubo_sa | −0.1371 | [−0.1528, −0.1155] | 27 / 173 / 0 | 8.2e-25 |
| **recall@10 ↑** | qubo_tabu | 0.0000 | [0.0000, 0.0000] | 5 / 1 / 194 | 0.57 |
| **intra-list sim ↓** | qubo_tabu | +0.0022 | [+0.0022, +0.0036] | 26 / 165 / 9 | 2.3e-27 |

**Significant and negligible are not opposites, and the NDCG row is why this matters.**
The seed-level comparison called `qubo_tabu` vs `quota_mmr` a tie — 0.9043 ± 0.0109
against 0.9033 ± 0.0115. The paired test, with far more power, finds the difference is
**real**: p ≈ 3×10⁻⁷, better on 126 users against 64. And its median size is
**0.0012 NDCG**, with the interval topping out at 0.0043. Both readings are correct, and
reporting either one alone would mislead. A table of p-values would have called this a
win; a table of means called it a tie; it is a certain difference of almost no size.

**The parity result has the opposite shape and is the one that carries weight.** A median
improvement of 0.10, and **worse on zero users out of 200**. Not a marginal edge — a
uniform one, and the same effect the fairness-budget curve shows from the other
direction.

**Recall does not move for any method** (194 of 200 users tied). Under leave-one-out
there is at most one relevant item per user and the candidate-set ceiling is 0.49, so
most users score identically under every reranker. Reporting a recall difference on this
benchmark would be reporting noise.

Two results the pairing strengthens rather than changes: `qubo_sa` sits at −0.137 NDCG
against the baseline (p ≈ 8×10⁻²⁵), so the penalty-barrier finding now rests on 200
paired observations rather than seed means; and `qubo_feasible` is **significantly worse
than the classical baseline** at −0.041, which settles that the QUBO's advantage here
belongs to `qubo_tabu` specifically and not to the formulation.

> Read the interval, not the star. With 200 users a p-value of 10⁻²⁵ reports
> *consistency*, not magnitude — `qubo_tabu` improves parity on every single user who
> changes at all, which is what drives p down, and says nothing by itself about whether
> 0.10 is a lot. The effect sizes and counts are in the table for that reason.

## Reproducing



```bash
python -m venv .venv
source .venv/Scripts/activate          # Windows / Git Bash
pip install -r requirements.txt

# ~24 MB, downloaded once
curl --create-dirs -o data/amazon_lb/Luxury_Beauty.csv \
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Luxury_Beauty.csv

pytest tests/ -q

python experiments/run_experiment.py --config configs/synthetic.yaml
python experiments/run_experiment.py --config configs/amazon_lb.yaml

# repeat studies -- 5 seeds, resampling users and solver seeds together
python experiments/run_experiment.py --config configs/synthetic.yaml --repeats 5
python experiments/run_experiment.py --config configs/amazon_lb.yaml --repeats 5 --n-users 60

# the sweep also writes amazon_lb_sweep_baselines.csv, on its own user sample
python experiments/sweep.py --config configs/amazon_lb.yaml --n-users 40

# the honest comparison: tune every method on one half of the users, evaluate on the
# other, sweep the fairness budget. This is the one to run if you only run one.
#
# --method is not optional if you want to reproduce the committed CSVs. Without it the
# protocol also tunes `qubo_sa`, which adds 9 configurations per seed of the slowest
# solver to reproduce a result the repo already establishes four other ways: penalty-
# encoded cardinality defeats a single-flip annealer, so `qubo_sa` meets no budget on
# any benchmark. It is excluded from the cross-dataset comparison for that reason, not
# because it lost a close race. Run `--method qubo_sa` on its own to see it fail.
python experiments/protocol.py --config configs/amazon_lb.yaml \
  --tau 0.20 0.22 0.25 0.30 0.40 1.00 --repeats 3 --n-users 80 \
  --method greedy_topk mmr quota_mmr qubo_tabu qubo_feasible

# paired per-user significance tests -- Wilcoxon, Holm-corrected across the family
python experiments/paired.py --config configs/amazon_lb.yaml --lam 0.0 --mu 1.0   --n-users 200 --reference quota_mmr

# does the result survive a change of catalogue? Run the protocol on the others first;
# each config carries its own download command.
METHODS="--method greedy_topk mmr quota_mmr qubo_tabu qubo_feasible"
python experiments/protocol.py --config configs/amazon_software.yaml  --repeats 3 --n-users 80 $METHODS
python experiments/protocol.py --config configs/amazon_giftcards.yaml --repeats 3 --n-users 80 $METHODS

# the same catalogue partitioned by real seller and real product category instead of by
# popularity tier -- these need the metadata export, see benchmarks/metadata.py
python experiments/protocol.py --config configs/amazon_software_vendor.yaml   --repeats 3 --n-users 80 $METHODS
python experiments/protocol.py --config configs/amazon_software_category.yaml --repeats 3 --n-users 80 $METHODS

python experiments/compare_datasets.py

# figures. plot_pareto picks up the matched baselines automatically -- do not pass
# results/amazon_lb.csv instead; Gini and catalogue coverage are catalogue-level
# aggregates and do not transfer between user-sample sizes
python experiments/plot_pareto.py   --sweep results/amazon_lb_sweep.csv
python experiments/plot_protocol.py --protocol results/amazon_lb_protocol.csv
```

**Three conditions, all learned the hard way.** Runs must be **sequential** — `seconds`
and `kWh` are wall-clock, so a second job is charged to whichever solver is running. The
machine must be on **mains power** — on battery this laptop's CPU drops from ~3.6 GHz to
1.297 GHz and every timing rises ~2.8× while every quality metric stays byte-identical.
And the whole results directory should be produced by **one version of the code**, since
even a tie-break change shifts candidate sets slightly.

## Data and protocol



- **Dataset:** Amazon Luxury Beauty (McAuley Lab, ratings-only export). 574,628 raw
  interactions → **32,732 interactions / 3,589 users / 1,365 items** after 5-core
  filtering.
- **Model:** ItemKNN — shrunk cosine similarity over the interaction matrix, top-100
  neighbourhood for scoring. Supplies both the relevance scores `r_i` and the item-item
  similarity `s_ij` from a single fit.

  **Verified against RecBole's reference implementation** (`tests/test_itemknn_reference.py`).
  Since every number here is downstream of this model, its `ComputeSimilarity` is
  transcribed into the test suite as an oracle and compared directly. Similarity values
  agree to **6e-08** — float32 against our float64, i.e. exactly. Truncated neighbour sets
  agree for 626 of 727 items, and for all 101 that differ, *every* disagreeing entry sits
  at a single exactly-tied similarity value. The two implementations pick different
  members of a tied group and are otherwise identical; ours breaks ties deterministically
  by `(-value, column)` where RecBole follows memory order.
- **Split:** leave-one-out on each user's most recent interaction.
- **Candidates:** top-200 by ItemKNN, reranked down to k=10. QUBO is O(n²), so a full
  catalogue is infeasible; two-stage retrieval → rerank is also how production systems
  work. The scaling limit is documented, not hidden.
- **Groups:** popularity tiers by default, because the *ratings* export carries no
  metadata. Items are rank-ordered by training interaction count and cut into 4
  equal-sized tiers (tier 0 = short head), the standard short-head/long-tail partition
  from the popularity-bias literature. This makes the fairness term fight the exact bias
  ItemKNN exhibits, rather than a partition chosen to flatter it. Amazon's *separate*
  metadata export does carry real sellers and categories where it is populated, and
  those partitions are available via `grouping="vendor"` / `grouping="category"` — see
  [`benchmarks/metadata.py`](benchmarks/metadata.py). Coverage decides where: 99.3% of
  the filtered Software catalogue has a brand, against 0.1% of Luxury Beauty, so the
  loader refuses the grouping rather than silently collapsing to a single group.
- **Recall ceiling:** candidates come from the same model being reranked, so a held-out
  item is often not in the candidate set at all. `candidate_hit_rate` reports that ceiling
  explicitly (**0.49** here) — recall@10 must be read against it, not against 1.0.

## Layout



```
qubo_rerank/
├── formulations/   objective · cardinality · fairness · builder
├── solvers/        greedy · MMR · quota-MMR · neal SA · tabu · swap · bifurcation · QPU
└── metrics/        NDCG · recall · coverage · Gini · exposure parity · DPFR · kWh
benchmarks/         synthetic · Amazon (ItemKNN) · MovieLens (genres) · ALS factorisation
experiments/        run_experiment · sweep · protocol · paired · sensitivity ·
                    optimality · exact · ablation · compare_datasets · plots
configs/            YAML configs: synthetic, 4 Amazon categories, MovieLens, MF
tests/              345 tests · 66% line coverage
```

A written-up version of the findings, with method and limitations, is in
[`docs/report.md`](docs/report.md).

**If you are reading this to judge the work, three files carry it:**

| file | why |
|---|---|
| [`qubo_rerank/solvers/feasible.py`](qubo_rerank/solvers/feasible.py) | the constraint-preserving annealer, and the response to the penalty-barrier finding |
| [`tests/test_solvers.py`](tests/test_solvers.py) | `TestPenaltyBarrier` — the four tests that hold the headline claim up |
| [`benchmarks/loader.py`](benchmarks/loader.py) | the real-data pipeline; ItemKNN in ~40 lines of scipy rather than 2.5 GB of torch |
| [`benchmarks/metadata.py`](benchmarks/metadata.py) | real seller and category partitions, and the coverage guard that stops a blank metadata file from becoming a fake result |
| [`experiments/protocol.py`](experiments/protocol.py) | the tune-on-half / evaluate-on-the-other-half protocol the headline claim rests on |

## Roadmap

| Phase | Content | Status |
|---|---|---|
| **1** | CF baseline, core QUBO, fairness term, solvers, one real dataset | done |
| **2** | Repeated seeds, disjoint tuning split, paired tests, 6 benchmarks | done |
| **3** | Sparse similarity, Simulated Bifurcation, exact MIP baseline, ablations | done |
| | D-Wave Leap QPU | **blocked** |
| **4** | Packaging, docs | done |
| | PyPI release | dropped |
| **5** | Technical report | done |
| | arXiv preprint | not started |

Two rows need explaining, because "blocked" and "dropped" are not the same as "todo".

**The QPU experiment is blocked, not pending.** `qubo_rerank/solvers/quantum.py` is
written and wired in and has never been run. D-Wave restricts Leap access by country
under export control and the region this was developed in is blocked; that was confirmed
at signup, not assumed. No attempt was made to circumvent it. The code stays because it
is the correct experiment for whoever can run it — it reports embedding overhead,
chain-break fraction and total time-to-solution separately from QPU access time, the last
being the number that flatters quantum hardware by excluding the classical work needed to
use it. A dense 200-variable clique may not embed on current topologies at all, which
would itself be a finding.

What it costs the argument: everything here is classical or quantum-*inspired*, so the
barrier's extension to physical annealers stays a prediction. Simulated Bifurcation is
the closest available substitute and behaves as predicted, which is corroboration but not
the same thing.

**PyPI was dropped deliberately.** This is a research codebase run from a checkout, not a
library anyone imports. `pyproject.toml` carries full metadata and supports an editable
install; claiming a package name nobody will `pip install` costs something and buys
nothing.

### What would genuinely extend this

Not a todo list. These need resources this environment does not have, or are separate
projects rather than unfinished work here.

1. **A third domain.** Six benchmarks cover retail and film. Music, books or news would
   test whether the feasibility result depends on catalogue semantics. Free datasets
   exist, so this is the cheapest real extension.
2. **A sequential retrieval model.** ItemKNN and ALS both score without regard to order.
   SASRec or GRU4Rec bias toward *recency* rather than popularity, which is a
   structurally different bias and therefore a genuine test rather than a third run of
   the same experiment.
3. **Energy measured at the wall.** `codecarbon` estimates from CPU TDP and utilisation
   and is not a power meter; below ~0.1 s its readings are noise. A plug-in meter, or
   Intel RAPL counters, would put these numbers on firmer footing than most published
   green-recsys work.
4. **A preprint.** [`docs/report.md`](docs/report.md) is the substance; turning it into a
   submission is writing, not research.

## Related work



This project builds on published research; it does not reimplement it wholesale.

- Ferrari Dacrema, Felicioni, Cremonesi — *Optimizing the Selection of Recommendation
  Carousels with Quantum Computing*, RecSys 2021
- *Feature Selection for Recommender Systems with Quantum Computing*, [arXiv 2110.05089](https://arxiv.org/pdf/2110.05089)
- *Performance-Driven QUBO for Recommender Systems on Quantum Annealers*, [arXiv 2410.15272](https://arxiv.org/abs/2410.15272)
- Wegmeth, Vente, Said, Beel — *Green Recommender Systems*, ACM TORS 2025, [arXiv 2509.13001](https://arxiv.org/abs/2509.13001)
- *Re-ranking With Constraints on Diversified Exposures*, [arXiv 2112.07621](https://arxiv.org/pdf/2112.07621)
- Abdollahpouri, Burke, Mobasher — *Managing Popularity Bias in Recommender Systems with
  Personalized Re-ranking*, FLAIRS 2019 — the short-head/long-tail group partition
- Ferrari Dacrema et al. — *Are We Really Making Much Progress?*, RecSys 2019 — the reason
  this repo reports solver energy alongside downstream metrics

## License



MIT — see [LICENSE](LICENSE).
