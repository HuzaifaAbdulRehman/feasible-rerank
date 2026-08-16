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
> End-to-end on real data across 6 benchmarks, every headline comparison averaged over
> repeated seeds with paired significance testing, checked against mixed-integer-proven
> optima, and built around a negative result about the standard QUBO recipe that turned
> out to be the most interesting thing here.

**Documentation.** This file is the summary: what was found, the headline numbers, and
how to reproduce them. Two companions carry the detail —
[`docs/findings.md`](docs/findings.md) for the mechanisms and the experiments behind each
claim, and [`docs/report.md`](docs/report.md) for a standalone technical write-up with
the limitations stated in one place.

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
3. **Under a protocol that tunes every method on one half of the users and scores it on
   the other, the QUBO's advantage is not accuracy — it is feasibility**, and it applies
   only when `k` does not divide evenly by the number of groups (at k=20 over 4 groups
   the advantage is +0.0001; at k=5 it is +0.2325). At a fairness
   requirement of `τ ≤ 0.25`, no classical baseline can satisfy the constraint at *any*
   setting of its own hyperparameters, while `qubo_tabu` can and still returns NDCG
   0.904. Loosen the requirement to `τ ≥ 0.30` and quota-MMR becomes feasible and the
   accuracy gap closes: 0.9033 ± 0.0115 against 0.9043 ± 0.0109. A paired test over 200
   users finds that residual gap is *real* — p ≈ 2×10⁻⁷ — and that its median size is
   **0.0012 NDCG**. Certain, and negligible.
4. **That feasibility result holds on 6 of 6 benchmarks**, spanning 77× in catalogue
   size and 63× in density, across two domains, two group definitions and **two
   retrieval models**. On MovieLens the groups are curator-assigned genres rather than
   popularity tiers; on `amazon_lb_mf` the candidates come from matrix factorisation
   rather than ItemKNN. Together those close the two obvious objections — that the QUBO
   only wins because the groups are defined by the signal it exploits, and that it only
   wins because of ItemKNN's particular bias.
5. **It still costs ~100× the compute** (16.1 s against 0.16), and `quota_mmr` still
   wins intra-list similarity outright.

(1) is why (3) is worth trusting: without the barrier fix the solver never optimises
well enough for the operating point to matter. Numbers in [Results](#results).

> An intermediate version of this file claimed the QUBO beat quota-MMR on NDCG as well.
> That margin was measured with `λ` and `μ` selected on the same dataset they were
> scored on, and it **did not survive** the disjoint-split protocol. What survived is
> (3), which is a narrower claim and a more useful one — it says *when* to reach for a
> QUBO rather than that it is better.

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
| density | 0.0067 | 0.0096 | **0.0442** |
| candidate-set ceiling on recall | 0.49 | 0.28 | — |
| candidates reranked | 200 | 200 | 100 |

Identical budgets, grids and seeds throughout — nothing is tuned per catalogue.

Six benchmarks: five catalogues across two domains, plus one that swaps the retrieval
model. Two are the important ones: **MovieLens groups are
curator-assigned genres** (Drama / Comedy / Romance / other), not popularity tiers, so it
tests the constraint against categories defined independently of the data being scored.

| dataset | items | density | groups from | retrieval |
|---|---|---|---|---|
| Gift Cards | 147 | 0.0442 | popularity tier | ItemKNN |
| Software | 729 | 0.0096 | popularity tier | ItemKNN |
| Luxury Beauty | 1,366 | 0.0067 | popularity tier | ItemKNN |
| Digital Music | 11,268 | 0.0007 | popularity tier | ItemKNN |
| **MovieLens 100K** | 1,349 | 0.0773 | **genre** | ItemKNN |
| **Luxury Beauty (MF)** | 1,366 | 0.0067 | popularity tier | **ALS** |

**Reach** — the tightest fairness budget each method meets *on every seed*. Lower is a
stronger guarantee:

| method | Gift Cards | Luxury Beauty | Digital Music | Software | MovieLens | LB (MF) |
|---|---|---|---|---|---|---|
| quota_mmr | 0.25 | 0.30 | 0.30 | 0.30 | 0.25 | 0.30 |
| **qubo_tabu** | **0.20** | **0.22** | **0.22** | **0.20** | **0.20** | **0.20** |
| **qubo_feasible** | **0.20** | **0.22** | **0.22** | **0.20** | **0.20** | **0.20** |

`greedy_topk` and `mmr` are omitted: neither meets any budget on most benchmarks.

NDCG@10 delivered at that tightest budget:

| method | Gift Cards | Luxury Beauty | Digital Music | Software | MovieLens |
|---|---|---|---|---|---|
| quota_mmr | 0.5540 | 0.9034 | 0.8398 | 0.9033 | **0.9069** |
| **qubo_tabu** | **0.7589** | 0.9044 | 0.8457 | 0.9026 | 0.9009 |
| qubo_feasible | 0.7253 | 0.8475 | 0.7573 | 0.8333 | 0.8111 |

![cross-dataset fairness budget curves](results/datasets_budget.png)

**Holds on 6 of 6.** Gift Cards is the interesting one — it was included precisely
because it was the case most likely to break the result, being small and dense enough
that a greedy heuristic has room to do well. Instead the QUBO wins *both* axes there:
a tighter reach **and** +0.20 NDCG at it. That matches what the config predicted before
the run: small `n` is exactly where exhaustive-ish search should pay off, which makes it
a fair place to look for the method's best case rather than a hostile one.

**Why quota-MMR stalls, mechanically.** Its reach is 0.30 on both 200-candidate
datasets, and the arithmetic floor for k=10 over 4 groups is 0.20. That gap is not about
data: quota-MMR fills group quotas greedily and cannot backtrack, so a slot spent early
on a group that later proves cheap to fill is not recoverable. The QUBO chooses the
whole allocation at once and reaches the floor. Gift Cards' smaller candidate set
loosens the constraint enough for quota-MMR to reach 0.25, which is consistent with the
same explanation.

**Two benchmarks close the two obvious objections.**

*MovieLens closes the group objection.* Every other benchmark groups items by
popularity tier — a partition derived from the very interaction counts being evaluated,
so a sceptic can reasonably say the QUBO only wins because those groups are structurally
easy to balance. MovieLens groups by genre, assigned by the dataset's curators, and the
result is unchanged: reach 0.20 against quota-MMR's 0.25. Note also that quota-MMR is
*slightly more accurate* there (0.9069 vs 0.9009) while still unable to meet the tighter
budget — which is the whole claim in one row.

**This is the strongest claim in the repo**, and it is a claim about *when* to use the
method rather than that the method is better: below a group-exposure requirement of
roughly 0.25, the classical rerankers tested here cannot satisfy the constraint at any
setting of their own hyperparameters, and the QUBO can.

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
python experiments/protocol.py --config configs/amazon_lb.yaml \
  --tau 0.20 0.22 0.25 0.30 0.40 1.00 --repeats 3 --n-users 80

# paired per-user significance tests -- Wilcoxon, Holm-corrected across the family
python experiments/paired.py --config configs/amazon_lb.yaml --lam 0.0 --mu 1.0   --n-users 200 --reference quota_mmr

# does the result survive a change of catalogue? Run the protocol on the others first;
# each config carries its own download command.
python experiments/protocol.py --config configs/amazon_software.yaml  --repeats 3 --n-users 80
python experiments/protocol.py --config configs/amazon_giftcards.yaml --repeats 3 --n-users 80
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
- **Groups:** popularity tiers, not categories — the ratings export carries no metadata.
  Items are rank-ordered by training interaction count and cut into 4 equal-sized tiers
  (tier 0 = short head), the standard short-head/long-tail partition from the
  popularity-bias literature. This makes the fairness term fight the exact bias ItemKNN
  exhibits, rather than a partition chosen to flatter it.
- **Recall ceiling:** candidates come from the same model being reranked, so a held-out
  item is often not in the candidate set at all. `candidate_hit_rate` reports that ceiling
  explicitly (**0.49** here) — recall@10 must be read against it, not against 1.0.

## Layout



```
qubo_rerank/
├── formulations/   objective · cardinality · fairness · builder
├── solvers/        greedy · MMR · quota-MMR · neal SA · tabu · swap · bifurcation · QPU
└── metrics/        NDCG · recall · coverage · Gini · exposure parity · DPFR · kWh
benchmarks/         synthetic generator · Amazon loader (k-core, ItemKNN, LOO split)
experiments/        run_experiment · sweep · protocol · paired · sensitivity ·
                    optimality · exact · ablation · compare_datasets · plots
configs/            YAML experiment configs (synthetic + 3 Amazon categories)
tests/              223 tests · ~75% line coverage
```

A written-up version of the findings, with method and limitations, is in
[`docs/report.md`](docs/report.md).

**If you are reading this to judge the work, three files carry it:**

| file | why |
|---|---|
| [`qubo_rerank/solvers/feasible.py`](qubo_rerank/solvers/feasible.py) | the constraint-preserving annealer, and the response to the penalty-barrier finding |
| [`tests/test_solvers.py`](tests/test_solvers.py) | `TestPenaltyBarrier` — the four tests that hold the headline claim up |
| [`benchmarks/loader.py`](benchmarks/loader.py) | the real-data pipeline; ItemKNN in ~40 lines of scipy rather than 2.5 GB of torch |
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
