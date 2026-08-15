# qubo-rerank

**Fair and energy-aware recommendation list selection via QUBO.**

Conventional recommenders pick the top-k items greedily by score. The resulting lists are
often redundant (near-identical items) and skewed toward popular sellers. `qubo-rerank`
instead selects the *entire list at once* as a Quadratic Unconstrained Binary Optimization
(QUBO) problem, jointly balancing relevance, diversity, and exposure fairness — and reports
the energy cost of doing so.

> Status: **Phase 1 complete** — end-to-end on real data, with a negative result about the
> standard QUBO recipe that turned out to be the most interesting thing here.

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
   here, each ~2× better on the annealer's own objective.
2. **Even fixed, QUBO reranking loses to a classical baseline on real data.** Group-quota
   MMR beats every QUBO variant on NDCG, recall, exposure parity and IBO, in 0.5 seconds
   against 118 and 1/220th the energy.

The second is the bottom line; the first is why the second is worth trusting. Detail
below, numbers in [Results](#results).

### 1. The penalty encoding breaks the sampler

The textbook recipe — encode `exactly k` as a penalty `P(Σx − k)²`, hand the BQM to a
simulated annealer — **does not work at realistic candidate-set sizes**, and it fails
*silently*.

The mechanism is structural, not a matter of tuning. Two feasible k-sets are never
adjacent under single-spin-flip moves: getting from one to another means removing an item
and adding another, and the state in between has `k±1` items and therefore costs the full
penalty `P`. So the sampler faces a dilemma with no good side:

- cold enough to resolve the objective → it cannot cross the barrier, and freezes in
  whichever feasible set it first stumbled into;
- hot enough to cross the barrier → the objective is thermal noise.

On Amazon Luxury Beauty (`n=200`, `k=10`), the cardinality term's largest coefficient is
**737.8** against the objective's **1.0**. Measured consequences:

- `neal` returns lists of exactly the right length **100% of the time** — the constraint
  is satisfied, so nothing looks wrong.
- Its mean QUBO energy is **worse than a greedy MMR baseline** evaluated on the same BQM.
  A hand-written heuristic beats simulated annealing at minimising the annealer's own
  objective.
- Raising `num_sweeps` 10× makes it **slightly worse, not better** — the giveaway that
  more search cannot help when the search is looking at the wrong scale.
- Lowering the penalty strength does not help either: the barrier height *is* the
  strength, and a strength low enough to cross is too low to enforce the constraint.

Two fixes, both in the repo, both beating `neal` by ~2× on its own objective:

| solver | idea | keeps generic `dimod` sampler? |
|---|---|---|
| `qubo_tabu` | same BQM, same single-flip move set, but with **search memory** | yes |
| `qubo_feasible` | anneal over **k-subsets** via swap moves; the barrier never exists | no |

That `qubo_tabu` also works is what rules out the *formulation* as the culprit: a generic
single-flip sampler can solve this BQM given a strategy that escapes basins deliberately
rather than thermally. The problem is the interaction between penalty encoding and
thermal search, which is why D-Wave ships constrained (CQM) solvers rather than expecting
users to tune penalty weights.

**Why this matters for the quantum framing.** A physical annealer receives the same
penalty-encoded BQM and the same barrier. This result is a concrete, measured argument for
constraint-aware solvers over generic QUBO sampling in recommendation reranking — and a
caution against reporting "QUBO reranking achieves good diversity" without checking
whether the sampler optimised anything at all. A near-random feasible list scores *well*
on diversity and coverage. Reporting only downstream metrics would have hidden this
completely.

## Results

### Synthetic benchmark (n=60, k=10, 25 users)

| method | NDCG@10 | cat. coverage | exposure parity ↓ | intra-list sim ↓ | Gini ↓ | secs | kWh |
|---|---|---|---|---|---|---|---|
| greedy_topk | **1.0000** | 0.5333 | 1.0773 | 0.3336 | 0.6472 | **0.003** | 2.1e-07 |
| mmr | 0.9502 | **1.0000** | 0.5333 | 0.1952 | 0.3516 | 0.026 | 2.3e-09 |
| quota_mmr | 0.9072 | **1.0000** | **0.2667** | 0.1524 | 0.2872 | 0.028 | 2.1e-09 |
| qubo_sa | 0.7448 | 0.9933 | 0.2960 | 0.1508 | 0.2887 | 8.31 | 2.8e-05 |
| qubo_tabu | 0.8510 | **1.0000** | **0.2667** | **0.1312** | 0.2984 | 5.61 | 1.8e-05 |
| qubo_feasible | 0.8585 | **1.0000** | **0.2667** | 0.1331 | **0.2696** | 4.23 | 1.4e-05 |

`qubo_feasible` trades 0.049 NDCG against `quota_mmr` for a better Gini (0.2696 vs 0.2872)
and lower intra-list similarity (0.1331 vs 0.1524) — a real Pareto trade, but one costing
**150× the wall-clock**. It does at least dominate `qubo_sa` outright: +0.11 NDCG, better
diversity, better Gini, and roughly half the time.

Two things the λ/μ sweep shows (`results/synthetic_sweep.csv`):

**λ behaves as it should**, and the solver is verifiably optimising. At λ=0 the objective
is pure relevance, so the optimum is exactly greedy top-k — and `qubo_feasible` returns
NDCG **1.0000**, recovering the known answer outright. As λ rises 0 → 16, NDCG falls
1.000 → 0.657 and intra-list similarity falls 0.334 → 0.120.

**μ works, but λ makes it redundant here.** The entire 4×3 grid produces only *two*
distinct `exposure_parity` values: 1.0773 at (λ=0, μ=0), and 0.2667 everywhere else —
which is its arithmetic floor for 6 groups and k=10 (four groups get 2 slots, two get 1).
So μ genuinely fixes parity when nothing else is, but any λ≥1 already drives it to the
floor on its own. The synthetic generator makes items within a category mutually similar,
so diversity and fairness are the same axis by construction and there is no trade-off
curve to trace. This is exactly why the real dataset matters: there, groups are popularity
tiers, which are *not* aligned with the similarity structure, and μ traces a real curve.

### Amazon Luxury Beauty (n=200, k=10, 200 users, λ=4, μ=0)

Recall's ceiling is **0.49** — the fraction of users whose held-out item was in the
top-200 candidate set at all. Read `recall@10` against that, not against 1.0.

| method | NDCG@10 | recall@10 | cat. cov. | parity ↓ | ILS ↓ | cat. coverage | Gini ↓ | AI-F ↓ | IBO | secs | kWh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy_topk | **1.0000** | **0.250** | 0.610 | 0.9983 | 0.0524 | 0.326 | 0.8776 | 2.4e-04 | 0.366 | **0.03** | **1.9e-07** |
| mmr | 0.9159 | 0.245 | 0.733 | 0.8225 | 0.0189 | 0.441 | 0.7924 | 1.3e-04 | **0.380** | 0.71 | 2.3e-06 |
| quota_mmr | 0.8369 | 0.225 | **1.000** | **0.2607** | 0.0095 | 0.510 | 0.7342 | 9.9e-05 | 0.324 | 0.53 | 1.7e-06 |
| qubo_sa | 0.4370 | 0.170 | 0.910 | 0.5205 | **0.0020** | **0.667** | **0.5824** | **3.7e-05** | 0.197 | 377.3 | 1.2e-03 |
| qubo_tabu | 0.6646 | 0.200 | 0.883 | 0.5648 | 0.0014 | 0.539 | 0.7166 | 6.3e-05 | 0.268 | 56.9 | 1.8e-04 |
| qubo_feasible | 0.6593 | 0.205 | 0.888 | 0.5283 | 0.0014 | 0.560 | 0.6975 | 5.6e-05 | 0.282 | 117.8 | 3.8e-04 |

**The honest bottom line: on real data the classical baseline wins.** `quota_mmr` beats
every QUBO variant on NDCG (0.837 vs 0.659), recall (0.225 vs 0.205), group exposure
parity (0.261 vs 0.528) and IBO (0.324 vs 0.282) — in **0.53 seconds against 118**, using
**1/220th the energy**. The fixed solvers rescue the QUBO from `qubo_sa`'s 0.437 NDCG, but
they rescue it into second place.

Where the QUBO does win — Gini, catalogue coverage, intra-list similarity, AI-F — deserves
scepticism rather than celebration. Every one of those is improved by *spreading selections
around*, which is also what a bad optimiser does by accident. `qubo_sa` "wins" Gini
(0.5824), catalogue coverage (0.667) and AI-F outright, and it is the method demonstrably
closest to picking at random. That is precisely why the energy column and the
solver-energy diagnostics are in this repo: without them, the worst optimiser here looks
like the fairest method.

II-F is not reported per-method above because it barely discriminates: every method lands
between 0.002083 and 0.002138, a 2.6% spread across solvers whose NDCG differs by a
factor of two. Under leave-one-out there is one relevant item per user against a
1,365-item catalogue, so the measure is dominated by the shared sea of zeros — the same
degeneracy flagged in `metrics/dpfr.py`. It is in the CSV; it should not carry an
argument.

**What would change this verdict.** The QUBO is being asked to beat a strong greedy
heuristic on a 200-item candidate set, where exhaustive-ish search has little room to pay
off. The case for it has to come from Phase 3 — larger candidate sets where greedy's
myopia actually costs something, and constraints (per-seller contracts, hard slot quotas)
that MMR cannot express but a QUBO can. On this benchmark, at this size, it does not.

### What the λ/μ sweep shows on real data

Grid: λ ∈ {0, 4, 16} × μ ∈ {0, 1, 4, 16}, 40 users, `qubo_feasible`
(`results/amazon_lb_sweep.csv`).

**The solver is verifiably optimising.** At λ=0, μ=0 the objective is pure relevance, so
the optimum is exactly greedy top-k — and it returns NDCG **1.0000** on a 200-variable
instance. That is the one problem size here whose answer is known independently, and it
gets it exactly right.

**μ does real work here, unlike on the synthetic benchmark.** Turning μ from 0 to 1 moves
exposure parity from 0.929 to **0.199** — its arithmetic floor for 4 groups and k=10 — and
category coverage from 0.635 to **1.000**, at any λ. Popularity tiers are not aligned with
the similarity structure, so the diversity term cannot reach this on its own and the
fairness term has something to do.

**NDCG substantially overstates what fairness costs.** Going from (λ=0, μ=0) to
(λ=4, μ=1):

| | NDCG@10 | recall@10 | parity | cat. coverage |
|---|---|---|---|---|
| λ=0, μ=0 | 1.0000 | 0.3250 | 0.9292 | 0.6354 |
| λ=4, μ=1 | 0.6429 | 0.2750 | **0.1992** | **1.0000** |
| change | −36% | **−15%** | −79% | +57% |

NDCG here is computed against the retrieval model's *own* scores, so it is maximised by
definition when you simply agree with the model — reranking away from it always looks
expensive. Recall is measured against a genuinely held-out purchase. By that measure,
driving exposure parity down by 79% and taking category coverage to 100% costs 15% of
predictive accuracy, not 36%. **Papers that report only NDCG overstate the price of
fairness by roughly a factor of two on this data.** This is the most transferable result
here and it does not depend on the QUBO at all.

**Diversity and individual-item fairness pull against each other.** IBO falls from 0.667
to 0.444 as λ rises 0 → 4. Spreading a list across dissimilar items is not the same as
giving individual items their due exposure, and optimising the first can damage the
second.

### Trade-off curves

![synthetic Pareto](results/synthetic_sweep_pareto.png)

![amazon Pareto](results/amazon_lb_sweep_pareto.png)

Each panel plots NDCG against one cost axis; the classical baselines are fixed reference
points. The question is not whether the QUBO curve *moves* — any reranker moves — but
whether any point on it sits above and to the left of `quota_mmr` (green star). On
intra-list similarity and Gini it does. On group exposure parity, the axis the fairness
term optimises directly, it does not.

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

# picks up those matched baselines automatically -- do not pass results/amazon_lb.csv
# instead; Gini and catalogue coverage are catalogue-level aggregates and do not
# transfer between user-sample sizes
python experiments/plot_pareto.py --sweep results/amazon_lb_sweep.csv
```

Runs must be sequential. `seconds` and `kWh` are wall-clock measurements, so a second
job on the same machine is reported as the first job's energy cost.

## Data and protocol

- **Dataset:** Amazon Luxury Beauty (McAuley Lab, ratings-only export). 574,628 raw
  interactions → **32,732 interactions / 3,589 users / 1,365 items** after 5-core
  filtering.
- **Model:** ItemKNN — shrunk cosine similarity over the interaction matrix, top-100
  neighbourhood for scoring. Supplies both the relevance scores `r_i` and the item-item
  similarity `s_ij` from a single fit.
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

### Evaluation choices that change the numbers

Two decisions here are load-bearing, and both were found by looking rather than assumed.

**Lists are presented in descending relevance order.** A QUBO selects a *set*; it says
nothing about display order. The solvers return their sets index-sorted, and NDCG is
position-discounted, so scoring the raw output charges the QUBO methods for an ordering
they never claimed to produce. On the synthetic benchmark that is worth up to **0.09
NDCG** — larger than several of the differences being reported. Every solver's output is
therefore sorted by descending relevance before scoring, baselines included (worth a
further ~0.03 to MMR). The comparison is then strictly about *which items were chosen*.
On the Amazon benchmark this is a no-op, because the loader already returns candidates in
descending relevance order.

**Fairness is reported on axes the QUBO does not optimise.** `exposure_parity` is exactly
what the `μ` penalty targets, so a good score there is close to tautological. The table
therefore also carries Gini, catalogue coverage, and the individual-item measures II-F,
AI-F and IBO from Rampisela et al. (`qubo_rerank/metrics/dpfr.py`, ported from their
MIT-licensed reference implementation and checked against it to 1e-12 in
`tests/test_dpfr.py`). Those are group-agnostic and independent of the penalty, which
makes them the ones worth believing.

**The energy column measures solving only, and is still shaky.** Three things had to be
got right before the numbers meant anything:

- The timer used to start *before* `EmissionsTracker.start()`, which probes the hardware
  and takes seconds — so every solver was charged a constant ~5 s. `greedy_topk` read
  5.4 s for 0.008 s of work.
- Scoring used to happen inside the measured window. Intra-list similarity alone is
  O(k²) in Python per user, which is noise for the QUBO solvers and most of the measured
  time for the baselines. Solving and scoring are now separate passes.
- Runs must not be executed concurrently; CPU contention contaminates both timing and
  energy.

Even then: codecarbon estimates from CPU TDP and utilisation, it is not a power meter,
and below ~0.1 s of work its readings are dominated by noise — the sub-second baselines
here disagree with each other by 100× on kWh while differing by only 10× in wall-clock.
Treat these as a relative comparison between the *slow* solvers on identical hardware,
and not as absolute claims. Wegmeth et al. (below) used a physical meter and are the
right citation for rigorous figures.

### Why not RecBole

The plan called for RecBole's `ItemKNN` to supply `r_i` and `s_ij`. Reading
`recbole/model/general_recommender/itemknn.py` shows the model is a shrunk cosine
similarity plus a top-k truncation — no gradients, no GPU, nothing `torch` actually does.
Pulling in ~2.5 GB of dependency to reach forty lines of sparse linear algebra buys a
dependency, not a capability, so the same formulation is implemented directly against
`scipy.sparse` in `benchmarks/loader.py`. The on-disk format stays RecBole-compatible, so
swapping the real thing back in is a loader change and nothing else.

## Layout

```
qubo_rerank/
├── formulations/   objective · cardinality · fairness · builder
├── decomposition/  (Phase 3) sparsification · clustering · hierarchical selection
├── solvers/        greedy · MMR · quota-MMR · neal SA · tabu · feasible-set annealing
└── metrics/        NDCG · recall · coverage · Gini · exposure parity · kWh · CO2e
benchmarks/         synthetic generator · Amazon loader (k-core, ItemKNN, LOO split)
experiments/        run_experiment · sweep · plot_pareto
configs/            YAML experiment configs
```

## Roadmap

| Phase | Content | Status |
|---|---|---|
| **1** | CF baseline, core QUBO, fairness term, solvers, one real dataset | done |
| **2** | 3 datasets, repeated seeds, confidence intervals, full Pareto analysis | next |
| **3** | Decomposition for large catalogues; Simulated Bifurcation; D-Wave Leap QPU | |
| **4** | Packaging, docs, PyPI release | |
| **5** | Technical report / preprint | |

Phase 2's first job is confidence intervals. Every number above is a single seed, and the
gaps between the QUBO solvers are small enough that some of them will not survive repeated
runs. The `neal` result will; a ~2× energy gap is not noise.

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
