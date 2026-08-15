# When is a QUBO worth it for fair recommendation reranking?

**Technical report — `qubo-rerank`, Phase 3**
Huzaifa Abdul Rehman

---

## Abstract

We study whether formulating recommendation-list selection as a Quadratic Unconstrained
Binary Optimization (QUBO) problem — jointly optimising relevance, intra-list diversity
and group exposure fairness — offers anything over classical greedy rerankers on real
e-commerce data.

We report three results. First, the textbook recipe fails silently: encoding the
"exactly k" constraint as a quadratic penalty and handing the problem to a simulated
annealer produces lists that are always the right length and close to arbitrary among
lists of that length, losing to a hand-written greedy heuristic on the annealer's own
objective. Second, this is a property of the **encoding**, not of thermal search: a
continuous-dynamics solver (Simulated Bifurcation) fails on the same problem by a
different mechanism, while tabu search and a constraint-preserving swap annealer both
succeed. Third, under a protocol that tunes every method — baselines included — on a
disjoint half of the users, the QUBO's advantage is not accuracy but **feasibility**:
below a group-exposure requirement of τ ≈ 0.25, no classical reranker tested can satisfy
the constraint at any setting of its own hyperparameters, and the QUBO can. This holds on
three Amazon categories differing by an order of magnitude in catalogue size and 6.6× in
density.

We also report what the method does *not* buy: at looser fairness requirements it ties
the strongest baseline on accuracy (a paired test over 200 users puts the difference at
a certain but negligible 0.0012 NDCG), it loses on intra-list diversity, and it costs
roughly 100× the wall-clock.

---

## 1. Problem

Given `n` candidate items with relevance scores `r_i` and pairwise similarity `s_ij`,
select exactly `k` items minimising

```
H  =  −Σ r_i·x_i                        relevance
      + λ Σ s_ij·x_i·x_j                diversity
      + P (Σ x_i − k)²                  cardinality
      + μ Σ_c (Σ_{i∈c} x_i − k/|C|)²    group exposure fairness
```

over `x ∈ {0,1}ⁿ`. The first two terms are the standard relevance/diversity trade-off;
the fourth is the contribution under study. The third is the conventional way to express
a hard cardinality constraint in an *unconstrained* binary model, and is where the
trouble starts.

**Setting.** Two-stage retrieval → rerank, as deployed systems work. ItemKNN (shrunk
cosine similarity, top-100 neighbourhood) retrieves the top-n candidates and supplies both
`r_i` and `s_ij` from one fit; the QUBO selects k=10 from n=200. Leave-one-out evaluation
on each user's most recent interaction, 5-core filtering.

**Groups.** The ratings export carries no product metadata, so groups are *popularity
tiers*: items rank-ordered by training interaction count and cut into four equal-sized
tiers, the standard short-head/long-tail partition. This makes the fairness term fight
the exact bias the retrieval model exhibits, rather than a partition chosen to flatter
it.

---

## 2. The penalty encoding defeats unaided search

### 2.1 Discrete search: the barrier

Two valid k-item lists are never adjacent under single-bit flips. Moving between them
means removing one item and adding another, and the intermediate state has k±1 items and
pays the full penalty `P`. On Amazon Luxury Beauty at n=200, k=10, the cardinality term's
largest coefficient is **737.8** against an objective of **1.0**.

The sampler therefore faces a dilemma with no good side: cold enough to resolve the
objective and it cannot cross the barrier; hot enough to cross and the objective is
thermal noise.

Four independent observations confirm this is the mechanism rather than a tuning failure:

1. `neal` returns lists of exactly the right length **100% of the time** — the constraint
   holds, so nothing looks wrong.
2. Its mean QUBO energy is **worse than greedy MMR's** on the same model. A hand-written
   heuristic beats simulated annealing at minimising the annealer's own objective.
3. Raising the sweep count tenfold makes it slightly **worse**, not better.
4. Lowering the penalty does not help: the barrier height *is* the penalty, and a
   strength low enough to cross is too low to enforce the constraint.

Across 5 seeds, `qubo_sa` scores 0.475 ± 0.030 NDCG against 0.692 ± 0.038 for the
corrected solvers — a gap exceeding five standard deviations.

### 2.2 Continuous dynamics: the common-mode field

The barrier argument concerns move sets, and so says nothing about a method that has
none. Simulated Bifurcation (Goto et al., *Science Advances* 2019) integrates a
Hamiltonian system through the interior of the hypercube and recovers spins only at the
end. If the penalty encoding were merely hostile to discrete local search, SB should be
unaffected.

**It performs worst of all: on the barrier instance it returns the empty set**, at 600
and 2000 steps, in both ballistic and discrete variants.

The reason is structural and distinct from the barrier. The cardinality penalty couples
every pair of items positively, so the Ising form carries a large *uniform* field — mean
4.7413, spread 0.0024. The signal distinguishing items is 5×10⁻⁴ of the common mode;
every variable feels the same drive and they bifurcate together. Sweeping the penalty:

| penalty strength | field spread ÷ mean | items selected (target 10) |
|---|---|---|
| 277.8 (enforces the constraint) | 5.0×10⁻⁴ | 0 |
| 27.8 | 5.0×10⁻³ | 0 |
| 8.3 | 1.6×10⁻² | 0 |
| 2.8 | 4.5×10⁻² | 9 |
| 0.8 | 1.2×10⁻¹ | 2 |

SB resolves the objective only once the penalty falls ~100× below the level that enforces
the constraint — `neal`'s dilemma in continuous form.

### 2.3 What does work

| solver | paradigm | outcome |
|---|---|---|
| `qubo_sa` | discrete, thermal | fails |
| `qubo_sb` | continuous dynamics | fails worse |
| `qubo_tabu` | discrete + explicit memory | works |
| `qubo_feasible` | constraint-preserving swap moves | works |

The recovery comes from **memory** or from **never leaving the feasible set** — not from
better search. Two solvers from unrelated paradigms both fail when handed the penalty
encoding unaided, which argues for constraint-aware solvers (D-Wave's CQM, or move sets
restricted to the feasible region) more strongly than either failure alone.

**Implication for the quantum framing.** A physical annealer receives the same
penalty-encoded model and the same barrier. A near-random feasible list still scores
*well* on diversity and coverage, so a study reporting only downstream metrics would show
a clean result and never detect that the sampler optimised nothing.

---

## 3. Evaluation protocol

Two methodological errors were made and corrected during this work; both are worth
stating because both are easy to make.

**The operating point decides the comparison.** At λ=4, μ=0 every QUBO variant loses to
group-quota MMR on essentially everything, and an earlier draft concluded exactly that.
But μ=0 switches the fairness term *off*, so that comparison pits a QUBO with no reason
to optimise exposure against a baseline with group quotas hard-coded — and then scores
both on exposure. The configuration mattered more than the method did.

**Selection and evaluation must not share data.** Choosing λ and μ by inspecting a sweep
and then reporting the best cell measures, in part, how many cells were tried. The
protocol used here instead:

1. Split users into **disjoint** tuning and evaluation halves.
2. Declare a fairness budget τ — an upper bound on mean exposure-parity deviation, as a
   deployer would state the requirement.
3. Tune **every** method, baselines included, by maximising NDCG@10 subject to
   `parity ≤ τ` on the tuning half.
4. Evaluate the chosen configuration once, on the other half.
5. Repeat over seeds; sweep τ.

Tuning the baselines matters: MMR and quota-MMR have a trade-off parameter of their own,
and searching the QUBO's two weights while leaving the baseline's at its default is the
same asymmetry that produced the wrong answer the first time, pointed the other way.
Methods that cannot meet a budget are recorded as *infeasible* rather than quietly
reported at their least-bad setting alongside methods that met it.

---

## 4. Results

### 4.1 Feasibility, not accuracy

**Reach** — the tightest budget a method meets on *every* seed:

| method | Gift Cards | Luxury Beauty | Software |
|---|---|---|---|
| greedy top-k | — | — | 1.00 |
| MMR | 1.00 | 1.00 | 1.00 |
| quota-MMR | 0.25 | 0.30 | 0.30 |
| **QUBO + tabu** | **0.20** | **0.22** | **0.20** |
| **QUBO + swap annealing** | **0.20** | **0.22** | **0.20** |

NDCG@10 at that reach:

| method | Gift Cards | Luxury Beauty | Software |
|---|---|---|---|
| quota-MMR | 0.5540 | 0.9034 | 0.9033 |
| QUBO + tabu | **0.7589** | 0.9044 | 0.9026 |

The result holds on all three catalogues, which span 147–1,366 items and 0.0067–0.0442
density. It is strongest on the smallest and densest (Gift Cards), where the QUBO takes
both a tighter budget and +0.20 NDCG — consistent with small `n` being where
exhaustive-ish search should pay off.

**Why quota-MMR stalls.** Its reach is 0.30 on both 200-candidate datasets while the
arithmetic floor for k=10 over 4 groups is 0.20. Quota-MMR fills group quotas greedily
and cannot backtrack, so a slot spent early is not recoverable; the QUBO chooses the whole
allocation at once. Gift Cards' smaller candidate set loosens the constraint enough for
quota-MMR to reach 0.25, consistent with the same explanation.

### 4.2 What it does not buy

Paired Wilcoxon signed-rank over 200 users, Holm-corrected, at λ=0/μ=1, against
quota-MMR:

| metric | median diff | 95% CI | better/worse/tied | p (Holm) |
|---|---|---|---|---|
| exposure parity ↓ | **−0.1000** | [−0.1000, −0.0500] | 114 / **0** / 86 | 2.8×10⁻²⁵ |
| NDCG@10 ↑ | +0.0012 | [+0.0012, +0.0043] | 126 / 64 / 10 | 3.3×10⁻⁷ |
| recall@10 ↑ | 0.0000 | [0.0000, 0.0000] | 5 / 1 / 194 | 0.57 |
| intra-list sim ↓ | +0.0022 | [+0.0022, +0.0036] | 26 / 165 / 9 | 2.3×10⁻²⁷ |

The NDCG row is the useful one. A seed-level comparison called this a tie; the paired
test finds the difference is *certain* and that its median size is **0.0012**. Both
readings are correct and either alone misleads — a table of p-values would call it a win,
a table of means called it a tie. **Significant and negligible are not opposites.**

The parity row has the opposite shape: a median improvement of 0.10, better on 115 users
and worse on **none**.

Recall does not move for any method (194 of 200 users tie): under leave-one-out there is
at most one relevant item per user against a candidate-set ceiling of 0.49, so most users
score identically under every reranker. A recall difference on this benchmark would be
noise.

Cost is unchanged throughout: **16.1 s against 0.16 s**, a factor of ~100.

---

## 5. Measurement

Energy and timing are reported, and getting them to mean anything required four
corrections. Three were bugs; the fourth was the environment.

- The timer started *before* `EmissionsTracker.start()`, whose hardware probe takes
  seconds — charging every solver a constant ~5 s. `greedy_topk` read 5.4 s for 0.008 s
  of work.
- Scoring happened inside the measured window. Intra-list similarity is O(k²) per user in
  Python: noise for the QUBO solvers, most of the measured time for the baselines.
- Runs must be sequential; CPU contention contaminates both.
- **The machine must stay on mains power.** Mid-session the laptop was unplugged; on
  battery the CPU dropped from ~3.6 GHz turbo to a pinned 1.297 GHz and every timing rose
  ~2.8×, reproducibly. **Every quality metric was byte-identical across the two regimes.**
  In a project that reports energy, an energy column that moves by a factor of three
  because a cable came out — with nothing else moving — is worth stating plainly.

**One solver's `seconds` is not a measurement.** `dwave-samplers`' `TabuSampler` defaults
to `timeout=20` ms — a wall-clock stopping rule. `qubo_tabu` therefore runs for a
configured budget, so its wall-clock is roughly constant regardless of machine speed
(1.09× across the power-state change, against 2.8× for everything else) and its **quality
is hardware-dependent**. The unplugged laptop turned that inference into a measurement:
on identical seeds, `quota_mmr` and `qubo_feasible` were identical to four decimals while
`qubo_tabu` moved 0.8801 → 0.8873. Since the headline result is this solver's, it
qualifies it directly.

`codecarbon` estimates from CPU TDP and utilisation and is not a power meter; below ~0.1 s
its readings are noise-dominated. Treat the figures as a relative comparison between the
slow solvers on identical hardware.

---

## 6. Limitations

1. **Groups are popularity tiers, not categories or sellers.** The ratings export carries
   no metadata. The partition is defensible and standard, but it is not seller fairness.
2. **One model family.** ItemKNN supplies both relevance and similarity from one fit,
   which is convenient and means every result is conditioned on a single retrieval model.
3. **`qubo_tabu`'s results are hardware-dependent** (§5). A work-based stopping criterion
   is needed before cross-machine claims.
4. **Candidate sets are small** (n=100–200). The scaling question — whether any of this
   survives a catalogue an order of magnitude larger — is now *tractable* thanks to the
   sparse similarity path, but has not been run.
5. **Three catalogues, all Amazon.** Cross-domain generalisation is untested.
6. **No physical quantum hardware.** Everything here is classical or quantum-*inspired*.

---

## 7. Reproducing

```bash
pip install -r requirements.txt
pytest tests/                                    # 223 tests

curl --create-dirs -o data/amazon_lb/Luxury_Beauty.csv \
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Luxury_Beauty.csv

python experiments/protocol.py --config configs/amazon_lb.yaml \
  --tau 0.20 0.22 0.25 0.30 0.40 1.00 --repeats 3 --n-users 80
python experiments/compare_datasets.py
python experiments/paired.py --config configs/amazon_lb.yaml --lam 0.0 --mu 1.0 \
  --n-users 200 --reference quota_mmr
```

Runs that report `seconds` or `kWh` must be sequential and on mains power.

---

## References

- Goto, Tatsumura, Dixon. *Combinatorial optimization by simulating adiabatic
  bifurcations in nonlinear Hamiltonian systems.* Science Advances, 2019.
- Ferrari Dacrema, Felicioni, Cremonesi. *Optimizing the Selection of Recommendation
  Carousels with Quantum Computing.* RecSys 2021.
- Ferrari Dacrema et al. *Are We Really Making Much Progress?* RecSys 2019.
- Nembrini, Ferrari Dacrema, Cremonesi. *Feature Selection for Recommender Systems with
  Quantum Computing.* arXiv:2110.05089.
- Rampisela et al. *Item Fairness of Recommender Systems.* WWW 2025. (II-F, AI-F, IBO;
  ported from the authors' MIT-licensed reference and verified to 1e-12.)
- Abdollahpouri, Burke, Mobasher. *Managing Popularity Bias in Recommender Systems with
  Personalized Re-ranking.* FLAIRS 2019.
- Wegmeth, Vente, Said, Beel. *Green Recommender Systems.* ACM TORS 2025.
