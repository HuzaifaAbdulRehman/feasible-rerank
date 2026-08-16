# When is a QUBO worth it for fair recommendation reranking?

**Technical report — `feasible-rerank`**
Huzaifa Abdul Rehman

---

## Abstract

We study whether formulating recommendation-list selection as a Quadratic Unconstrained
Binary Optimization (QUBO) problem — jointly optimising relevance, intra-list diversity
and group exposure fairness — offers anything over classical greedy rerankers on real
e-commerce data.

We report five results, the third of which corrects a claim made by an earlier version of
this report. First, the textbook recipe fails silently: encoding the
"exactly k" constraint as a quadratic penalty and handing the problem to a simulated
annealer produces lists that are always the right length and close to arbitrary among
lists of that length, losing to a hand-written greedy heuristic on the annealer's own
objective. Second, this is a property of the **encoding**, not of thermal search: a
continuous-dynamics solver (Simulated Bifurcation) fails on the same problem by a
different mechanism, while tabu search and a constraint-preserving swap annealer both
succeed. Third, under a protocol that tunes every method — baselines included — on a
disjoint half of the users, the QUBO's advantage is not accuracy but **feasibility**:
below a group-exposure requirement of τ ≈ 0.25, no classical reranker tested can satisfy
the constraint at any setting of its own hyperparameters — **and that claim did not
survive**. It measured a missing remainder rule in the one classical fairness baseline
implemented at the time. Largest-remainder apportionment (`BalancedQuota`) attains the
same budget on all **eight benchmarks**, deterministically and at a small fraction of
the compute (no multiplier is quoted; see section 6): four Amazon
categories spanning 77x in size and 53x in density, MovieLens 100K (curator-assigned
genres), one swapping ItemKNN for matrix factorisation, and two partitioning the Software
catalogue by **real seller** and **real product category** from Amazon's metadata export.
Reach ties 8 of 8.

Fourth, what survives is narrower and concerns accuracy rather than feasibility. At that
shared tightest budget the QUBO returns a more accurate list than apportionment on **8 of
8** benchmarks (mean +0.0086 NDCG, sign test and Wilcoxon both p = 0.008), with the
largest margin (+0.0176) where ALS makes the similarity matrix densest. The mechanism is
non-separability: apportionment fills each group greedily, which is optimal for a
separable objective and not for one carrying `lam * sum s_ij x_i x_j`. On the objective
itself the QUBO's margin over the best classical method grows with `lam` — tied at 0,
+5.5% at 4, 2x at `lam=4, mu=0`.

Fifth, the naive quota heuristic collapses entirely where exposure targets are
proportional rather than equal — which is what real, unequally sized product categories
call for — meeting no budget below 1.00 while both apportionment and the QUBO reach 0.20.
A matched ablation changing only the target vector isolates the cause.

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

Because every result is downstream of this model, it is validated against RecBole's
reference `ComputeSimilarity`, transcribed into the test suite as an oracle. Similarity
values agree to 6e-08 (float32 vs float64); truncated neighbour sets agree for 626 of 727
items, and all 101 disagreements sit at exactly-tied similarity values — the two
implementations choose different members of a tied group and are otherwise identical.

**Groups.** The *ratings* export carries no product metadata, so the default partition is
*popularity tiers*: items rank-ordered by training interaction count and cut into four
equal-sized tiers, the standard short-head/long-tail partition. This makes the fairness
term fight the exact bias the retrieval model exhibits, rather than a partition chosen to
flatter it.

Amazon's *separate* metadata export does carry sellers and categories where it is
populated, and two of the eight benchmarks use them: 99.3% of the filtered Software
catalogue carries a brand and 93.5% a category path, against 0.1% for Luxury Beauty. Both
partitions are near-independent of the popularity tiers they replace (NMI 0.012 and
0.016), so they pose a different fairness question rather than restating this one. The
loader refuses a metadata grouping below a coverage threshold rather than silently
collapsing to a single group. See `benchmarks/metadata.py`.

---

## 2. The penalty encoding defeats unaided search

### 2.1 Discrete search: the barrier

Two valid k-item lists are never adjacent under single-bit flips. Moving between them
means removing one item and adding another, and the intermediate state has k±1 items and
pays the full penalty `P`. The penalty is scaled per instance from the objective it has to
dominate, so its size varies: across ten Amazon Luxury Beauty instances at n=200, k=10 the
cardinality term's largest coefficient ranges from **505 to 4,806** (mean ~2,000) against a
normalised objective of **1.0**. The ratio, not any single value, is the point -- the
constraint outweighs the objective by three orders of magnitude on a typical instance.

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

### 2.1a Measured against the true optimum

At small n the optimum is enumerable -- C(26,5) = 65,780 subsets -- so solvers can be
scored as a fraction of the available improvement recovered, anchored between a random
feasible set (0%) and the exact optimum (100%). At n=22, k=5 over 12 instances:

  qubo_tabu       100.0%   exactly optimal on 100% of instances
  qubo_feasible   100.0%   exactly optimal on 100% of instances
  qubo_sa          80.1%
  mmr              75.3%
  quota_mmr        67.1%
  qubo_sb          18.9%   averaged over the 6 of 12 runs that were feasible
  greedy_topk      14.4%

Both constraint-preserving solvers are *exactly* optimal, which is what licenses trusting
them at n=200 where enumeration is impossible.

The barrier also worsens with n, broadly as the mechanism predicts -- the penalty scales
with the number of variables while the objective does not. 12 instances per size,
regenerated by the command in section 7:

  n                  14      18      22      26      30     200
  qubo_sa         87.7%   82.0%   80.1%   74.7%   75.2%   worse than no search
  qubo_tabu       100.0%  100.0%  100.0%  100.0%  100.0%   --
  qubo_feasible   100.0%  100.0%  100.0%  100.0%  100.0%   --

**The trend is downward but not monotone**, and this table is weaker than the version it
replaces. An earlier edition printed 87.6/81.1/77.7/76.9/73.4 -- a clean monotone decline
-- from runs that were never saved and could not be regenerated: `--n-items` took a single
integer, the CSV had no `n` column, and no driver script existed. Rebuilt honestly, the
decline over the range is real (87.7% to ~75%) but n=26 and n=30 sit within noise of each
other at 12 instances. The n=200 endpoint, where `qubo_sa` is beaten by a method doing no
search at all, is what actually carries the claim; the ladder is supporting evidence and
should not be read as a clean law.

This explains why the failure is rarely reported: at textbook scale the penalty encoding
works well enough to look fine. It breaks at the sizes a real reranker faces.

### 2.1b Is it a budget problem? No.

Each solver was run across three orders of magnitude of compute and scored on QUBO
energy. `neal`'s best result at any budget -- 1,000 reads x 10,000 sweeps, 184 s -- is
**+0.00152**, still worse than MMR's **+0.00065**, a single pass performing no search.
`qubo_feasible` reaches -0.00037 in **0.2 s**, a ~900x smaller budget, and neither fixed
solver ever fails to cross zero.

More compute is sometimes actively harmful: on one seed, 100 reads x 10,000 sweeps gives
+0.00127 in 19 s while the same reads at 100,000 sweeps gives +0.00260 in 183 s. A longer
anneal is a colder one, so it freezes harder into whichever feasible set it first found.

`qubo_tabu` is flat across the ladder (-0.00062 to -0.00070) and `qubo_feasible` improves
monotonically -- the latter being the only curve here that behaves as a search algorithm
should.

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

| method | Gift Cards | Luxury Beauty | Digital Music | Software | MovieLens |
|---|---|---|---|---|---|
| greedy top-k | — | — | — | 1.00 | — |
| MMR | 1.00 | 1.00 | — | 1.00 | 1.00 |
| quota-MMR | 0.25 | 0.30 | 0.30 | 0.30 | 0.25 |
| **QUBO + tabu** | **0.20** | **0.22** | **0.22** | **0.20** | **0.20** |
| **QUBO + swap annealing** | **0.20** | **0.22** | **0.22** | **0.20** | **0.20** |

NDCG@10 at that reach:

| method | Gift Cards | Luxury Beauty | Digital Music | Software | MovieLens |
|---|---|---|---|---|---|
| quota-MMR | 0.5540 | 0.9034 | 0.8398 | 0.9033 | **0.9069** |
| QUBO + tabu | **0.7589** | 0.9044 | 0.8457 | 0.9026 | 0.9009 |

The result holds on all five catalogues, spanning 147–11,268 items and 0.0007–0.0773
density, across two domains.

**MovieLens closes the obvious objection.** The four Amazon benchmarks group items by
popularity tier -- a partition derived from the same interaction counts being evaluated,
so a sceptic can fairly argue the QUBO wins only because those groups are structurally
easy to balance. MovieLens groups by curator-assigned genre, and the result is unchanged.
It also shows the claim in its cleanest form: quota-MMR is *more accurate* there
(0.9069 vs 0.9009) and still cannot meet the tighter budget. It is strongest on the smallest and densest (Gift Cards), where the QUBO takes
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

## 5b. Relation to published QUBO recommender work

Ferrari Dacrema, Nembrini, Felicioni and Cremonesi have a line of papers applying quantum
annealing to recommender systems -- carousel selection (RecSys 2021) and feature
selection (CQFS). CQFS is open source, and reading it establishes directly that:

1. it encodes cardinality with the same penalty generator used here
   (`dimod.generators.combinations`, `core/CQFS.py`);
2. it samples with `neal`, the sampler shown above to fail on that encoding; and
3. it *also* uses tabu, via `dwave_qbsolv.QBSolv(solver='tabu')`.

Point 3 is corroboration rather than criticism: this project found independently that
tabu recovers what `neal` cannot on a penalty-encoded problem, and those authors
evidently found tabu worth using too. Two groups reaching the same workaround from
different directions suggests the difficulty is real rather than an artefact of one
implementation.

Points 1 and 2 mean the failure mode documented here is *available* in that setting.
Whether it occurs there is untested and this report makes no claim that it does: feature
selection over thousands of features is a different problem shape from choosing 10 items
from 200, the penalty-to-objective ratio differs, and CQFS sweeps the penalty strength
rather than fixing it.

What can be said is that the diagnostic is absent. There is no feasibility check and no
comparison of sampler energy against a greedy baseline anywhere in the CQFS core. Without
those, a run where the sampler returned an arbitrary feasible set is indistinguishable
from one where it optimised, because the downstream metrics remain plausible either way.
That is the practical content of §2 and it applied to this project too, before those
checks were written.

**What is new here** is therefore not the formulation, which is standard and owes its
shape to that prior work: it is the failure analysis, the constraint-preserving solver,
the disjoint-split evaluation protocol, and the fairness-budget framing.

## 5c. Why not an exact solver?

Selecting k items under a linear-plus-quadratic objective and a cardinality constraint is
a mixed-integer program. Built with a McCormick linearisation and solved by HiGHS, with
cardinality as a real constraint rather than a penalty (mean of 3 instances, k=10):

  n      exact (proven)   secs     qubo_tabu   secs    qubo_feasible   secs
  20        +0.828         0.4       +1.214     1.5       +1.214        0.06
  50        -3.913        27.6       -3.132     4.1       -3.417        0.19
  100       -5.543       113.6       -5.513     3.1       -5.165        0.37
  200       -6.930       642.2       -6.296     2.3       -5.988        0.64

Exact solving is tractable at reranking scale and impractical anyway: 642 s at n=200
against qubo_feasible's 0.64 s, a 1,000x gap for a 14% better objective. Offline that may
be the right trade; for a live request it is not a trade.

The recommendation is therefore a decision rule, not a winner. Below n=50 and offline,
use the MIP solver -- it is exact and 28 s is nothing in a batch. At n>=100 or online,
use qubo_tabu: within 0.031 of the proven optimum at n=100 in 1/37th of the time, 0.634 at n=200
in 1/284th. Never use penalty-encoded neal, which at n=200 scores +1.900 against greedy
top-k's +1.831 -- worse than a method that performs no search, measured against a proven
optimum.

This also qualifies section 2.1a. Tabu and swap annealing are *exactly* optimal at n<=30
where enumeration is possible, but near-optimal and degrading at realistic sizes. The
stronger claim was true of the regime it was measured in and false in general.

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
5. **Eight benchmarks, but only five catalogues, two domains and two retrieval models.**
   MovieLens adds a second domain; the MF benchmark adds a second retrieval model; the
   seller and category benchmarks add group partitions taken from real metadata rather
   than derived from the interactions being scored. Breadth beyond retail and film, and beyond ItemKNN and ALS,
   is untested.
6. **No physical quantum hardware.** Everything here is classical or quantum-*inspired*.

---

## 7. Reproducing

```bash
pip install -r requirements.txt
pytest tests/                                    # 345 tests

curl --create-dirs -o data/amazon_lb/Luxury_Beauty.csv \
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Luxury_Beauty.csv

# --method is required to reproduce the committed CSVs: without it the protocol also
# tunes qubo_sa, which meets no budget on any benchmark and is excluded from the
# cross-dataset comparison for that reason.
python experiments/protocol.py --config configs/amazon_lb.yaml \
  --tau 0.20 0.22 0.25 0.30 0.40 1.00 --repeats 3 --n-users 80 \
  --method greedy_topk mmr quota_mmr qubo_tabu qubo_feasible
python experiments/compare_datasets.py

# the n-ladder in section 2.1a. Previously unreproducible: --n-items took one integer
# and the CSV recorded no n, so the published table could not be regenerated.
python experiments/optimality.py --n-items 14 18 22 26 30 --k 5 --repeats 12
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
