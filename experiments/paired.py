"""Paired per-user significance tests between rerankers.

Every method in this benchmark sees the **identical candidate set** for the identical
user. That makes their per-user scores paired observations, and pairing is worth a great
deal here: comparing means over 3 seeds is a test with n=3, while comparing per-user
differences over 200 users is a test with n=200 on data already collected. The seeds were
never the sample -- the users are.

Pairing also removes the dominant source of variance. Users differ enormously in how
predictable they are; a user whose held-out purchase never entered the candidate set
scores badly under every method. Comparing unpaired means makes every method carry that
noise. Differencing within a user cancels it exactly.

**Test.** Wilcoxon signed-rank, two-sided, on the per-user differences. Not a paired
t-test: per-user NDCG is bounded on [0, 1], strongly left-skewed, and spikes at 1.0, so
normality is not a safe assumption at this sample size. Wilcoxon assumes only that the
differences are symmetric about their median.

**Correction.** Holm-Bonferroni across the whole family of comparisons reported in one
run. Testing five methods on three metrics is fifteen chances to find a p below 0.05.

**Effect size is reported next to every p-value, and matters more.** With 200 users a
difference of 0.002 NDCG can be significant and still be worth nothing. Each row carries
the median paired difference, a bootstrap confidence interval for it, and the plain count
of users better off, worse off, and tied.

Usage::

    python experiments/paired.py --config configs/amazon_lb.yaml \
        --lam 0.0 --mu 1.0 --n-users 200 --reference quota_mmr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_experiment import (
    PER_USER_METRICS,
    build_benchmark,
    build_solvers,
    score_selections,
)

#: Direction of improvement. Used only to phrase the verdict; the test itself is
#: two-sided and never assumes which way a difference should go.
LOWER_IS_BETTER = {"exposure_parity", "intra_list_sim"}

N_BOOTSTRAP = 10_000


def collect_per_user(cfg: dict, bench) -> pd.DataFrame:
    """Score every enabled solver on every user. One row per (method, user)."""
    frames = []
    for solver in build_solvers(cfg):
        selections = [solver.solve(inst).selection for inst in bench.instances]
        scored = score_selections(bench, selections)

        rows = {"method": solver.name, "user": np.arange(len(bench.instances))}
        for metric, values in scored.per_user.items():
            if values:  # recall is empty on benchmarks without ground truth
                rows[metric] = values
        frames.append(pd.DataFrame(rows))
        print(f"  scored {solver.name:<14} on {len(bench.instances)} users")

    return pd.concat(frames, ignore_index=True)


def bootstrap_ci(diff: np.ndarray, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval for the median paired difference."""
    if diff.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(N_BOOTSTRAP, diff.size))
    medians = np.median(diff[idx], axis=1)
    return (
        float(np.quantile(medians, alpha / 2)),
        float(np.quantile(medians, 1 - alpha / 2)),
    )


def compare(a: np.ndarray, b: np.ndarray, metric: str, seed: int = 0) -> dict:
    """One paired comparison: ``a`` against ``b``, per user.

    Returns the median difference, its bootstrap interval, win/loss/tie counts, the
    Wilcoxon statistic and an uncorrected p-value. Correction happens once across the
    whole family, in :func:`holm`.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    nonzero = diff[diff != 0.0]

    better = diff < 0 if metric in LOWER_IS_BETTER else diff > 0
    worse = diff > 0 if metric in LOWER_IS_BETTER else diff < 0

    if nonzero.size == 0:
        # Identical on every user. This is a real outcome, not an error: two solvers can
        # return the same lists. Reporting p=1 is correct; running the test would raise.
        p = 1.0
        statistic = float("nan")
    else:
        statistic, p = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")

    lo, hi = bootstrap_ci(diff, seed=seed)
    return {
        "metric": metric,
        "n_users": int(diff.size),
        "median_diff": float(np.median(diff)),
        "ci_lo": lo,
        "ci_hi": hi,
        "better": int(better.sum()),
        "worse": int(worse.sum()),
        "tied": int((diff == 0.0).sum()),
        "statistic": float(statistic),
        "p_raw": float(p),
    }


def holm(rows: list[dict], alpha: float = 0.05) -> list[dict]:
    """Holm-Bonferroni step-down correction across the whole reported family.

    Holm rather than plain Bonferroni: it is uniformly more powerful and makes no extra
    assumptions. Once a hypothesis fails to be rejected, every larger p-value is held
    non-significant too -- that step-down rule is what controls the family-wise error
    rate, and skipping it is the usual way corrections are got wrong.
    """
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_raw"])
    m = len(rows)
    still_rejecting = True

    for rank, i in enumerate(order):
        adjusted = min(1.0, rows[i]["p_raw"] * (m - rank))
        # Enforce monotonicity: an adjusted p can never fall below an earlier one.
        if rank > 0:
            adjusted = max(adjusted, rows[order[rank - 1]]["p_holm"])
        rows[i]["p_holm"] = adjusted
        if still_rejecting and adjusted > alpha:
            still_rejecting = False
        rows[i]["significant"] = still_rejecting

    return rows


def run_comparisons(
    per_user: pd.DataFrame, reference: str, metrics: list[str], seed: int = 0
) -> pd.DataFrame:
    """Compare every method against ``reference`` on every metric, then correct once."""
    methods = [m for m in per_user["method"].unique() if m != reference]
    if reference not in set(per_user["method"]):
        raise SystemExit(f"reference {reference!r} not among {sorted(set(per_user.method))}")

    ref = per_user[per_user.method == reference].sort_values("user")
    rows = []
    for method in methods:
        this = per_user[per_user.method == method].sort_values("user")
        for metric in metrics:
            if metric not in per_user.columns or per_user[metric].isna().all():
                continue
            rows.append(
                {
                    "method": method,
                    "reference": reference,
                    **compare(this[metric].to_numpy(), ref[metric].to_numpy(), metric, seed),
                }
            )

    return pd.DataFrame(holm(rows))


def verdict(row: pd.Series) -> str:
    """Plain-language reading of one row, effect size first."""
    if not row["significant"]:
        return "no detectable difference"
    direction = "better" if row["better"] > row["worse"] else "worse"
    return f"{direction} (n={row['better']} vs {row['worse']})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--reference", default="quota_mmr",
                    help="method every other method is compared against")
    # The default is the family the published tables report, and that is not cosmetic:
    # Holm's step-down multiplier is the family size, so running three metrics instead
    # of four changes every corrected p-value in the write-up. The documented command
    # passes no --metric, so the default has to be the published family or the numbers
    # cannot be reproduced from the instructions given.
    #
    # `exposure_parity_equal` is deliberately excluded: it is the same selections read
    # against a second definition, so including it would inflate the family with a
    # comparison that is not independent.
    ap.add_argument("--metric", nargs="+",
                    default=["ndcg@k", "exposure_parity", "recall@k", "intra_list_sim"],
                    help=f"paired metrics; available: {', '.join(PER_USER_METRICS)}")
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--mu", type=float, default=None)
    # Without this the comparison is asymmetric and quietly so. `--lam/--mu` move the
    # QUBO to a protocol-selected operating point, while the classical baselines keep
    # whatever `mmr_lam` the config happens to carry -- 0.5 everywhere in this repo,
    # which is a value the protocol never selects (it picks 0.3 under tight budgets and
    # 0.7 under loose ones). Tuning one side and not the other is the exact asymmetry
    # that produced this project's first, wrong conclusion, pointed the other way.
    ap.add_argument("--mmr-lam", type=float, default=None,
                    help="trade-off for mmr / quota_mmr / balanced_quota; without it "
                         "they run at the config default while the QUBO runs tuned")
    ap.add_argument("--n-users", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.n_users:
        cfg["data"]["n_users"] = args.n_users
    if args.lam is not None:
        cfg["lam"] = args.lam
    if args.mu is not None:
        cfg["mu"] = args.mu
    if args.mmr_lam is not None:
        cfg.setdefault("solvers", {})["mmr_lam"] = args.mmr_lam

    bench = build_benchmark(cfg)
    print(
        f"paired tests on {len(bench.instances)} users   "
        f"lam={cfg.get('lam')}  mu={cfg.get('mu')}   reference={args.reference}\n"
    )

    per_user = collect_per_user(cfg, bench)
    results = run_comparisons(per_user, args.reference, args.metric)
    results = results.sort_values(["metric", "p_holm"])

    out = args.out or (ROOT / "results" / f"{args.config.stem}_paired.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)

    print(f"\nWilcoxon signed-rank, two-sided, Holm-corrected across "
          f"{len(results)} comparisons (alpha={args.alpha}).")
    print(f"Every method compared against {args.reference} on the same users.\n")

    for metric, block in results.groupby("metric"):
        arrow = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
        print(f"{metric}  ({arrow})")
        print(f"  {'method':<15} {'median diff':>12} {'95% CI':>22} "
              f"{'better/worse/tied':>19} {'p (Holm)':>10}   verdict")
        for _, row in block.iterrows():
            counts = f"{row['better']}/{row['worse']}/{row['tied']}"
            print(
                f"  {row['method']:<15} {row['median_diff']:>12.4f} "
                f"[{row['ci_lo']:>8.4f},{row['ci_hi']:>8.4f}] "
                f"{counts:>19} {row['p_holm']:>10.2e}   {verdict(row)}"
            )
        print()

    print(f"wrote {out}")
    print(
        "\nEffect size before p-value: with this many users a difference can be "
        "significant and still be too small to matter. Read the CI, not the star."
    )


if __name__ == "__main__":
    main()
