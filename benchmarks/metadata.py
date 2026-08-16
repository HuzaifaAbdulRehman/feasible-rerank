"""Real seller and category partitions for the Amazon catalogues.

Every group partition used elsewhere in this project is *derived*: popularity tiers in
:mod:`benchmarks.loader`, genre buckets in :mod:`benchmarks.movielens`. Both are
defensible, but neither is what a marketplace operator actually owes fairness to. The
fairness literature and the industrial motivation both talk about **sellers** and
**product categories**, and until now this project could not, because the McAuley
ratings-only exports carry four columns and none of them is metadata.

The separate ``metaFiles2`` exports do carry it. This module joins them in.

**Coverage is not uniform, and that decides which catalogues can use this.** Measured on
the 5-core-filtered catalogues actually used by the benchmarks:

===================  =============  ==============  ===============
catalogue            items          with ``brand``  with ``category``
===================  =============  ==============  ===============
Software                       727   722 (99.3%)      680 (93.5%)
Gift Cards                       -   34.1% (raw)      99.7% (raw)
Luxury Beauty                 1366    12 (0.1%)         0 (0.0%)
===================  =============  ==============  ===============

Luxury Beauty's metadata file exists, parses cleanly, and has the ``brand`` and
``category`` keys present on all 12,299 records -- as empty strings. A partition built
from it would put every item in one group, which is not an error any assertion downstream
would catch: the fairness term would simply become a constant and every method would look
perfectly fair. :func:`vendor_tiers` and :func:`category_groups` therefore refuse to
build a partition below ``min_coverage`` rather than returning a degenerate one. That
guard is the whole reason this module has its own error type.

**Why vendor *tiers* rather than vendor names.** There are 4,378 distinct vendors in the
raw Software catalogue and 122 in the filtered one, against a list size of k=10. Naming
each vendor as its own group makes most groups empty in any given candidate set, and the
equal-share target ``k/|C|`` becomes unattainable by construction rather than by any
property of the method -- which would manufacture the feasibility result this project
reports. :func:`vendor_tiers` instead rank-orders items by *how large their vendor's
catalogue is* and cuts equal-sized item buckets, exactly as
:func:`benchmarks.loader.popularity_tiers` does for popularity. Tier 0 is the major
publishers (Microsoft, Intuit, Corel, Symantec); the last tier is the independents.

That is also the harm the motivating literature actually names -- recommendations
concentrating on "a limited set of products or vendors" -- expressed as a partition the
fairness term can act on.

**These are not popularity in disguise.** The obvious objection is that big vendors are
popular vendors, in which case this adds a second name for a partition the project
already has. Measured against the existing popularity tiers on filtered Software:

=================  =======  =======
partition            NMI      ARI
=================  =======  =======
vendor tier         0.0119  +0.0067
vendor name         0.1477  +0.0064
category            0.0163  +0.0096
=================  =======  =======

Zero is independence and one is identity, so all three are effectively orthogonal to
popularity. Reproduce with ``python benchmarks/metadata.py --data ... --meta ...``.

**Categories get proportional targets, not equal ones.** The 22 real Software categories
are genuinely unequal -- 121 items in Digital Software against 36 in Video -- and equal
share would demand a list composition the catalogue cannot supply. Real category sizes
are therefore carried through as proportional targets via
:func:`benchmarks.metadata.proportional_targets`. This is the case classical quota
reranking cannot express at all (see ``docs/findings.md``), so the category partition is
also the one that exercises the formulation's one uncontested advantage on real data.

Download the metadata for a catalogue with::

    python benchmarks/metadata.py --download Software --out data/meta_Software.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: ``{category}`` is the same token used in the ratings filename, e.g. ``Software``.
METADATA_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/"
    "metaFiles2/meta_{category}.json.gz"
)


class MetadataCoverageError(RuntimeError):
    """Raised when too few catalogue items carry the requested field.

    Distinct from :class:`ValueError` so that a caller sweeping several catalogues can
    skip the ones without usable metadata without also swallowing genuine shape bugs.
    """


@dataclass(frozen=True)
class ItemMetadata:
    """Item-level side information, keyed by the ratings file's ``item_id`` (an ASIN).

    Attributes:
        vendor: ASIN -> brand/publisher. Missing and blank brands are absent, not "".
        category: ASIN -> the full category path, root first.
    """

    vendor: Mapping[str, str]
    category: Mapping[str, tuple[str, ...]]

    def vendor_coverage(self, item_ids: Sequence[str]) -> float:
        if not len(item_ids):
            return 0.0
        return sum(1 for i in item_ids if i in self.vendor) / len(item_ids)

    def category_coverage(self, item_ids: Sequence[str]) -> float:
        if not len(item_ids):
            return 0.0
        return sum(1 for i in item_ids if i in self.category) / len(item_ids)


def load_metadata(path: str | Path) -> ItemMetadata:
    """Parse a gzipped ``meta_*.json.gz`` export into an :class:`ItemMetadata`.

    The export is JSON-lines. A small number of records in the public files are
    malformed; they are skipped rather than aborting the load, which is the same
    tolerance the upstream loaders use. Blank ``brand`` values -- the dominant case in
    some catalogues -- are treated as *missing* rather than as a vendor literally named
    "", because the difference decides whether the coverage guard fires.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it with:\n"
            f"    python benchmarks/metadata.py --download <Category> --out {path}"
        )

    vendor: dict[str, str] = {}
    category: dict[str, tuple[str, ...]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = record.get("asin")
            if not asin:
                continue
            brand = (record.get("brand") or "").strip()
            if brand:
                vendor[asin] = brand
            path_ = record.get("category") or []
            cleaned = tuple(c.strip() for c in path_ if isinstance(c, str) and c.strip())
            if cleaned:
                category[asin] = cleaned
    return ItemMetadata(vendor=vendor, category=category)


def vendor_tiers(
    item_ids: Sequence[str],
    metadata: ItemMetadata,
    n_tiers: int = 4,
    min_coverage: float = 0.80,
) -> np.ndarray:
    """Cut items into equal-sized vendor-concentration tiers; tier 0 is the majors.

    The seller analogue of :func:`benchmarks.loader.popularity_tiers`: items are
    rank-ordered by their vendor's catalogue count and split into equal *item* counts, so
    the fairness term's default equal-share targets stay attainable.

    Items whose vendor is unknown are assigned vendor size 0, which places them in the
    final tier alongside the smallest independents. That is the semantically right
    default -- an item absent from the brand export is not a major publisher's -- but it
    is a default, so the coverage guard keeps it from applying to most of the catalogue.

    Raises:
        MetadataCoverageError: if fewer than ``min_coverage`` of ``item_ids`` have a
            vendor. Without this, a catalogue with empty metadata silently yields one
            group and a fairness term that is identically zero.
    """
    item_ids = list(item_ids)
    coverage = metadata.vendor_coverage(item_ids)
    if coverage < min_coverage:
        raise MetadataCoverageError(
            f"only {coverage:.1%} of {len(item_ids)} catalogue items carry a vendor "
            f"(need {min_coverage:.0%}). Amazon's Luxury Beauty metadata is present but "
            f"blank, for instance. Use grouping='popularity' for that catalogue."
        )

    vendors = [metadata.vendor.get(i, "") for i in item_ids]
    counts = Counter(v for v in vendors if v)
    size = np.array([counts.get(v, 0) for v in vendors], dtype=float)

    n_items = len(item_ids)
    # Rank 0 = largest vendor. Negate so argsort descends; the stable double argsort
    # turns the ordering into a per-item rank, matching popularity_tiers exactly.
    rank = np.argsort(np.argsort(-size, kind="stable"), kind="stable")
    return (rank * n_tiers // n_items).astype(int)


def category_groups(
    item_ids: Sequence[str],
    metadata: ItemMetadata,
    n_groups: int = 4,
    level: int = 1,
    min_coverage: float = 0.80,
) -> tuple[np.ndarray, list[str]]:
    """Group items by real product category: the ``n_groups - 1`` largest, plus a pool.

    Args:
        level: which node of the category path to cut at. Level 0 is the catalogue root
            ("Software") and is constant, so level 1 is the first informative node.
        n_groups: total groups including the pooled remainder.

    Returns:
        ``(labels, names)`` where ``labels[i]`` indexes ``names``. The final name is
        always the pooled ``"other"`` bucket, which also absorbs items with no category.

    Unlike :func:`vendor_tiers` these groups are deliberately *unequal*, because real
    categories are. Pair them with :func:`proportional_targets` rather than with the
    equal-share default.

    Raises:
        MetadataCoverageError: if fewer than ``min_coverage`` items carry a category.
    """
    item_ids = list(item_ids)
    coverage = metadata.category_coverage(item_ids)
    if coverage < min_coverage:
        raise MetadataCoverageError(
            f"only {coverage:.1%} of {len(item_ids)} catalogue items carry a category "
            f"(need {min_coverage:.0%})."
        )
    if n_groups < 2:
        raise ValueError(f"n_groups must be at least 2, got {n_groups}")

    def key(asin: str) -> str:
        path = metadata.category.get(asin) or ()
        if len(path) > level:
            return path[level]
        return path[-1] if path else ""

    keys = [key(i) for i in item_ids]
    counts = Counter(k for k in keys if k)
    # Sort by count then name so the partition is reproducible across runs rather than
    # depending on dict iteration order for equally-sized categories.
    top = [name for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    top = top[: n_groups - 1]
    index = {name: j for j, name in enumerate(top)}

    labels = np.array([index.get(k, n_groups - 1) for k in keys], dtype=int)
    return labels, [*top, "other"]


def proportional_targets(groups: np.ndarray, k: int) -> dict[int, float]:
    """Per-group target counts proportional to each group's share of the candidates.

    Equal targets punish a large group for being large. For the real category partition
    the group sizes are the ground truth about what the catalogue can supply, so the
    target vector should follow them.

    This is the target shape classical quota reranking cannot express -- it allocates
    integer quotas round-robin -- while the QUBO takes an arbitrary real-valued vector.
    """
    groups = np.asarray(groups).ravel()
    if groups.size == 0:
        return {}
    unique, counts = np.unique(groups, return_counts=True)
    share = counts / counts.sum()
    return {int(g): float(k * s) for g, s in zip(unique, share, strict=True)}


# ----------------------------------------------------------------------------- cli


def _download(category: str, out: Path) -> None:
    import urllib.request

    url = METADATA_URL.format(category=category)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def _agreement(a: Sequence, b: Sequence) -> tuple[float, float]:
    """Normalised mutual information and adjusted Rand between two partitions.

    Implemented here rather than pulled from scikit-learn: these two scores are the only
    thing that would need it, and this project's dependency list is short on purpose.
    """
    ua = {v: i for i, v in enumerate(sorted(set(a), key=str))}
    ub = {v: i for i, v in enumerate(sorted(set(b), key=str))}
    m = np.zeros((len(ua), len(ub)))
    for x, y in zip(a, b, strict=True):
        m[ua[x], ub[y]] += 1

    def choose2(x):
        return (x * (x - 1) / 2).sum()

    n = m.sum()
    index, row, col = choose2(m), choose2(m.sum(1)), choose2(m.sum(0))
    expected = row * col / (n * (n - 1) / 2)
    ari = (index - expected) / (0.5 * (row + col) - expected)

    p = m / n
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(np.where(p > 0, p * np.log(p / (pa * pb)), 0.0))

    def entropy(q):
        return -np.nansum(np.where(q > 0, q * np.log(q), 0.0))

    denom = max(float(np.sqrt(entropy(pa.ravel()) * entropy(pb.ravel()))), 1e-12)
    return float(mi / denom), float(ari)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--download", help="catalogue name, e.g. Software")
    parser.add_argument("--out", type=Path, help="destination for --download")
    parser.add_argument("--data", type=Path, help="ratings CSV to report coverage against")
    parser.add_argument("--meta", type=Path, help="metadata .json.gz")
    parser.add_argument("--min-interactions", type=int, default=5)
    parser.add_argument("--n-groups", type=int, default=4)
    args = parser.parse_args(argv)

    if args.download:
        if args.out is None:
            parser.error("--download requires --out")
        _download(args.download, args.out)
        return 0

    if not (args.data and args.meta):
        parser.error("give either --download/--out, or --data/--meta")

    from benchmarks.loader import (
        interaction_matrix,
        k_core,
        leave_one_out,
        load_ratings,
        popularity_tiers,
    )

    raw = load_ratings(args.data)
    train, _ = leave_one_out(k_core(raw, min_interactions=args.min_interactions))
    matrix, _, item_ids = interaction_matrix(train, binary=True)
    metadata = load_metadata(args.meta)

    popularity = np.asarray(matrix.sum(axis=0)).ravel()
    pop = popularity_tiers(popularity, n_tiers=args.n_groups)

    print(f"catalogue after {args.min_interactions}-core: {len(item_ids)} items")
    print(f"  vendor coverage  : {metadata.vendor_coverage(item_ids):.1%}")
    print(f"  category coverage: {metadata.category_coverage(item_ids):.1%}")
    print()
    print("agreement with the popularity partition (0 = independent, 1 = identical)")
    for name, labels in [
        ("vendor tier", vendor_tiers(item_ids, metadata, n_tiers=args.n_groups)),
        ("vendor name", [metadata.vendor.get(i, "") for i in item_ids]),
        ("category", category_groups(item_ids, metadata, n_groups=args.n_groups)[0]),
    ]:
        try:
            nmi, ari = _agreement(pop, labels)
        except MetadataCoverageError as exc:  # pragma: no cover - catalogue dependent
            print(f"  {name:<14} unavailable: {exc}")
            continue
        print(f"  {name:<14} NMI {nmi:.4f}   ARI {ari:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
