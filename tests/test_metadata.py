"""Tests for the real seller and product-category partitions.

The failure mode this module exists to prevent is silent and total. A catalogue whose
metadata export has the ``brand`` key present but blank -- Amazon's Luxury Beauty, for
12,287 of its 12,299 records -- yields a partition with every item in one group. Nothing
downstream errors: the fairness term becomes identically zero, every method scores
perfect parity, and the resulting table looks like a *result*. So the guard against that
is tested before anything else here.

The second concern is subtler. A vendor partition that merely recovers the popularity
tiers would add a second name for a fairness axis the project already measures, and
reporting it as a distinct benchmark would inflate the evidence. The independence of the
partitions is therefore asserted, not assumed.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.metadata import (
    ItemMetadata,
    MetadataCoverageError,
    _agreement,
    category_groups,
    load_metadata,
    proportional_targets,
    vendor_tiers,
)


@pytest.fixture
def metadata() -> ItemMetadata:
    """A catalogue with one dominant vendor, two mid-sized, and a long tail."""
    vendor = {}
    for i in range(10):
        vendor[f"big{i}"] = "Megacorp"
    for i in range(4):
        vendor[f"mid{i}"] = "Middling"
    for i in range(4):
        vendor[f"small{i}"] = f"Indie{i}"
    category = {
        **{f"big{i}": ("Software", "Utilities", "Backup") for i in range(10)},
        **{f"mid{i}": ("Software", "Games") for i in range(4)},
        **{f"small{i}": ("Software", "Education") for i in range(4)},
    }
    return ItemMetadata(vendor=vendor, category=category)


@pytest.fixture
def item_ids() -> list[str]:
    return (
        [f"big{i}" for i in range(10)]
        + [f"mid{i}" for i in range(4)]
        + [f"small{i}" for i in range(4)]
    )


class TestCoverageGuard:
    """The guard that stops a blank metadata file from becoming a fake result."""

    def test_vendor_refuses_a_catalogue_with_blank_brands(self):
        """Luxury Beauty in miniature: the key exists, the value is empty."""
        empty = ItemMetadata(vendor={}, category={})

        with pytest.raises(MetadataCoverageError, match=r"0\.0%"):
            vendor_tiers([f"item{i}" for i in range(50)], empty)

    def test_category_refuses_a_catalogue_without_categories(self):
        empty = ItemMetadata(vendor={"a": "X"}, category={})

        with pytest.raises(MetadataCoverageError, match="need 80%"):
            category_groups(["a", "b", "c"], empty)

    def test_partial_coverage_below_threshold_still_raises(self, metadata):
        """Half a catalogue is not enough; the unknown half would all land in one tier."""
        ids = [f"big{i}" for i in range(10)] + [f"ghost{i}" for i in range(10)]

        with pytest.raises(MetadataCoverageError):
            vendor_tiers(ids, metadata, min_coverage=0.8)

    def test_coverage_above_threshold_is_allowed(self, metadata, item_ids):
        tiers = vendor_tiers([*item_ids, "ghost"], metadata, min_coverage=0.8)

        assert len(tiers) == len(item_ids) + 1

    def test_blank_brand_is_missing_rather_than_a_vendor_named_empty(self, tmp_path):
        """The distinction decides whether the coverage guard can ever fire."""
        path = tmp_path / "meta.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(json.dumps({"asin": "a", "brand": "", "category": []}) + "\n")
            fh.write(json.dumps({"asin": "b", "brand": "  ", "category": []}) + "\n")
            fh.write(json.dumps({"asin": "c", "brand": "Real", "category": []}) + "\n")

        loaded = load_metadata(path)

        assert loaded.vendor == {"c": "Real"}
        assert loaded.vendor_coverage(["a", "b", "c"]) == pytest.approx(1 / 3)


class TestVendorTiers:
    def test_tiers_are_equal_sized(self, metadata, item_ids):
        """Equal item counts per tier keep the equal-share target attainable, which is
        the same contract popularity_tiers holds to."""
        tiers = vendor_tiers(item_ids, metadata, n_tiers=2)

        counts = np.bincount(tiers, minlength=2)
        assert counts.tolist() == [9, 9]

    def test_tier_zero_holds_the_largest_vendor(self, metadata, item_ids):
        tiers = vendor_tiers(item_ids, metadata, n_tiers=3)

        assert all(tiers[item_ids.index(f"big{i}")] == 0 for i in range(6))

    def test_independents_land_in_the_last_tier(self, metadata, item_ids):
        tiers = vendor_tiers(item_ids, metadata, n_tiers=3)

        assert {int(tiers[item_ids.index(f"small{i}")]) for i in range(4)} == {2}

    def test_unknown_vendor_is_treated_as_smallest(self, metadata, item_ids):
        """An item absent from the brand export is not a major publisher's, so the
        default has to place it with the independents rather than the majors."""
        ids = [*item_ids, "ghost"]
        tiers = vendor_tiers(ids, metadata, n_tiers=2, min_coverage=0.5)

        assert tiers[-1] == 1

    def test_is_deterministic(self, metadata, item_ids):
        first = vendor_tiers(item_ids, metadata, n_tiers=4)
        second = vendor_tiers(item_ids, metadata, n_tiers=4)

        assert np.array_equal(first, second)


class TestCategoryGroups:
    def test_keeps_the_largest_categories_and_pools_the_rest(self, metadata, item_ids):
        labels, names = category_groups(item_ids, metadata, n_groups=3)

        # Level 1 of ("Software", "Utilities", "Backup") is "Utilities". Education and
        # Games are both size 4, so the name tie-break puts Education in and pools Games.
        assert names == ["Utilities", "Education", "other"]
        assert np.bincount(labels, minlength=3).tolist() == [10, 4, 4]

    def test_level_selects_the_node_of_the_category_path(self, metadata, item_ids):
        """Level 0 is the catalogue root and constant, so it must collapse to one group
        -- the check that the level argument is doing what it claims."""
        labels, names = category_groups(item_ids, metadata, n_groups=3, level=0)

        assert names[0] == "Software"
        assert set(np.unique(labels).tolist()) == {0}

    def test_items_without_a_category_go_to_the_pool(self, metadata, item_ids):
        ids = [*item_ids, "ghost"]
        labels, names = category_groups(ids, metadata, n_groups=3, min_coverage=0.5)

        assert names[-1] == "other"
        assert labels[-1] == 2

    def test_ties_break_by_name_so_the_partition_is_reproducible(self):
        """Two categories of equal size must not be ordered by dict insertion."""
        meta = ItemMetadata(
            vendor={},
            category={"a": ("R", "Zeta"), "b": ("R", "Alpha")},
        )

        _, names = category_groups(["a", "b"], meta, n_groups=3)

        assert names == ["Alpha", "Zeta", "other"]

    def test_rejects_fewer_than_two_groups(self, metadata, item_ids):
        with pytest.raises(ValueError, match="at least 2"):
            category_groups(item_ids, metadata, n_groups=1)


class TestProportionalTargets:
    def test_targets_sum_to_k(self):
        groups = np.array([0, 0, 0, 1, 1, 2])

        targets = proportional_targets(groups, k=10)

        assert sum(targets.values()) == pytest.approx(10.0)

    def test_targets_follow_group_share(self):
        groups = np.array([0] * 75 + [1] * 25)

        targets = proportional_targets(groups, k=8)

        assert targets == {0: pytest.approx(6.0), 1: pytest.approx(2.0)}

    def test_equal_groups_reduce_to_equal_share(self):
        """The proportional rule must not disagree with the equal rule where the two
        coincide, or the auto mode would change results for the wrong reason."""
        groups = np.array([0, 0, 1, 1])

        assert proportional_targets(groups, k=10) == {
            0: pytest.approx(5.0),
            1: pytest.approx(5.0),
        }

    def test_empty_groups_give_no_targets(self):
        assert proportional_targets(np.array([]), k=10) == {}


class TestAgreementScores:
    """The independence claim in the module docstring is a measurement, so the thing
    that measures it gets tested against partitions whose answer is known."""

    def test_identical_partitions_score_one(self):
        a = [0, 0, 1, 1, 2, 2]

        nmi, ari = _agreement(a, a)

        assert nmi == pytest.approx(1.0)
        assert ari == pytest.approx(1.0)

    def test_orthogonal_partitions_score_near_zero(self):
        a = [0, 0, 1, 1]
        b = [0, 1, 0, 1]

        nmi, ari = _agreement(a, b)

        assert nmi == pytest.approx(0.0, abs=1e-9)
        assert ari == pytest.approx(-0.5, abs=1e-9)

    def test_relabelling_does_not_change_the_score(self):
        a = [0, 0, 1, 1]
        relabelled = ["x", "x", "y", "y"]

        assert _agreement(a, relabelled)[1] == pytest.approx(1.0)


class TestLoadMetadata:
    def test_skips_malformed_lines_rather_than_aborting(self, tmp_path):
        path = tmp_path / "meta.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write("{not json\n")
            fh.write(json.dumps({"asin": "a", "brand": "X", "category": ["R", "C"]}) + "\n")

        loaded = load_metadata(path)

        assert loaded.vendor == {"a": "X"}
        assert loaded.category == {"a": ("R", "C")}

    def test_missing_file_names_the_download_command(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="--download"):
            load_metadata(tmp_path / "absent.json.gz")

    def test_records_without_an_asin_are_ignored(self, tmp_path):
        path = tmp_path / "meta.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(json.dumps({"brand": "Orphan"}) + "\n")

        assert load_metadata(path).vendor == {}
