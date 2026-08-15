"""Evaluation metrics: relevance, fairness/diversity, and energy."""

from .dpfr import individual_item_fairness, item_better_off
from .energy import EnergyReading, codecarbon_available, measure_energy
from .fairness import (
    catalogue_coverage,
    category_coverage,
    exposure_parity,
    gini,
    group_counts,
    intra_list_similarity,
    item_exposure,
)
from .relevance import dcg, ndcg_at_k, precision_at_k, recall_at_k

__all__ = [
    "dcg",
    "ndcg_at_k",
    "recall_at_k",
    "precision_at_k",
    "group_counts",
    "category_coverage",
    "exposure_parity",
    "gini",
    "catalogue_coverage",
    "item_exposure",
    "intra_list_similarity",
    "individual_item_fairness",
    "item_better_off",
    "measure_energy",
    "EnergyReading",
    "codecarbon_available",
]
