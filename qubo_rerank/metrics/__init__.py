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
    "EnergyReading",
    "catalogue_coverage",
    "category_coverage",
    "codecarbon_available",
    "dcg",
    "exposure_parity",
    "gini",
    "group_counts",
    "individual_item_fairness",
    "intra_list_similarity",
    "item_better_off",
    "item_exposure",
    "measure_energy",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
