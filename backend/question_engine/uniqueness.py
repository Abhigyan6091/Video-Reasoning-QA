"""
Uniqueness filter – removes near-duplicate questions using embedding similarity.
"""
from __future__ import annotations

import logging

import numpy as np

from backend.config import settings
from backend.scene_graph.memory import compute_pairwise_similarity

logger = logging.getLogger(__name__)


def filter_duplicates(questions: list[dict],
                      threshold: float | None = None) -> list[dict]:
    """
    Remove near-duplicate questions based on cosine similarity.

    Process:
    1. Compute pairwise cosine similarity of question texts
    2. For each pair above threshold, keep the one with higher novelty
    3. Assign novelty scores

    Returns filtered list of questions.
    """
    if not questions:
        return []

    thresh = threshold or settings.SIMILARITY_THRESHOLD
    texts = [q["question_text"] for q in questions]

    # Compute similarity matrix
    sim_matrix = compute_pairwise_similarity(texts)
    n = len(questions)

    # Calculate novelty scores
    # Novelty = inverse of average similarity to all other questions
    for i in range(n):
        # Average similarity excluding self
        similarities = [float(sim_matrix[i][j]) for j in range(n) if j != i]
        avg_sim = np.mean(similarities) if similarities else 0.0
        questions[i]["novelty_score"] = round(1.0 - avg_sim, 4)

    # Find duplicates: for each pair above threshold, mark the one
    # with lower novelty for removal
    to_remove: set[int] = set()
    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            if float(sim_matrix[i][j]) > thresh:
                # Remove the less novel one (or the later one if tied)
                if questions[i]["novelty_score"] >= questions[j]["novelty_score"]:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break  # i is removed, stop checking pairs for i

    filtered = [q for idx, q in enumerate(questions) if idx not in to_remove]
    removed_count = len(questions) - len(filtered)

    if removed_count > 0:
        logger.info(f"Removed {removed_count} duplicate questions "
                    f"(threshold={thresh}). {len(filtered)} remain.")

    return filtered


def deduplicate_across_categories(questions: list[dict],
                                  threshold: float | None = None) -> list[dict]:
    """
    Cross-category deduplication – questions from different categories
    can still be similar in phrasing.
    """
    return filter_duplicates(questions, threshold)
