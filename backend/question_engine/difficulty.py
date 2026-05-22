"""
Difficulty estimation for generated questions.

Classifies questions into Easy / Medium / Hard / Expert based on:
  - Number of scenes involved
  - Reasoning depth
  - Temporal hops required
  - Inference complexity
"""
from __future__ import annotations


def estimate_difficulty(question: dict, total_scenes: int) -> dict:
    """
    Score and classify difficulty for a single question.
    Returns the question dict augmented with difficulty and difficulty_score.
    """
    score = 0.0

    # ── Factor 1: Scenes involved ──────────────────────────────────
    scenes = question.get("scenes_involved", [])
    num_scenes = len(scenes) if isinstance(scenes, list) else 0

    if num_scenes <= 1:
        score += 1.0
    elif num_scenes <= 2:
        score += 2.5
    elif num_scenes <= 4:
        score += 4.0
    else:
        score += 5.0

    # ── Factor 2: Reasoning depth (from LLM) ──────────────────────
    depth = question.get("reasoning_depth", 1)
    if isinstance(depth, (int, float)):
        score += min(depth * 0.8, 4.0)

    # ── Factor 3: Category complexity ──────────────────────────────
    category_weights = {
        "temporal": 1.5,
        "causal": 2.5,
        "counterfactual": 3.5,
        "contradiction": 3.0,
        "emotion": 2.0,
        "multi_scene": 4.0,
        "symbolic": 3.5,
    }
    cat = question.get("category", "")
    score += category_weights.get(cat, 2.0)

    # ── Factor 4: Temporal hops (how spread out are the scenes) ────
    if num_scenes >= 2 and total_scenes > 1:
        if isinstance(scenes, list) and all(isinstance(s, (int, float)) for s in scenes):
            span = max(scenes) - min(scenes)
            hop_ratio = span / max(total_scenes - 1, 1)
            score += hop_ratio * 2.0

    # ── Factor 5: Question length as proxy for complexity ──────────
    q_text = question.get("question_text", "")
    word_count = len(q_text.split())
    if word_count > 25:
        score += 1.0
    elif word_count > 15:
        score += 0.5

    # ── Normalize to 0–10 range ────────────────────────────────────
    max_possible = 5.0 + 4.0 + 4.0 + 2.0 + 1.0  # 16.0
    normalized = min(score / max_possible * 10, 10.0)

    # ── Classify ───────────────────────────────────────────────────
    if normalized <= 2.5:
        difficulty = "easy"
    elif normalized <= 5.0:
        difficulty = "medium"
    elif normalized <= 7.5:
        difficulty = "hard"
    else:
        difficulty = "expert"

    question["difficulty"] = difficulty
    question["difficulty_score"] = round(normalized, 2)
    return question


def estimate_all(questions: list[dict], total_scenes: int) -> list[dict]:
    """Score and classify all questions."""
    return [estimate_difficulty(q, total_scenes) for q in questions]
