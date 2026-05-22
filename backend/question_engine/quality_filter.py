"""
Quality filter for generated questions – removes low-quality, vague, or garbage questions.

Checks:
  1. Relevance to transcript/video content
  2. Clarity and proper grammar
  3. No vague/generic questions (e.g., "What did you see?")
  4. Audio-visual correlation present
  5. No question length extremes
  6. Proper MCQ structure with valid options
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Threshold below which questions are rejected
QUALITY_THRESHOLD = 0.6

# Generic/vague question patterns that should be filtered
VAGUE_PATTERNS = [
    r"^what.*see.*\?$",
    r"^what.*hear.*\?$",
    r"^what.*happen.*\?$",
    r"^describe.*\?$",
    r"^tell.*about.*\?$",
    r"^how.*feel.*\?$",
    r"^is this.*\?$",
    r"^can you.*\?$",
    r"^do you.*\?$",
]

# Terms that should be replaced or avoided
BANNED_TERMS = [
    r"\bscene\b",  # Replace with context-specific terms
    r"\breference\s+scene",
    r"\bcompare\s+scenes",
    r"\brelation\s+between\s+scenes",
]


def _check_grammar_and_clarity(question_text: str) -> float:
    """
    Simple heuristic for grammar/clarity.
    Returns 0.0-1.0 score.
    """
    score = 1.0
    
    # Check basic structure (should have a question mark)
    if not question_text.strip().endswith("?"):
        score -= 0.3
    
    # Check minimum length (too short is vague)
    if len(question_text.split()) < 5:
        score -= 0.4
    
    # Check maximum length (too long is likely rambling)
    if len(question_text.split()) > 50:
        score -= 0.2
    
    # Check for multiple question marks (sign of poor structure)
    if question_text.count("?") > 1:
        score -= 0.3
    
    # Check for ALL CAPS (usually indicates poor quality)
    if question_text.isupper() and len(question_text) > 10:
        score -= 0.3
    
    return max(score, 0.0)


def _check_for_vague_patterns(question_text: str) -> float:
    """
    Check if question matches vague/generic patterns.
    Returns penalty (0.0-1.0, where 1.0 is perfect).
    """
    question_lower = question_text.lower().strip()
    
    for pattern in VAGUE_PATTERNS:
        if re.search(pattern, question_lower):
            return 0.3  # Heavily penalize vague patterns
    
    return 1.0


def _check_audio_visual_alignment(question_text: str, answer_text: str) -> float:
    """
    Check if answer explicitly mentions both audio (transcript/narrator/spoken) 
    and visual elements (shown/seen/UI/display/etc).
    
    Returns 0.0-1.0 score.
    """
    score = 0.5  # Base score
    
    answer_lower = answer_text.lower() if answer_text else ""
    
    # Check for audio indicators
    audio_keywords = ["narrator", "spoken", "says", "explains", "mentions", "claims", "audio"]
    if any(kw in answer_lower for kw in audio_keywords):
        score += 0.25
    
    # Check for visual indicators
    visual_keywords = ["shown", "seen", "display", "ui", "visual", "frame", "clip", "demonstrates", "illustrates"]
    if any(kw in visual_keywords for kw in visual_keywords):
        score += 0.25
    
    return min(score, 1.0)


def _check_mcq_structure(question_data: dict) -> float:
    """
    Validate MCQ structure: must have valid options and correct answer index.
    Returns 0.0-1.0 score.
    """
    score = 0.5  # Base for having MCQ data
    
    options = question_data.get("mc_options", [])
    correct_idx = question_data.get("correct_option")
    
    # Must have 4 options
    if isinstance(options, list) and len(options) == 4:
        score += 0.3
    else:
        score -= 0.3
    
    # Must have valid correct_option index
    if isinstance(correct_idx, int) and 0 <= correct_idx < len(options or []):
        score += 0.2
    else:
        score -= 0.3
    
    # Check that options are not empty or duplicates
    if options:
        if all(isinstance(opt, str) and len(opt.strip()) > 5 for opt in options):
            score += 0.1
        
        # Penalty for duplicate options
        if len(set(opt.lower() for opt in options)) < len(options):
            score -= 0.5
    
    return max(min(score, 1.0), 0.0)


def _check_for_banned_terms(question_text: str, answer_text: str) -> float:
    """
    Check for banned terms like 'scene' and penalize their usage.
    Returns 0.0-1.0 score (1.0 = no banned terms).
    """
    full_text = f"{question_text} {answer_text}".lower()
    
    score = 1.0
    for pattern in BANNED_TERMS:
        matches = len(re.findall(pattern, full_text, re.IGNORECASE))
        if matches > 0:
            score -= 0.15 * matches  # Penalty per occurrence
    
    return max(score, 0.0)


def _check_relevance_to_context(question_text: str, graph_context: str, transcript: str) -> float:
    """
    Simple relevance check: do key words from question appear in context?
    Returns 0.0-1.0 score.
    """
    score = 0.5
    
    context = f"{graph_context} {transcript}".lower()
    question_words = set(word.lower() for word in question_text.split() if len(word) > 4)
    
    if not question_words:
        return score
    
    # Count how many question words appear in context
    matches = sum(1 for word in question_words if word in context)
    relevance_ratio = matches / len(question_words) if question_words else 0
    
    score = 0.5 + (relevance_ratio * 0.5)
    return min(score, 1.0)


def calculate_quality_score(question_data: dict, 
                          graph_context: str = "", 
                          transcript: str = "") -> float:
    """
    Calculate overall quality score for a question (0.0-1.0).
    Combines multiple quality metrics.
    """
    question_text = question_data.get("question_text", "")
    answer_text = question_data.get("answer_text", "")
    
    scores = {
        "grammar_clarity": _check_grammar_and_clarity(question_text),
        "vague_patterns": _check_for_vague_patterns(question_text),
        "audio_visual": _check_audio_visual_alignment(question_text, answer_text),
        "mcq_structure": _check_mcq_structure(question_data),
        "banned_terms": _check_for_banned_terms(question_text, answer_text),
        "relevance": _check_relevance_to_context(question_text, graph_context, transcript),
    }
    
    # Weighted average (MCQ structure and audio-visual are most important)
    weights = {
        "grammar_clarity": 1.0,
        "vague_patterns": 1.5,
        "audio_visual": 2.0,
        "mcq_structure": 2.0,
        "banned_terms": 1.0,
        "relevance": 1.5,
    }
    
    total_weight = sum(weights.values())
    weighted_score = sum(scores[k] * weights[k] for k in scores) / total_weight
    
    return round(weighted_score, 3)


def filter_questions(questions: list[dict], 
                    graph_context: str = "", 
                    transcript: str = "",
                    threshold: float | None = None) -> list[dict]:
    """
    Filter out low-quality questions based on multiple criteria.
    
    Returns:
        - questions that pass quality threshold
        - questions with quality_score field populated
    """
    thresh = threshold or QUALITY_THRESHOLD
    
    scored_questions = []
    for q in questions:
        quality = calculate_quality_score(q, graph_context, transcript)
        q["quality_score"] = quality
        scored_questions.append(q)
    
    filtered = [q for q in scored_questions if q["quality_score"] >= thresh]
    removed = len(scored_questions) - len(filtered)
    
    if removed > 0:
        logger.info(f"Quality filter: Removed {removed}/{len(scored_questions)} questions "
                   f"(threshold={thresh}). {len(filtered)} high-quality questions remain.")
    
    return filtered
