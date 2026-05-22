"""
Question generation engine – uses Ollama to produce diverse, reasoning-heavy
questions from structured video analysis data.

Supports 7 categories × 7 personas for maximum diversity.
Questions are generated in MCQ format with explanations and audio-visual alignment.
"""
from __future__ import annotations

import json
import logging
import re

import ollama

from backend.config import settings

logger = logging.getLogger(__name__)

# Descriptive language mapping for the prompt
LANGUAGE_INSTRUCTIONS = {
    "english": "English",
    "hindi": "Hindi (using Devanagari script)",
    "mixed": "Hinglish (a mix of Hindi and English, using Devanagari for Hindi parts and English for technical terms)",
}

# ── Categories & Personas ──────────────────────────────────────────

CATEGORIES = [
    "temporal",
    "causal",
    "counterfactual",
    "contradiction",
    "emotion",
    "multi_scene",
    "symbolic",
    "audio_visual_alignment",
]

CATEGORY_DESCRIPTIONS = {
    "temporal": "Questions about the sequence and timing of the narration's claims. "
                "E.g., 'What prerequisite does the narrator mention before explaining X?', 'How long after this point does result Z occur?'",
    "causal": "Questions about the logical 'Why' behind the narrator's explanations. "
              "E.g., 'Why does the narrator recommend method A over B?', 'What cause is cited for the system failure shown?'",
    "counterfactual": "Questions about hypothetical scenarios related to the technical concepts. "
                      "E.g., 'If the system had no Kubernetes integration, how would self-hosting change?', 'What would the narrator's alternative be if X was unavailable?'",
    "contradiction": "Questions about discrepancies between the narrator's claims and the visual state. "
                     "E.g., 'The narrator claims X is secure, but what visual evidence suggests a vulnerability?', 'Which spoken instruction conflicts with the UI action?'",
    "emotion": "Questions about the speaker's tone, emphasis, and intent for the audience. "
               "E.g., 'Which part of the explanation does the narrator emphasize as most critical?', 'How does the tone change when discussing security risks?'",
    "multi_scene": "Questions requiring reasoning across large chunks of the narrator's script and different parts of the video. "
                   "E.g., 'How does the conclusion about Y reconcile with the early definition of X?', 'Trace the evolution of the concept throughout the video.'",
    "symbolic": "Questions about visual metaphors and structural flow intended by the creator. "
                "E.g., 'What core conceptual model does the recurring diagram represent in the context of the explanation?', 'How does the flow chart illustrate the underlying logic?'",
    "audio_visual_alignment": "Questions focusing on how the visuals conceptually demonstrate or illustrate the narrator's specific spoken technical claims. "
                               "E.g., 'How does the database architecture diagram shown on screen illustrate the scalability logic explained by the narrator?'",
}

# Personas and persona prompts removed to favour uniform, technical questioning


_GENERATION_PROMPT = """You are an expert technical question writer. Focus ONLY on the narrator's spoken content and the technical concepts explained. Use the provided graph_context and the FULL transcript to craft high-quality, information-dense multiple-choice questions that require reasoning and synthesis across subtopics.

GUIDELINES:
- Prioritize explicit information and technical detail present in the transcript.
- Ensure at least 40% of produced questions synthesize two or more distinct subtopics/moments (for example, connect an early definition to a later demonstration and its implication).
- Do NOT use persona framing, rhetorical tones, or stylistic flourishes.
- Do NOT invent facts not supported by the transcript or context.
- Avoid trivial visual-surface questions (icons, colors, layout). Prefer conceptual, causal, and explanatory questions grounded in audio/transcript.
- Aim for uniqueness: avoid near-duplicate or paraphrased items.

CONTEXT:
{graph_context}

FULL TRANSCRIPT:
{transcript}

Produce exactly {num_questions} unique, information-rich multiple-choice questions for category {category_upper}.
The entire output (questions, options, answer_text, and explanation) MUST be in {target_language}.

REQUIREMENTS:
1. Each question object must include these fields: question_text, mc_options (array of 4 strings), correct_option (0-indexed int), answer_text (concise correct answer), explanation (80-180 words) that cites or paraphrases specific transcript fragment(s) and explains why distractors are incorrect.
2. Include a boolean field 'synthesizes' indicating whether the question intentionally synthesizes multiple subtopics.
3. Do NOT include persona or tone fields. Keep language precise and technical.
4. Avoid visual_refs entirely unless a question cannot be grounded without one; in that case include at most one url.
5. Prefer domain-specific vocabulary and exact phrasing from the transcript when possible.
6. MANDATORY: Respond in {target_language}.
7. REMINDER: The user specifically requested all content to be in {target_language}.

Return ONLY a JSON array of question objects and nothing else.
"""


_OPTION_PREFIX_RE = re.compile(r"^\s*(?:[A-Da-d][\).:-]\s*)")
_FRAME_URL_RE = re.compile(r"/frames/[^\s,)\]]+")
_MOMENT_FRAME_RE = re.compile(
    r"Moment\s+(?P<idx>\d+).*?\n\s+Visual reference URL:\s+(?P<url>/frames/[^\s]+)",
    re.DOTALL,
)


def _clean_option(option: object) -> str:
    text = str(option or "").strip()
    text = text.replace("(the correct answer)", "").strip()
    return _OPTION_PREFIX_RE.sub("", text).strip()


def _fallback_options(answer: str) -> list[str]:
    answer = answer.strip() or "The explanation that best connects the narration to the visual evidence"
    return [
        answer[:180],
        "A surface-level detail that is visible but not conceptually important",
        "An unrelated interpretation that ignores the narrator's explanation",
        "A claim that reverses the cause-and-effect relationship in the video",
    ]


def _moment_frame_map(graph_context: str) -> dict[int, str]:
    return {
        int(match.group("idx")): match.group("url")
        for match in _MOMENT_FRAME_RE.finditer(graph_context)
    }


def _remove_scene_wording(text: str) -> str:
    text = re.sub(r"\b[Ss]cene\s+\d+\b", "the relevant moment", text)
    text = re.sub(r"\b[Ss]cenes\b", "moments", text)
    text = re.sub(r"\b[Ss]cene\b", "moment", text)
    return text.strip()


def _normalize_question(q: dict, category: str, graph_context: str) -> dict | None:
    """Normalize a raw question dict coming from the LLM.

    This version intentionally strips persona information and visual references
    to ensure questions are text-first and technically focused.
    """
    question_text = _remove_scene_wording(str(q.get("question_text", "")).strip())
    if not question_text:
        return None

    options = [_clean_option(opt) for opt in q.get("mc_options", [])]
    options = [opt for opt in options if opt]
    used_fallback_options = len(options) != 4
    if used_fallback_options:
        answer_seed = str(q.get("answer_text") or q.get("explanation") or "").strip()
        options = _fallback_options(answer_seed)

    correct_option = q.get("correct_option", 0)
    if isinstance(correct_option, str) and correct_option.strip().upper() in {"A", "B", "C", "D"}:
        correct_option = "ABCD".index(correct_option.strip().upper())
    if used_fallback_options or not isinstance(correct_option, int) or not 0 <= correct_option < 4:
        correct_option = 0

    explanation = _remove_scene_wording(str(q.get("explanation") or q.get("answer_text") or "").strip())
    # Ensure explanation is reasonably long but not excessively verbose
    if len(explanation.split()) < 30:
        explanation = (
            f"{explanation} The correct option best connects specific transcript statements to the technical "
            "implication; distractors either misinterpret the causal chain or ignore key technical details."
        ).strip()

    # Do not attach visual references by default; prefer transcript-grounded questions
    normalized_refs = []

    q_out = {
        "question_text": question_text,
        "mc_options": options,
        "correct_option": correct_option,
        "answer_text": explanation,
        "explanation": explanation,
        "visual_refs": normalized_refs,
        "category": category,
        # persona deliberately omitted
        "reasoning_depth": q.get("reasoning_depth", 3),
        "scenes_involved": q.get("scenes_involved", []),
    }

    return q_out


async def generate_questions_for_category(
    category: str,
    graph_context: str,
    transcript: str,
    num_questions: int | None = None,
    target_language: str = "english",
) -> list[dict]:
    """
    Generate questions for a specific category + persona combination using Ollama.
    Returns list of question dicts with MCQ format and explanations.
    """
    n = num_questions or settings.QUESTIONS_PER_CATEGORY

    lang_instruction = LANGUAGE_INSTRUCTIONS.get(target_language.lower(), target_language)
    logger.info("Generating questions for category: %s in language: %s (Instruction: %s)", category, target_language, lang_instruction)
    
    prompt = _GENERATION_PROMPT.format(
        category_upper=category.upper().replace("_", " "),
        graph_context=graph_context,
        transcript=transcript[:12000],
        num_questions=n,
        target_language=lang_instruction,
    )

    try:
        client = ollama.AsyncClient(host=settings.OLLAMA_URL)
        
        response = await client.chat(
           model=settings.OLLAMA_TEXT_MODEL,
           messages=[{'role': 'user', 'content': prompt}],
           options={'temperature': 0.2}
        )
        
        raw = response['message']['content'].strip()

        # Clean common wrappers (```, ```json, markdown, leading labels)
        if raw.startswith("```"):
            # remove leading fence and optional language tag
            parts = raw.split("\n", 1)
            raw = parts[1] if len(parts) > 1 else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        if raw.lower().startswith("json"):
            raw = raw[4:]

        # Attempt to extract a JSON array from the response even if extra text is present
        parsed = None
        try:
            parsed = json.loads(raw.strip())
        except Exception:
            # fallback: find the first '[' and last ']' and try to parse the slice
            start = raw.find('[')
            end = raw.rfind(']')
            if start != -1 and end != -1 and end > start:
                candidate = raw[start:end+1]
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    # final fallback: try a loose regex match
                    import re
                    m = re.search(r"(\[.*\])", raw, re.DOTALL)
                    if m:
                        try:
                            parsed = json.loads(m.group(1))
                        except Exception:
                            parsed = None

        if not parsed:
            # If still not parsed, log raw response for debugging and raise
            logger.error('Failed to parse JSON from model response:\n%s', raw)
            raise ValueError('Could not parse JSON array from model response')

        questions = parsed

        if not isinstance(questions, list):
            raise ValueError("Response is not a list")

        normalized_questions = []
        for q in questions:
            normalized = _normalize_question(q, category, graph_context)
            if normalized:
                normalized_questions.append(normalized)

        return normalized_questions

    except Exception as e:
        logger.error(f"Question generation failed for {category}: {e}")
        return []


async def generate_all_questions(
    graph_context: str,
    transcript: str,
    categories: list[str] | None = None,
    questions_per_category: int | None = None,
    target_language: str = "english",
) -> list[dict]:
    """
    Generate questions across categories without persona variation.
    Produces more information-dense questions per category and avoids persona-driven phrasing.
    """
    cats = categories or CATEGORIES
    n = questions_per_category or max(3, settings.QUESTIONS_PER_CATEGORY)

    all_questions: list[dict] = []

    for category in cats:
        questions = await generate_questions_for_category(
            category=category,
            graph_context=graph_context,
            transcript=transcript,
            num_questions=n,
            target_language=target_language,
        )
        all_questions.extend(questions)

    return all_questions
