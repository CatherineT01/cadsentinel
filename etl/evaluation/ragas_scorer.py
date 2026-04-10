# ragas_scorer.py

"""
cadsentinel.etl.evaluation.ragas_scorer
-----------------------------------------
Computes per-spec RAGAS-style evaluation metrics for one spec execution run.

Metrics:
    retrieval_relevance    -- did retriever fetch relevant evidence?
    evidence_coverage      -- did evidence cover expected types?
    decision_correctness   -- was pass/fail verdict correct? (needs gold label)
    faithfulness           -- was explanation grounded in evidence?
    false_positive_risk    -- did system invent unsupported issues?
    confidence_calibration -- did confidence match correctness? (needs gold label)
    stability              -- consistent results across reruns? (needs multiple runs)

Composite score:
    0.25 * retrieval_relevance
  + 0.20 * evidence_coverage
  + 0.25 * decision_correctness   (skipped if no gold label)
  + 0.20 * faithfulness
  + 0.10 * (1 - false_positive_risk)

Similarity strategy:
    - Uses embedding cosine similarity when OPENAI_API_KEY is set
    - Falls back to keyword overlap (Jaccard) when embeddings unavailable
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

EMBEDDING_MODEL = os.environ.get("CADSENTINEL_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
GROK_API_KEY    = os.environ.get("GROK_API_KEY", "")

# Composite score weights (from blueprint section 10.3)
WEIGHTS = {
    "retrieval_relevance": 0.25,
    "evidence_coverage":   0.20,
    "decision_correctness": 0.25,
    "faithfulness":        0.20,
    "false_positive_risk": 0.10,   # applied as (1 - score)
}


# ── Result dataclass ─────────────────────────────────────────── #

@dataclass
class SpecEvalResult:
    """
    All evaluation scores for one spec execution run.
    None means the metric could not be computed (e.g. needs gold label).
    """
    spec_execution_run_id:  int
    retrieval_relevance:    Optional[float] = None
    evidence_coverage:      Optional[float] = None
    decision_correctness:   Optional[float] = None   # needs gold label
    faithfulness:           Optional[float] = None
    false_positive_risk:    Optional[float] = None
    confidence_calibration: Optional[float] = None   # needs gold label
    stability:              Optional[float] = None   # needs multiple runs
    composite_score:        Optional[float] = None
    recommendation:         str             = ""
    evaluator_name:         str             = "cadsentinel_ragas_v1"
    notes:                  list[str]       = field(default_factory=list)

    def to_metric_rows(self) -> list[dict]:
        """Convert to list of rows for spec_eval_scores table."""
        metrics = {
            "retrieval_relevance":    self.retrieval_relevance,
            "evidence_coverage":      self.evidence_coverage,
            "decision_correctness":   self.decision_correctness,
            "faithfulness":           self.faithfulness,
            "false_positive_risk":    self.false_positive_risk,
            "confidence_calibration": self.confidence_calibration,
            "stability":              self.stability,
        }
        rows = []
        for metric_name, value in metrics.items():
            if value is not None:
                rows.append({
                    "spec_execution_run_id": self.spec_execution_run_id,
                    "metric_name":           metric_name,
                    "metric_value":          value,
                    "evaluator_name":        self.evaluator_name,
                })
        if self.composite_score is not None:
            rows.append({
                "spec_execution_run_id": self.spec_execution_run_id,
                "metric_name":           "composite_score",
                "metric_value":          self.composite_score,
                "evaluator_name":        self.evaluator_name,
            })
        return rows


# ── Main scorer ───────────────────────────────────────────────── #

class RAGASScorer:
    """
    Computes RAGAS-style metrics for one spec execution run.

    Args:
        use_embeddings: If True, attempt embedding similarity for
                        retrieval_relevance. Falls back to keyword overlap.
    """

    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings
        self._embed_cache: dict[str, list[float]] = {}

    def score(
        self,
        spec_execution_run_id: int,
        rule_text:             str,
        rule_type:             str,
        expected_evidence_types: list[str],
        evidence_package:      dict,
        execution_result:      dict,
        gold_pass_fail:        Optional[str] = None,
    ) -> SpecEvalResult:
        """
        Compute all metrics for one spec execution run.

        Args:
            spec_execution_run_id: ID of the spec_execution_runs row
            rule_text:             normalized_rule_text from spec_rules
            rule_type:             e.g. "dimension_units"
            expected_evidence_types: from spec_rules.expected_evidence_types
            evidence_package:      output from EvidenceRetriever.retrieve()
            execution_result:      output from validator or llm_checker
            gold_pass_fail:        human-labeled ground truth (or None)

        Returns:
            SpecEvalResult with all computed metrics
        """
        result = SpecEvalResult(spec_execution_run_id=spec_execution_run_id)
        evidence_items = evidence_package.get("evidence", [])
        pass_fail      = execution_result.get("pass_fail", "needs_review")
        confidence     = execution_result.get("confidence", 0.0)
        issues         = execution_result.get("issues", [])
        issue_summary  = execution_result.get("issue_summary", "")
        evidence_used  = execution_result.get("evidence_used", [])

        # ── 1. Retrieval relevance ───────────────────────────── #
        result.retrieval_relevance = self._score_retrieval_relevance(
            rule_text, evidence_items
        )

        # ── 2. Evidence coverage ─────────────────────────────── #
        result.evidence_coverage = self._score_evidence_coverage(
            expected_evidence_types, evidence_items
        )

        # ── 3. Decision correctness (needs gold label) ────────── #
        if gold_pass_fail:
            result.decision_correctness = self._score_decision_correctness(
                pass_fail, gold_pass_fail
            )
            result.notes.append(f"Gold label: {gold_pass_fail}")

        # ── 4. Faithfulness ──────────────────────────────────── #
        result.faithfulness = self._score_faithfulness(
            issue_summary, issues, evidence_items, rule_text
        )

        # ── 5. False positive risk ────────────────────────────── #
        result.false_positive_risk = self._score_false_positive_risk(
            pass_fail, issues, evidence_items
        )

        # ── 6. Confidence calibration (needs gold label) ──────── #
        if gold_pass_fail and result.decision_correctness is not None:
            result.confidence_calibration = self._score_confidence_calibration(
                confidence, result.decision_correctness
            )

        # ── 7. Stability — deferred (needs multiple runs) ─────── #
        result.stability = None
        result.notes.append("stability: deferred — requires multiple run comparison")

        # ── Composite score ───────────────────────────────────── #
        result.composite_score = self._compute_composite(result)

        # ── Recommendation ────────────────────────────────────── #
        result.recommendation = self._make_recommendation(result)

        return result

    # ── Metric implementations ───────────────────────────────── #

    def _score_retrieval_relevance(
        self,
        rule_text:      str,
        evidence_items: list,
    ) -> float:
        """
        How relevant is the retrieved evidence to the rule being checked?
        Uses embedding cosine similarity if available, else keyword Jaccard.
        Accepts either a list of evidence dicts or a list of raw text strings.
        """
        if not evidence_items:
            return 0.0

        # Accept either raw strings or evidence dicts
        if isinstance(evidence_items[0], str):
            evidence_texts = evidence_items
        else:
            evidence_texts = _extract_text_from_evidence(evidence_items)
        if not evidence_texts:
            # Non-text evidence (layers, dimensions without text) gets a baseline
            return 0.60

        if self.use_embeddings and OPENAI_API_KEY:
            return self._embedding_relevance(rule_text, evidence_texts)
        else:
            return self._keyword_relevance(rule_text, evidence_texts)

    def _score_evidence_coverage(
        self,
        expected_types: list[str],
        evidence_items: list[dict],
    ) -> float:
        """
        What fraction of expected evidence types were actually retrieved?
        """
        if not expected_types:
            return 1.0  # no expectations = full coverage

        # Map evidence sources to evidence type labels
        SOURCE_TO_TYPE = {
            "drawing_title_block":  "title_block",
            "drawing_dimensions":   "dimension",
            "drawing_layers":       "layer",
            "drawing_entities":     "entity",
            "drawing_text_chunks":  "note",
        }

        retrieved_types = set()
        for item in evidence_items:
            source = item.get("source", "")
            etype  = SOURCE_TO_TYPE.get(source, source)
            retrieved_types.add(etype)
            # text/note overlap
            if etype == "note":
                retrieved_types.add("text")
            if etype == "entity":
                retrieved_types.add("block")
                retrieved_types.add("insert")

        matched = sum(
            1 for t in expected_types
            if t.lower() in retrieved_types
        )
        return matched / len(expected_types)

    def _score_decision_correctness(
        self,
        predicted:  str,
        gold_label: str,
    ) -> float:
        """
        Binary: did the system produce the correct verdict?
        1.0 = correct, 0.0 = incorrect.
        Partial credit for warning/needs_review vs fail.
        """
        pred  = predicted.lower()
        gold  = gold_label.lower()

        if pred == gold:
            return 1.0

        # Partial credit: warning is closer to fail than pass
        if gold in ("fail", "warning") and pred in ("fail", "warning"):
            return 0.5

        # needs_review is uncertain — penalise less than a wrong verdict
        if pred == "needs_review":
            return 0.25

        return 0.0

    def _score_faithfulness(
        self,
        issue_summary:  str,
        issues:         list[dict],
        evidence_items: list[dict],
        rule_text:      str,
    ) -> float:
        """
        Are the claims in the result grounded in the retrieved evidence?

        For deterministic results: always faithful (1.0) since no LLM invention.
        For LLM results: check whether issue descriptions reference evidence content.
        """
        if not issues:
            # Pass result — no claims to verify
            return 1.0

        evidence_texts = _extract_text_from_evidence(evidence_items)
        if not evidence_texts:
            # Can't verify faithfulness without text evidence
            return 0.5

        all_evidence_text = " ".join(evidence_texts).lower()

        # Check what fraction of issue descriptions have some grounding
        grounded = 0
        for issue in issues:
            desc      = (issue.get("description") or "").lower()
            evidence  = issue.get("evidence") or []

            # Issue cites evidence explicitly
            if evidence:
                grounded += 1
                continue

            # Issue description overlaps with retrieved evidence
            desc_tokens    = set(_tokenize(desc))
            evidence_tokens = set(_tokenize(all_evidence_text))
            overlap = len(desc_tokens & evidence_tokens)
            if overlap >= 3:
                grounded += 0.8
            elif overlap >= 1:
                grounded += 0.4

        return min(1.0, grounded / len(issues))

    def _score_false_positive_risk(
        self,
        pass_fail:      str,
        issues:         list[dict],
        evidence_items: list[dict],
    ) -> float:
        """
        Risk that the system invented issues not supported by evidence.
        Lower is better (0.0 = no false positive risk).
        """
        if pass_fail == "pass" or not issues:
            return 0.0

        if not evidence_items:
            # Issues raised with no evidence at all — high risk
            return 0.9

        evidence_texts = _extract_text_from_evidence(evidence_items)
        if not evidence_texts:
            return 0.5

        all_evidence_text = " ".join(evidence_texts).lower()

        # For each issue, estimate risk that it's unsupported
        risk_scores = []
        for issue in issues:
            desc      = (issue.get("description") or "").lower()
            evidence  = issue.get("evidence") or []

            if evidence:
                # Explicitly cited — low risk
                risk_scores.append(0.05)
                continue

            desc_tokens     = set(_tokenize(desc))
            evidence_tokens = set(_tokenize(all_evidence_text))
            overlap         = len(desc_tokens & evidence_tokens)

            if overlap >= 5:
                risk_scores.append(0.10)
            elif overlap >= 2:
                risk_scores.append(0.30)
            else:
                risk_scores.append(0.70)

        return sum(risk_scores) / len(risk_scores) if risk_scores else 0.0

    def _score_confidence_calibration(
        self,
        confidence:           float,
        decision_correctness: float,
    ) -> float:
        """
        How well-calibrated was the confidence score?
        Perfect calibration: high confidence when correct, low when wrong.
        Score = 1 - |confidence - correctness|
        """
        return max(0.0, 1.0 - abs(confidence - decision_correctness))

    # ── Composite score ───────────────────────────────────────── #

    def _compute_composite(self, result: SpecEvalResult) -> Optional[float]:
        """
        Weighted composite from blueprint section 10.3.
        Skips metrics that are None (e.g. no gold label).
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for metric, weight in WEIGHTS.items():
            value = getattr(result, metric, None)
            if value is None:
                continue
            if metric == "false_positive_risk":
                weighted_sum += weight * (1.0 - value)
            else:
                weighted_sum += weight * value
            total_weight += weight

        if total_weight == 0:
            return None

        # Normalise to available metrics
        return round(weighted_sum / total_weight, 4)

    def _make_recommendation(self, result: SpecEvalResult) -> str:
        """Generate a human-readable recommendation based on scores."""
        score = result.composite_score
        if score is None:
            return "insufficient_data"

        rr  = result.retrieval_relevance or 0
        fpr = result.false_positive_risk or 0
        fth = result.faithfulness or 0

        if score >= 0.85:
            return "retain_current_prompt"
        elif rr < 0.50:
            return "refine_retrieval_recipe"
        elif fpr > 0.60:
            return "review_for_hallucination"
        elif fth < 0.50:
            return "improve_evidence_citation"
        elif score >= 0.65:
            return "minor_prompt_tuning"
        else:
            return "major_revision_needed"

    # ── Embedding similarity ──────────────────────────────────── #

    def _embedding_relevance(
        self,
        rule_text:       str,
        evidence_texts:  list[str],
    ) -> float:
        """Cosine similarity between rule embedding and mean evidence embedding."""
        try:
            rule_vec  = self._embed(rule_text)
            ev_vecs   = [self._embed(t[:500]) for t in evidence_texts[:5]]
            mean_vec  = _mean_vector(ev_vecs)
            sim       = _cosine_similarity(rule_vec, mean_vec)
            return max(0.0, min(1.0, sim))
        except Exception:
            log.debug("Embedding similarity failed — falling back to keyword overlap")
            return self._keyword_relevance(rule_text, evidence_texts)

    def _embed(self, text: str) -> list[float]:
        if text in self._embed_cache:
            return self._embed_cache[text]
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        vec = response.data[0].embedding
        self._embed_cache[text] = vec
        return vec

    # ── Keyword overlap (fallback) ────────────────────────────── #

    def _keyword_relevance(
        self,
        rule_text:      str,
        evidence_texts: list[str],
    ) -> float:
        """
        Jaccard similarity between rule tokens and evidence tokens.
        Weighted by evidence item count to reward broader retrieval.
        """
        rule_tokens = set(_tokenize(rule_text))
        if not rule_tokens:
            return 0.0

        scores = []
        for text in evidence_texts:
            ev_tokens = set(_tokenize(text))
            if not ev_tokens:
                continue
            intersection = rule_tokens & ev_tokens
            union        = rule_tokens | ev_tokens
            scores.append(len(intersection) / len(union))

        if not scores:
            return 0.0

        # Mean across evidence items, boosted slightly if many items retrieved
        base  = sum(scores) / len(scores)
        boost = min(0.10, len(scores) * 0.02)
        return min(1.0, base + boost)


# ── Utility functions ─────────────────────────────────────────── #

def _extract_text_from_evidence(evidence_items: list[dict]) -> list[str]:
    """Pull readable text strings out of an evidence package."""
    texts = []
    for item in evidence_items:
        source = item.get("source", "")

        if source == "drawing_title_block":
            attrs = item.get("attributes") or {}
            for val in attrs.values():
                if val and str(val).strip():
                    texts.append(str(val))

        elif source in ("drawing_text_chunks", "drawing_entities"):
            text = item.get("text") or item.get("chunk_text") or ""
            if text.strip():
                texts.append(text)

        elif source == "drawing_dimensions":
            user_text = item.get("user_text") or ""
            if user_text.strip():
                texts.append(user_text)
            dim_type = item.get("dim_type") or ""
            if dim_type:
                texts.append(dim_type)

        elif source == "drawing_layers":
            name = item.get("layer_name") or ""
            if name:
                texts.append(name)

    return texts


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer — lowercase, alphanumeric only."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Element-wise mean of a list of vectors."""
    if not vectors:
        return []
    n   = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]