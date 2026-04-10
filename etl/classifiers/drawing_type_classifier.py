# drawing_type_classifier.py

"""
cadsentinel.etl.classifiers.drawing_type_classifier
------------------------------------------------------
Classifies a drawing into one of the known JIT drawing types using
a three-stage approach:

  Stage 1: Title block fields (drawing title, description, part number)
  Stage 2: Filename pattern matching
  Stage 3: LLM text classification (using drawing text chunks)

Returns a DrawingTypeResult with type_code, confidence, and source.

Drawing types:
    assembly     -- Full cylinder assembly drawing
    rod          -- Piston rod component
    gland        -- Gland component
    barrel       -- Barrel component
    piston       -- Piston component
    rod_end_head -- Rod end head component
    cap_end_head -- Cap end head component
    tie_rod      -- Tie rod component
    generic_part -- Generic part drawing
    unknown      -- Could not determine type
"""

from __future__ import annotations
import os 
import json
import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

# ── Keyword patterns for each drawing type ───────────────────────── #
_KEYWORD_PATTERNS: dict[str, list[str]] = {
    "assembly":     ["assembly", "assy", "h-mt", "h-me", "h-mf", "h-ms", "h-mp",
                     "h-mx", "h-mh", "h-mc", "bore", "stroke", "model code"],
    "rod":          ["rod", "piston rod", "hrod", "h rod"],
    "gland":        ["gland", "hgland", "h gland"],
    "barrel":       ["barrel", "hbarrel", "h barrel", "tube", "cylinder tube"],
    "piston":       ["piston", "hpiston", "h piston"],
    "rod_end_head": ["rod end", "rod end head", "reh", "front head"],
    "cap_end_head": ["cap end", "cap end head", "ceh", "rear head", "back head"],
    "tie_rod":      ["tie rod", "tierod", "h tie"],
    "generic_part": ["part", "component", "detail"],
}

# ── LLM classification prompt ─────────────────────────────────────── #
_CLASSIFICATION_PROMPT = """You are an engineering drawing classifier for JIT Cylinders, a hydraulic cylinder manufacturer.

Classify the following drawing text into exactly one of these drawing types:
- assembly: Full cylinder assembly drawing with model code, bore, stroke, rod, and port specifications
- rod: Piston rod component drawing
- gland: Gland component drawing (seals the rod end of the cylinder)
- barrel: Cylinder barrel/tube component drawing
- piston: Piston component drawing
- rod_end_head: Rod end head component drawing (front cap with rod hole)
- cap_end_head: Cap end head component drawing (rear cap, no rod hole)
- tie_rod: Tie rod component drawing
- generic_part: Generic part or component that does not match any specific cylinder component type
- unknown: Cannot determine drawing type from available text

Drawing text:
{drawing_text}

Filename: {filename}

Respond with ONLY a JSON object, no other text:
{{
  "type_code": "one of the types above",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}}"""


@dataclass
class DrawingTypeResult:
    type_code:   str
    confidence:  float
    source:      str   # 'title_block', 'filename', 'llm', 'default'
    reasoning:   str   = ""


class DrawingTypeClassifier:
    """
    Classifies a drawing into a JIT drawing type using
    title block, filename, and LLM text classification.
    """

    def __init__(self, provider: str = "openai"):
        self.provider = provider.lower()

    def classify(
        self,
        drawing_id: int,
        filename:   str,
        evidence:   dict,
        dwg_path:   str | None = None,
    ) -> DrawingTypeResult:
        """
        Classify a drawing into a drawing type.
        Stage 0: Vision, Stage 1: Title block, Stage 2: Filename, Stage 3: LLM text
        """
        # Stage 0 - vision classification (most accurate)
        if dwg_path and os.path.exists(dwg_path):
            try:
                from .vision_classifier import VisionClassifier
                vision = VisionClassifier(provider=self.provider)
                vresult = vision.classify_from_file(dwg_path, filename)
                if vresult and vresult.confidence >= 0.75:
                    log.info(f"Drawing {drawing_id} classified as '{vresult.type_code}' from vision")
                    return vresult
            except Exception as e:
                log.warning(f"Vision classification failed: {e}")

        # Stage 1 — title block
        result = self._classify_from_title_block(evidence)
        if result and result.confidence >= 0.80:
            log.info(f"Drawing {drawing_id} classified as '{result.type_code}' from title block")
            return result

        # Stage 2 — filename
        filename_result = self._classify_from_filename(filename)
        if filename_result and filename_result.confidence >= 0.70:
            log.info(f"Drawing {drawing_id} classified as '{filename_result.type_code}' from filename")
            return filename_result

        # Stage 3 — LLM text classification
        llm_result = self._classify_from_llm(filename, evidence)
        if llm_result:
            log.info(f"Drawing {drawing_id} classified as '{llm_result.type_code}' from LLM (confidence={llm_result.confidence})")
            return llm_result

        # If title block or filename gave a lower-confidence result, use it
        if result:
            return result
        if filename_result:
            return filename_result

        return DrawingTypeResult(
            type_code  = "unknown",
            confidence = 0.0,
            source     = "default",
            reasoning  = "Could not determine drawing type from any available source.",
        )

    def _classify_from_title_block(self, evidence: dict) -> DrawingTypeResult | None:
        """Check title block fields for drawing type indicators."""
        for item in evidence.get("evidence", []):
            if item.get("source") != "drawing_title_block":
                continue
            attrs = item.get("attributes") or {}
            # Combine all attribute values into one searchable string
            combined = " ".join(str(v) for v in attrs.values() if v).lower()
            if not combined:
                continue
            result = self._keyword_match(combined, source="title_block")
            if result:
                return result
        return None

    def _classify_from_filename(self, filename: str) -> DrawingTypeResult | None:
        """Check filename for drawing type indicators."""
        name = filename.lower().replace("_", " ").replace("-", " ")
        return self._keyword_match(name, source="filename")

    def _keyword_match(self, text: str, source: str) -> DrawingTypeResult | None:
        """Match text against keyword patterns for each drawing type."""
        scores: dict[str, int] = {}
        for type_code, keywords in _KEYWORD_PATTERNS.items():
            for kw in keywords:
                if kw.lower() in text:
                    scores[type_code] = scores.get(type_code, 0) + 1

        if not scores:
            return None

        best_type  = max(scores, key=lambda t: scores[t])
        best_score = scores[best_type]
        total      = sum(scores.values())
        confidence = min(0.90, 0.50 + (best_score / max(total, 1)) * 0.40)

        return DrawingTypeResult(
            type_code  = best_type,
            confidence = round(confidence, 4),
            source     = source,
            reasoning  = f"Matched {best_score} keyword(s): {[k for k in _KEYWORD_PATTERNS[best_type] if k in text]}",
        )

    def _classify_from_llm(
        self,
        filename: str,
        evidence: dict,
    ) -> DrawingTypeResult | None:
        """Use LLM to classify drawing type from text content."""
        # Collect drawing text
        texts = []
        for item in evidence.get("evidence", []):
            source = item.get("source", "")
            if source in ("drawing_text_chunks", "drawing_entities"):
                text = item.get("text") or item.get("chunk_text") or ""
                if text.strip():
                    texts.append(text.strip()[:200])
            elif source == "drawing_title_block":
                attrs = item.get("attributes") or {}
                for v in attrs.values():
                    if v:
                        texts.append(str(v).strip())

        if not texts:
            return None

        drawing_text = "\n".join(texts[:30])  # limit context
        prompt = _CLASSIFICATION_PROMPT.format(
            drawing_text = drawing_text,
            filename     = filename,
        )

        try:
            response = self._call_llm(prompt)
            data     = json.loads(response)
            return DrawingTypeResult(
                type_code  = data.get("type_code", "unknown"),
                confidence = float(data.get("confidence", 0.5)),
                source     = "llm",
                reasoning  = data.get("reasoning", ""),
            )
        except Exception as e:
            log.warning(f"LLM classification failed: {e}")
            return None

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM provider."""
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider in ("grok", "xai"):
            return self._call_grok(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai(self, prompt: str) -> str:
        import os
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model       = "gpt-4o-mini",
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.1,
            max_tokens  = 200,
        )
        return response.choices[0].message.content.strip()

    def _call_grok(self, prompt: str) -> str:
        import os
        from openai import OpenAI
        client = OpenAI(
            api_key  = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY"),
            base_url = "https://api.x.ai/v1",
        )
        response = client.chat.completions.create(
            model       = "grok-3-mini",
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.1,
            max_tokens  = 200,
        )
        return response.choices[0].message.content.strip()