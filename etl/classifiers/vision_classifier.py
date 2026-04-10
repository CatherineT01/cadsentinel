# vision_classifier.py

"""
cadsentinel.etl.classifiers.vision_classifier
------------------------------------------------
Classifies a drawing type using AI vision on a rendered PNG thumbnail.
Falls back to text-based LLM classification if rendering fails.
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import tempfile

from .drawing_type_classifier import DrawingTypeResult

log = logging.getLogger(__name__)

AUTOCAD_READER_PATH = os.environ.get(
    "AUTOCAD_READER_PATH",
    "D:/AutoCAD_Reader"
)

_VISION_PROMPT = """You are an engineering drawing classifier for JIT Cylinders, a hydraulic cylinder manufacturer.

Look at this engineering drawing image and classify it into exactly one of these types:

- assembly: Full cylinder assembly drawing. Look for: model code (H-MT4-...), bore/stroke/rod notes block, port specifications, multiple views of a complete cylinder
- rod: Piston rod component. Look for: long cylindrical shaft, thread details on ends, rod diameter callouts
- gland: Gland component. Look for: ring/sleeve shape, internal bore with seal grooves, rod hole through center
- barrel: Cylinder barrel/tube. Look for: hollow tube/cylinder shape, bore diameter table, wall thickness callouts
- piston: Piston component. Look for: disc/spool shape, seal ring grooves, bore fit dimensions
- rod_end_head: Rod end head (front cap). Look for: plate with central rod hole, mounting bolt pattern, port holes
- cap_end_head: Cap end head (rear cap). Look for: plate with NO central hole, mounting bolt pattern, port holes
- tie_rod: Tie rod. Look for: long threaded rod, smaller diameter than piston rod, used in sets of 4
- generic_part: Any other component not matching above types
- unknown: Cannot determine from image

Respond with ONLY a JSON object:
{
  "type_code": "one of the types above",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief description of what you see in the drawing"
}"""


class VisionClassifier:
    """
    Classifies drawing type using AI vision on a rendered PNG thumbnail.
    """

    def __init__(self, provider: str = "openai"):
        self.provider = provider.lower()

    def classify_from_file(
        self,
        dwg_path: str,
        filename: str,
    ) -> DrawingTypeResult | None:
        """Render a DWG to PNG and classify using vision model."""
        png_path = self._render_to_png(dwg_path)
        if not png_path:
            log.warning(f"Could not render {filename} to PNG — skipping vision")
            return None
        try:
            return self._classify_image(png_path, filename)
        finally:
            try:
                os.remove(png_path)
            except Exception:
                pass

    def _render_to_png(self, dwg_path: str) -> str | None:
            """Render DWG to PNG using AutoCAD_Reader DWGProcessor."""
            try:
                if AUTOCAD_READER_PATH not in sys.path:
                    sys.path.insert(0, AUTOCAD_READER_PATH)

                import src.DWG_Processor as dp_module
                oda_path = os.path.join(AUTOCAD_READER_PATH, "ODAFileConverter.exe")
                if os.path.exists(oda_path):
                    dp_module.ODA_CONVERTER_PATH       = oda_path
                    dp_module.DWG_CONVERSION_AVAILABLE = True

                from src.DWG_Processor import DWGProcessor
                import shutil
              
                processor = DWGProcessor()
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.close()

                result = processor.dwg_to_png(
                dwg_path        = dwg_path,
                output_png_path = tmp.name,
                dpi             = 150,
                silent          = True,
            )

                # Clean up DXF temp directory created by ODA converter
                if processor.temp_dir and os.path.exists(processor.temp_dir):
                    shutil.rmtree(processor.temp_dir, ignore_errors=True)

                if result and os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
                    return tmp.name

                # Clean up empty PNG if rendering failed
                if os.path.exists(tmp.name):
                    os.remove(tmp.name)
                return None

            except Exception as e:
                log.warning(f"PNG rendering failed: {e}")
                return None


    def _classify_image(
        self,
        png_path: str,
        filename: str,
    ) -> DrawingTypeResult | None:
        """Send PNG to vision model and parse result."""
        import json

        with open(png_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        try:
            if self.provider == "openai":
                response_text = self._call_openai_vision(image_data, filename)
            elif self.provider in ("grok", "xai"):
                response_text = self._call_grok_vision(image_data, filename)
            else:
                return None

            clean = response_text.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()

            data = json.loads(clean)
            return DrawingTypeResult(
                type_code  = data.get("type_code", "unknown"),
                confidence = float(data.get("confidence", 0.5)),
                source     = "vision",
                reasoning  = data.get("reasoning", ""),
            )

        except Exception as e:
            log.warning(f"Vision classification failed for {filename}: {e}")
            return None

    def _call_openai_vision(self, image_data: str, filename: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model    = "gpt-4o-mini",
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":    f"data:image/png;base64,{image_data}",
                            "detail": "high",
                        }
                    },
                    {
                        "type": "text",
                        "text": _VISION_PROMPT + f"\n\nFilename: {filename}",
                    }
                ]
            }],
            max_tokens  = 200,
            temperature = 0.1,
        )
        return response.choices[0].message.content.strip()

    def _call_grok_vision(self, image_data: str, filename: str) -> str:
        from openai import OpenAI
        client = OpenAI(
            api_key  = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY"),
            base_url = "https://api.x.ai/v1",
        )
        response = client.chat.completions.create(
            model    = "grok-2-vision-latest",
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}",
                        }
                    },
                    {
                        "type": "text",
                        "text": _VISION_PROMPT + f"\n\nFilename: {filename}",
                    }
                ]
            }],
            max_tokens  = 200,
            temperature = 0.1,
        )
        return response.choices[0].message.content.strip()