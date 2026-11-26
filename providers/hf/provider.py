"""Zero-shot HuggingFace recognizers, chosen by `HF_MODEL`.

  microsoft/trocr-large-printed    native OCR, 558 M, seq2seq over the line
  Qwen2.5-VL-7B-Instruct           generalist VLM

The Qwen prompt below asks for a diplomatic transcription and is part of the method.
"""
import os
import time

import torch
from PIL import Image

from providers.base import Provider
from providers.schema import Line, Result

MODEL = os.environ.get("HF_MODEL", "microsoft/trocr-large-printed")
BATCH = int(os.environ.get("HF_BATCH", "16"))
MAX_TOK = int(os.environ.get("HF_MAX_TOKENS", "256"))

DEFAULT_PROMPT = (
    "Transcribe this line of 18th-century printed text exactly as it appears. "
    "Preserve historical orthography, including the long s (ſ) wherever the "
    "printed glyph is a long s. Do not modernise spelling. Do not add "
    "commentary. Output only the transcribed line."
)
PROMPT = os.environ.get("HF_PROMPT") or DEFAULT_PROMPT


class HFProvider(Provider):
    name = "hf"

    def load(self):
        self.qwen = 'qwen' in MODEL.lower()
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.qwen:
            # 7 B in bf16 is ~16 GB; it fits in the 32 of the 5090.
            from transformers import (AutoProcessor,
                                      Qwen2_5_VLForConditionalGeneration)
            self.proc = AutoProcessor.from_pretrained(MODEL)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL, torch_dtype=torch.bfloat16, device_map=self.dev)
        else:
            # float32: TrOCR-large fits and avoids encoder/decoder dtype mismatches
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.proc = TrOCRProcessor.from_pretrained(MODEL)
            self.model = VisionEncoderDecoderModel.from_pretrained(
                MODEL).to(self.dev)
        self.model.eval()

    def predict(self, request):
        """`{line_paths: [...]}` -> one `Line` per crop, in the same order."""
        paths = request["line_paths"]
        started = time.perf_counter()
        texts = []
        step = 1 if self.qwen else BATCH
        for i in range(0, len(paths), step):
            block = paths[i:i + step]
            imgs = [Image.open(p).convert("RGB") for p in block]
            texts.extend(self._qwen(imgs) if self.qwen else self._trocr(imgs))
        lines = [Line(text=t, image_path=p) for t, p in zip(texts, paths)]
        return Result(provider=MODEL, lines=lines,
                      elapsed=time.perf_counter() - started,
                      warnings={"prompt": PROMPT} if self.qwen else {})

    @torch.inference_mode()
    def _trocr(self, imgs):
        px = self.proc(images=imgs, return_tensors="pt").pixel_values
        px = px.to(self.dev, dtype=next(self.model.parameters()).dtype)
        ids = self.model.generate(px, max_new_tokens=MAX_TOK)
        return [t.strip() for t in
                self.proc.batch_decode(ids, skip_special_tokens=True)]

    @torch.inference_mode()
    def _qwen(self, imgs):
        out = []
        for im in imgs:
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": im},
                {"type": "text", "text": PROMPT}]}]
            text = self.proc.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.proc(text=[text], images=[im], return_tensors="pt")
            inputs = inputs.to(self.dev)
            ids = self.model.generate(**inputs, max_new_tokens=MAX_TOK,
                                      do_sample=False)
            cut = ids[:, inputs.input_ids.shape[1]:]
            out.append(self.proc.batch_decode(
                cut, skip_special_tokens=True)[0].strip())
        return out
