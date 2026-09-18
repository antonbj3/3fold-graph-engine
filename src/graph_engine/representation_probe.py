#!/usr/bin/env python3
"""
representation_probe.py — read the sign a text asserts between two quantities from a language model's HIDDEN STATE
instead of from the token it emits.

What it is. For each (text, x, y) the model is run once on the e7 prompt (lens 0: "Text: …\\n\\nAccording to the text,
when {x} increases, the {y}:\\nA) increases\\nB) decreases"), the hidden state of the LAST PROMPT TOKEN is taken at
every layer, and one standardized logistic regression is fitted on the labels given to `fit`. One layer is used; when
`layer=None` it is chosen on the TRAINING data only, by an inner split over `groups` (in e20's `nested()` the groups
are the quantity pairs, so no layer is picked with the held-out quantities in view). `predict_proba` returns
P(the text asserts that y increases with x).

Why it exists (measured, e20). The same 0.5B/1.5B/3B models read "does this text assert X increases Y" at 0.86/0.89/0.92
from a mid-late layer while their emitted A/B token reads it at 0.52/0.78/0.85 — and the probe does NOT fall on the half
of the sentences that contradict textbook physics, where the token output falls back on the model's prior. The reading
therefore exists in the representation before the collapse to a token.

What it is NOT.
* It is not a physics oracle. It is fitted on the labels you hand it. In e20/e22(a) those are the SYMBOLIC RULE's labels
  (graph_engine.polarity_rules.asserted_sign) on the author's own templated sentences, so the probe inherits that rule's
  domain of forms — direction words from a fixed lexicon, one sentence, both quantities named verbatim — unless it is
  trained on more (e22(b) trains the same class on QuaRTz paragraphs with human labels instead).
* It is not an independent channel from the language model it reads. `as_lens()` tags itself
  "probe:<model>:<layer>" so `lens_pooling` can keep it as its own ORIGIN, but its errors may still correlate with the
  model's token-output lenses: that correlation is a thing to measure (lens_pooling.error_correlation), not to assume away.
* It carries no abstention. The rule returns 0 when it cannot read the sentence; the probe always returns a probability.
  If you want coverage, threshold on |p - 0.5| and report it.
* It is not calibrated beyond what logistic regression gives on the training distribution; on out-of-domain text
  (other people's paragraphs) treat the probability as a score and re-fit calibration (lens_pooling.StratifiedPlatt).

Cost. One forward pass per (text, pair) with `output_hidden_states=True`, batched, fp16 on cuda. Hidden states are
kept for all layers during `fit` (n × layers × d floats) and only the chosen layer is kept afterwards.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = ["RepresentationProbe", "PROMPT_QUESTION"]

PROMPT_QUESTION = "According to the text, when {x} increases, the {y}:"      # e7 QUESTIONS[0]


def _build_prompt(tok, text: str, x: str, y: str) -> str:
    """The e7 prompt at lens 0 (first phrasing, option order A=increases)."""
    msg = (f"Text: {text}\n\n{PROMPT_QUESTION.format(x=x, y=y)}\n"
           f"A) increases\nB) decreases\nAnswer with A or B.")
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)


class RepresentationProbe:
    """Linear probe on last-prompt-token hidden states; see module docstring.

    Parameters
    ----------
    model : str | tuple
        A local directory / Hugging Face id, or an already loaded ``(tokenizer, model)`` pair (then `device` should
        match the model's device). Passing a loaded pair avoids a second copy in GPU memory.
    device : "cuda" | "cpu"
    batch_size : int
    C : float
        Inverse regularization of the logistic regression (e20 used 0.05).
    name : str | None
        Short model name used in the lineage tag; defaults to the directory/id basename.
    """

    def __init__(self, model="Qwen/Qwen2.5-0.5B-Instruct", device: str = "cpu", batch_size: int = 16,
                 C: float = 0.05, name: Optional[str] = None):
        if isinstance(model, tuple):
            self.tok, self.model = model
            self.name = name or "model"
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
            self.model = AutoModelForCausalLM.from_pretrained(
                model, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
            self.name = name or str(model).rstrip("/").split("/")[-1]
        self.device, self.batch_size, self.C = device, batch_size, C
        self.layer: Optional[int] = None
        self.clf = None
        self.n_layers: Optional[int] = None
        self.layer_scores: Optional[List[float]] = None

    # ---------------------------------------------------------------- extraction
    def prompts(self, texts: Sequence[str], pairs: Sequence[Tuple[str, str]]) -> List[str]:
        if len(texts) != len(pairs):
            raise ValueError(f"{len(texts)} texts but {len(pairs)} pairs")
        return [_build_prompt(self.tok, t, x, y) for t, (x, y) in zip(texts, pairs)]

    def hidden_states(self, texts: Sequence[str], pairs: Sequence[Tuple[str, str]],
                      layer: Optional[int] = None) -> np.ndarray:
        """n × layers × d (all layers) or n × d when `layer` is given. Batched, no grad."""
        import torch
        prompts, out = self.prompts(texts, pairs), []
        with torch.no_grad():
            for i in range(0, len(prompts), self.batch_size):
                enc = self.tok(prompts[i:i + self.batch_size], return_tensors="pt", padding=True).to(self.device)
                hs = self.model(**enc, output_hidden_states=True).hidden_states
                if layer is None:
                    out.append(torch.stack([h[:, -1, :].float() for h in hs], 1).cpu().numpy())
                else:
                    out.append(hs[layer][:, -1, :].float().cpu().numpy())
        return np.concatenate(out, 0)

    # ---------------------------------------------------------------- fitting
    def _new_clf(self):
        return make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=2000))

    def _choose_layer(self, H: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> int:
        """Inner split over `groups` on the TRAINING data only (e20's nested() layer choice)."""
        uniq = np.unique(groups)
        if len(uniq) < 2:
            raise ValueError("layer choice needs at least 2 groups; pass layer=... instead")
        scores = np.zeros(H.shape[1])
        for s in range(n_splits):
            rng = np.random.default_rng(seed + s)
            held = rng.choice(uniq, max(1, len(uniq) // 2), replace=False)
            val = np.isin(groups, held)
            fit = ~val
            if len(np.unique(y[fit])) < 2 or not val.any():
                continue
            for l in range(H.shape[1]):
                scores[l] += (self._new_clf().fit(H[fit, l], y[fit]).predict(H[val, l]) == y[val]).mean()
        self.layer_scores = [float(v / n_splits) for v in scores]
        return int(np.argmax(scores))

    def fit(self, texts: Sequence[str], pairs: Sequence[Tuple[str, str]], labels: Sequence,
            layer: Optional[int] = None, groups: Optional[Sequence] = None, n_splits: int = 4,
            seed: int = 0) -> "RepresentationProbe":
        """Fit on `texts` with quantity `pairs` and boolean/±1 `labels` (True / +1 = "the text asserts y increases with x").

        `layer=None` chooses the layer by an inner split over `groups` (default: the (x, y) pair of each item, as in
        e20, so quantity-specific knowledge cannot pick the layer). Nothing outside `texts` is looked at.
        """
        y = np.asarray(labels)
        y = y > 0 if y.dtype != bool else y
        H = self.hidden_states(texts, pairs)
        self.n_layers = H.shape[1]
        if layer is None:
            g = np.asarray([f"{x}|{yy}" for x, yy in pairs] if groups is None else list(groups))
            layer = self._choose_layer(H, y, g, n_splits, seed)
        self.layer = int(layer)
        self.clf = self._new_clf().fit(H[:, self.layer], y)
        return self

    # ---------------------------------------------------------------- use
    def predict_proba(self, texts: Sequence[str], pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        """P(the text asserts that y increases with x), one value per item."""
        if self.clf is None:
            raise RuntimeError("fit() first")
        H = self.hidden_states(texts, pairs, layer=self.layer)
        return self.clf.predict_proba(H)[:, 1]

    def predict_sign(self, texts: Sequence[str], pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        """+1 / -1. The probe never abstains (no 0), unlike polarity_rules.asserted_sign."""
        return np.where(self.predict_proba(texts, pairs) > 0.5, 1, -1)

    @property
    def lineage(self) -> str:
        if self.clf is None:
            raise RuntimeError("fit() first")
        return f"probe:{self.name}:{self.layer}"

    def as_lens(self) -> Callable[[Sequence[str], Sequence[Tuple[str, str]]], np.ndarray]:
        """A callable lens(texts, pairs) -> P(increases), for use as one column next to the token-output lenses in
        lens_pooling. It carries `.lineage` = "probe:<model>:<layer>" so the pooled reading can keep it as a separate
        origin — see the module docstring on why a separate origin is not the same as an independent one."""
        lens = lambda texts, pairs: self.predict_proba(texts, pairs)
        lens.lineage = self.lineage
        lens.origin = "representation_probe"
        lens.model = self.name
        lens.layer = self.layer
        return lens
