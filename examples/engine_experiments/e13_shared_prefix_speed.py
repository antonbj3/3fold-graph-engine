#!/usr/bin/env python3
"""E13: how fast is the open mechanism behind a "typed decision" service on local hardware?
One state text (~400 tokens), F typed yes/no fields. Three ways to get P(option A) for every field from a small decoder:
  sequential   F separate prompts, one forward pass each                       (what a plain LLM call does)
  batched      F full prompts in one batch                                     (state re-read F times)
  shared       the state is prefilled ONCE, its KV cache is reused for all F field suffixes in one batch
The three must give the same probabilities (checked). Model: $JUDGE_MODEL (default Qwen/Qwen2.5-0.5B-Instruct)."""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"); dev = "cuda" if torch.cuda.is_available() and os.environ.get("E13_CPU") != "1" else "cpu"
dtype = torch.float16 if dev == "cuda" else torch.float32; torch.set_num_threads(6)
tok = AutoTokenizer.from_pretrained(MODEL); model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(dev).eval()
A, B = tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]
STATE = ("Measurement report. " + " ".join(
    f"Run {k}: the {q} was measured at {v} {u} under a load of {10 * k} N; the margin to the limit is {m}." for k, (q, v, u, m) in enumerate(
        [("bearing temperature", 71.5, "C", "positive"), ("shaft deflection", 0.42, "mm", "negative"), ("bolt preload", 18.2, "kN", "positive"),
         ("weld fatigue stress", 96, "MPa", "negative"), ("coolant flow", 3.1, "l/min", "positive"), ("housing resonance", 212, "Hz", "positive")] * 4)))
FIELDS = [f"Does the report state that the margin for {q} is negative?" for q in
          ["bearing temperature", "shaft deflection", "bolt preload", "weld fatigue stress", "coolant flow", "housing resonance", "gear backlash", "seal wear",
           "motor current", "spindle runout", "oil pressure", "frame stiffness", "belt tension", "brake temperature", "axle load", "pump cavitation",
           "valve lift", "rotor balance", "chain stretch", "tyre pressure"]]
head = tok.apply_chat_template([{"role": "user", "content": "Text: " + STATE + "\n\n§"}], tokenize=False, add_generation_prompt=True)
prefix, tail = head.split("§"); suf = [f"{q}\nA) yes\nB) no\nAnswer with A or B." + tail for q in FIELDS]
ids_p = tok(prefix, return_tensors="pt").input_ids.to(dev); n_state = ids_p.shape[1]
sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
pa = lambda lg: torch.softmax(lg[:, [A, B]].float(), -1)[:, 0].cpu().numpy()

@torch.no_grad()
def sequential():
    return np.concatenate([pa(model(**tok(prefix + s, return_tensors="pt").to(dev)).logits[:, -1]) for s in suf])
@torch.no_grad()
def batched():
    tok.padding_side = "left"; enc = tok([prefix + s for s in suf], return_tensors="pt", padding=True).to(dev); return pa(model(**enc).logits[:, -1])
@torch.no_grad()
def shared():
    cache = DynamicCache(); model(input_ids=ids_p, past_key_values=cache, use_cache=True); cache.batch_repeat_interleave(len(suf))
    tok.padding_side = "right"; enc = tok(suf, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    mask = torch.cat([torch.ones(len(suf), n_state, dtype=enc.attention_mask.dtype, device=dev), enc.attention_mask], 1)
    lg = model(input_ids=enc.input_ids, attention_mask=mask, past_key_values=cache, use_cache=True).logits
    last = enc.attention_mask.sum(1) - 1; return pa(lg[torch.arange(len(suf)), last])

res = {"model": MODEL if not os.path.isdir(MODEL) else "local directory (set JUDGE_MODEL)", "device": dev + (" " + torch.cuda.get_device_name(0) if dev == "cuda" else ""), "dtype": str(dtype), "state_tokens": int(n_state), "fields": len(FIELDS)}
ref = None
for name, fn in [("sequential", sequential), ("batched", batched), ("shared", shared)]:
    fn(); sync(); ts = []
    for _ in range(5 if dev == "cuda" else 2):
        t = time.perf_counter(); p = fn(); sync(); ts.append(time.perf_counter() - t)
    ref = p if ref is None else ref
    res[name] = {"ms_per_state": round(1000 * float(np.median(ts)), 1), "fields_per_s": round(len(FIELDS) / float(np.median(ts)), 1), "max_abs_diff_vs_sequential": round(float(np.abs(p - ref).max()), 4)}
    print(name, res[name], flush=True)
res["answers_A_is_yes"] = [round(float(x), 3) for x in p]
json.dump(res, open(Path(__file__).parent / f"e13_results_{dev}.json", "w"), indent=1)
