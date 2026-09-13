#!/usr/bin/env python3
"""d_llm_federation_gen.py -- RAW generation pass (GPU-bound). Produces raw per-claim, per-view sample lists for
every (model, condition) required by d_llm_federation_PREREG.md. Saves RAW samples (not pre-aggregated scores) so
the analysis pass (d_llm_federation_analysis.py, CPU-only) can recompute fusion/AUC without re-touching the GPU.

THE TENSOR BUG FIX (per task instructions): tok.apply_chat_template(msgs, add_generation_prompt=True,
return_tensors='pt') returns a bare TENSOR -- passing that into model.generate(**tensor) raises "argument after
** must be a mapping" (a Tensor is not a Mapping). Fixed structurally here by NEVER calling apply_chat_template
with return_tensors='pt': render to TEXT first (tokenize=False), then tok(text, return_tensors="pt") which
returns a BatchEncoding (IS a mapping, safe for **enc). Verified against all 4 models in pilot_harness.py.

Also fixed vs an agent worktree's l_llm_coherent_fake.py: L's sample_k constructed a torch.Generator but never passed it
to generate (transformers rejects a `generator=` kwarg outright -- ValueError, confirmed here) -- so their
"seeded" sampling silently ran on global RNG state. Fixed here via torch.manual_seed before each call.

Usage:python src/graph_engine/llm_federation_gen.py [--pilot]
"""
import os, sys, json, time, argparse, hashlib
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, "/tmp")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from gpu_lock import gpu_lock            # optional external GPU mutex
except ImportError:
    import contextlib

    @contextlib.contextmanager
    def gpu_lock(*a, **kw):
        yield
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_federation_data import build_claims, SYS_MSG

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_OUT = os.path.join(OUT_DIR, "d_llm_federation_raw.json")

K_FED = 5     # per-view samples in the federation conditions (3 views x 5 = 15 total, compute-matched)
K_SINGLE = 15  # single-view compute-matched samples
TEMP = 0.7

def build_text(tok, user_msg, enable_thinking=False):
    msgs = [{"role": "system", "content": SYS_MSG}, {"role": "user", "content": user_msg}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tok.apply_chat_template(msgs, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tok.apply_chat_template(msgs, **kwargs)

@torch.no_grad()
def gen_batch(tok, model, dev, user_msg, k, max_new=10, temp=TEMP, seed=0, greedy=False):
    text = build_text(tok, user_msg)
    enc = tok(text, return_tensors="pt").to(dev)
    torch.manual_seed(seed)
    if greedy:
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, num_return_sequences=1,
                              pad_token_id=tok.eos_token_id)
    else:
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=0.95,
                              num_return_sequences=k, pad_token_id=tok.eos_token_id)
    plen = enc["input_ids"].shape[1]
    return [tok.decode(out[j, plen:], skip_special_tokens=True).strip() for j in range(out.shape[0])]

def m1_prompt(claim_text):
    return f"True or False: {claim_text} Answer with just one word: True or False."

def m2_prompt(row):
    return row["decorr_q_tmpl"].format(ans=row["claim_answer"])

def m3_prompt(row):
    return row["fwd_q"]

def det_seed(*parts):
    """Deterministic seed across process runs -- Python's builtin hash randomizes str hashing per-process
    (PYTHONHASHSEED), which would silently make every re-run of this script draw different samples."""
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) % 1_000_000

def run_item_views(tok, model, dev, row, seed_base, k, max_new_m1=6, max_new_m2=20, max_new_m3=20):
    """Returns raw sample lists for M1 (T/F), M2 (decorr-crosscheck), M3 (independent-recall), one batched call each."""
    m1 = gen_batch(tok, model, dev, m1_prompt(row["claim_text"]), k, max_new=max_new_m1, seed=seed_base + 1)
    m2 = gen_batch(tok, model, dev, m2_prompt(row), k, max_new=max_new_m2, seed=seed_base + 2)
    m3 = gen_batch(tok, model, dev, m3_prompt(row), k, max_new=max_new_m3, seed=seed_base + 3)
    return dict(m1=m1, m2=m2, m3=m3)

def run_condition(tok, model, dev, claims, condition, model_tag, log):
    """condition in {'federation' (3 views x K_FED), 'single' (M1 only x K_SINGLE)}."""
    rows_out = []
    t0 = time.time()
    for i, row in enumerate(claims):
        seed_base = det_seed(model_tag, condition, row["category"], row["item"], row["label_fake"]) * 10
        if condition == "federation":
            views = run_item_views(tok, model, dev, row, seed_base, K_FED)
        elif condition == "single":
            m1 = gen_batch(tok, model, dev, m1_prompt(row["claim_text"]), K_SINGLE, max_new=6, seed=seed_base + 1)
            views = dict(m1=m1, m2=[], m3=[])
        else:
            raise ValueError(condition)
        rows_out.append(dict(category=row["category"], item=row["item"], label_fake=row["label_fake"],
                              claim_answer=row["claim_answer"], ground_truth=row["ground_truth"],
                              claim_text=row["claim_text"], views=views))
        if (i + 1) % 15 == 0:
            log(f"    [{condition}] {i+1}/{len(claims)}  ({time.time()-t0:.1f}s elapsed)")
    log(f"    [{condition}] DONE {len(claims)} claims in {time.time()-t0:.1f}s")
    return rows_out

def run_capability_floor(tok, model, dev, claims, log):
    """Greedy (K=1) forward-recall + M1 True/False, for the capability-floor (bias) measurement."""
    # de-dup forward-recall to one per (category,item) -- fwd_q doesn't depend on true/fake
    seen = {}
    fwd_rows = []
    t0 = time.time()
    for row in claims:
        key = (row["category"], row["item"])
        if key in seen:
            continue
        seen[key] = True
        ans = gen_batch(tok, model, dev, row["fwd_q"], 1, max_new=10, seed=999, greedy=True)[0]
        fwd_rows.append(dict(category=row["category"], item=row["item"], ground_truth=row["ground_truth"], raw=ans))
    log(f"    [capability fwd-recall] {len(fwd_rows)} items in {time.time()-t0:.1f}s")
    t0 = time.time()
    tf_rows = []
    for row in claims:
        ans = gen_batch(tok, model, dev, m1_prompt(row["claim_text"]), 1, max_new=6, seed=998, greedy=True)[0]
        tf_rows.append(dict(category=row["category"], item=row["item"], label_fake=row["label_fake"],
                             claim_text=row["claim_text"], raw=ans))
    log(f"    [capability M1-TF] {len(tf_rows)} claims in {time.time()-t0:.1f}s")
    return dict(fwd_recall=fwd_rows, tf_judgment=tf_rows)

MODEL_PLAN = {
    "smollm135m": dict(hf="HuggingFaceTB/SmolLM2-135M-Instruct", conditions=["federation", "single"], capability=True),
    "qwen05b":    dict(hf="Qwen/Qwen2.5-0.5B-Instruct",          conditions=["federation", "single"], capability=True),
    "qwen15b":    dict(hf="Qwen/Qwen2.5-1.5B-Instruct",          conditions=["federation", "single"], capability=False),
    "qwen06b":    dict(hf="Qwen/Qwen3-0.6B",                     conditions=[],                        capability=True),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="tiny subset (6 claims) end-to-end smoke test")
    ap.add_argument("--only", default=None, help="comma-separated subset of MODEL_PLAN keys")
    a = ap.parse_args()

    claims = build_claims()
    if a.pilot:
        claims = claims[:3] + claims[45:48]  # 3 capitals-true+fake-ish + 3 elements, tiny smoke test
    print(f"n_claims={len(claims)}")

    plan = MODEL_PLAN if not a.only else {k: MODEL_PLAN[k] for k in a.only.split(",")}
    results = {}
    if os.path.exists(RAW_OUT) and not a.pilot:
        results = json.load(open(RAW_OUT))

    for tag, spec in plan.items():
        def log(msg, tag=tag):
            print(f"[{tag}] {msg}", flush=True)
        log(f"loading {spec['hf']} ...")
        t0 = time.time()
        with gpu_lock(what=f"wave D-federation gen ({tag})", timeout=600):
            dev = "cuda"
            tok = AutoTokenizer.from_pretrained(spec["hf"])
            model = AutoModelForCausalLM.from_pretrained(spec["hf"], torch_dtype=torch.bfloat16).to(dev).eval()
            log(f"loaded in {time.time()-t0:.1f}s")
            model_result = results.get(tag, {})
            for cond in spec["conditions"]:
                model_result[cond] = run_condition(tok, model, dev, claims, cond, tag, log)
            if spec["capability"]:
                model_result["capability"] = run_capability_floor(tok, model, dev, claims, log)
            results[tag] = model_result
            del model
            torch.cuda.empty_cache()
        # save after each model releases the lock -- bounded lock hold time, incremental progress persisted
        if not a.pilot:
            with open(RAW_OUT, "w") as f:
                json.dump(results, f)
            log(f"saved -> {RAW_OUT}")

    print("ALL DONE" if not a.pilot else "PILOT DONE (not saved)")
    if a.pilot:
        print(json.dumps(results, indent=2)[:3000])

if __name__ == "__main__":
    main()
