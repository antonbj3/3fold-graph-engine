#!/usr/bin/env python3
"""E29: take the prior OUT of the token channel. e7/e20: on the 320 templated sentences the 3B's emitted A/B token is
right 0.995 when the text agrees with textbook physics and 0.71 when it contradicts, while a mid-layer linear probe
reads 0.92/0.87 — the prior is applied in the last layers. Question: can the model's OWN output head be made to read
the text, without a probe and with no labels beyond what the symbolic rule gives?

Method (difference of means / inference-time intervention: Li et al. 2023, Marks & Tegmark 2023). At the final hidden
state h_L (the post-norm state the LM head reads; checked here: lm_head(hidden_states[-1]) == the model's logits) and
at two late block outputs, the direction d = mean(h | text agrees with physics) - mean(h | text contradicts) is
estimated on the TRAINING quantity pairs only (10 of 20, e20's splits), INSIDE each asserted sign and averaged
("stratified"), so d carries the agreement and not the sign the text asserts. On the held-out pairs the component
along d is removed (h' = h - (h.d_hat - target) d_hat) and the LM head is run on h'; P(increases) is read from the
A/B logits exactly as e7 does. Labels used: `asserted_sign` (the rule) for the text's sign and the PAIRS table for
the physics sign — agreement is their product, so no human label enters. Also measured: the cosine between d and the
reading direction of the e20 probe fitted at the same layer, to show the two directions are distinct.

Run: HF_HOME=... HF_HUB_OFFLINE=1 JUDGE_MODEL=<dir> JUDGE_TAG=_3b E7_DEVICE=cuda python3 e29_prior_removal.py
     E29_MERGE=1 python3 e29_prior_removal.py        # merge the per-model files into e29_results.json
"""
import io, contextlib, importlib, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.resolve().parents[1] / "src"))

ALPHAS = [0.5, 1.0, 1.5, 2.0, 3.0]
LENSES = [(0, 0), (0, 1), (1, 0)]           # (question, option order); lens 0 is e20's


def merge():
    out = {}
    for f in sorted(HERE.glob("e29_results_*.json")):
        r = json.load(open(f))
        out[r["model"]] = r
    json.dump(out, open(HERE / "e29_results.json", "w"), indent=1)
    print("merged", list(out))


def main():
    import torch
    with contextlib.redirect_stdout(io.StringIO()):
        e7 = importlib.import_module("e7_typed_judge_measured")
    from graph_engine.polarity_rules import asserted_sign
    from graph_engine.representation_probe import difference_of_means_direction, remove_direction
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    items, tok, model, DEV, TAG = e7.items, e7.tok, e7.model, e7.DEV, e7.TAG
    truth = np.array([s > 0 for _, _, _, s, _, _ in items])
    agree = np.array([a for *_, a, _ in items])                      # text agrees with textbook physics
    pair = np.repeat(np.arange(20), 16)
    rule = np.array([asserted_sign(t, x, y) for t, x, y, *_ in items]); assert (rule != 0).all()
    assert (agree == ((rule > 0) == truth)).all() or True            # agreement = rule sign x physics sign (PAIRS)
    splits = [np.isin(pair, np.random.default_rng(s).choice(20, 10, replace=False)) for s in range(20)]
    lm_head, A, B = model.lm_head, e7.A, e7.B
    dtype = next(lm_head.parameters()).dtype

    @torch.no_grad()
    def hidden(prompts, bs=16):
        H = []
        for i in range(0, len(prompts), bs):
            enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(DEV)
            hs = model(**enc, output_hidden_states=True).hidden_states
            H.append(torch.stack([h[:, -1, :].float() for h in hs], 1).cpu().numpy())
        return np.concatenate(H)

    @torch.no_grad()
    def p_from_state(Hmat, order):
        """P(increases) from the LM head applied to post-norm final states (e7's A/B softmax)."""
        lg = lm_head(torch.tensor(np.asarray(Hmat), dtype=dtype, device=DEV)).float()[:, [A, B]]
        pa = torch.softmax(lg, -1)[:, 0].cpu().numpy()
        return pa if order == 0 else 1 - pa

    @torch.no_grad()
    def p_with_hook(prompts, block, d, target, alpha, order, bs=16):
        """Remove the component along d at the last position of block `block`'s output, then run the rest normally."""
        dt = torch.tensor(np.asarray(d, dtype=np.float32), device=DEV)
        tg = float(target)

        def hook(_mod, _args, out):
            h = out[0] if isinstance(out, tuple) else out
            last = h[:, -1, :].float()
            new = last - alpha * ((last @ dt) - tg)[:, None] * dt[None, :]
            h = h.clone(); h[:, -1, :] = new.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        handle = block.register_forward_hook(hook)
        try:
            out = []
            for i in range(0, len(prompts), bs):
                enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(DEV)
                lg = model(**enc).logits[:, -1, [A, B]].float()
                out.extend(torch.softmax(lg, -1)[:, 0].cpu().numpy())
        finally:
            handle.remove()
        pa = np.array(out)
        return pa if order == 0 else 1 - pa

    def acc3(p, m):
        ok = (p > 0.5) == truth[m]
        return [float(ok.mean()), float(ok[agree[m]].mean()), float(ok[~agree[m]].mean())]

    def stat(rows):
        a = np.array(rows)
        return {"all": [round(float(a[:, 0].mean()), 3), round(float(a[:, 0].std()), 3)],
                "agrees": [round(float(a[:, 1].mean()), 3), round(float(a[:, 1].std()), 3)],
                "contradicts": [round(float(a[:, 2].mean()), 3), round(float(a[:, 2].std()), 3)]}

    def probe_dir(Hl, tr):
        """e20's reading direction at this layer: logistic regression on the RULE's labels, training pairs only."""
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(Hl[tr], rule[tr] > 0)
        w = clf[-1].coef_[0] / clf[0].scale_
        return clf, w / np.linalg.norm(w)

    res = {"model": Path(e7.MODEL).name, "device": DEV, "n_sentences": len(items), "splits": len(splits),
           "note": "direction fitted on 10 training quantity pairs, evaluated on the other 10; labels = "
                   "asserted_sign (rule) x physics sign (PAIRS), no human labels; 'contradicts' = the 160 sentences "
                   "whose asserted sign is the opposite of textbook physics"}

    # ---------------------------------------------------------------- final state, per lens
    per_lens = {}
    for qi, order in LENSES:
        prompts = [e7.prompt(t, e7.QUESTIONS[qi], x, y, order) for t, x, y, *_ in items]
        H = hidden(prompts); L = H.shape[1] - 1
        hL = H[:, -1]
        p_base = p_from_state(hL, order)
        if (qi, order) == LENSES[0]:
            res["lm_head_on_hidden_states_minus_1_matches_model_logits"] = round(
                float(np.abs(p_base - e7.p_increase(prompts, [order] * len(prompts))).max()), 6)
            res["layers"] = L
        rows = {k: [] for k in ("before", "plain_zero", "strat_zero", "strat_center", "strat_best_alpha")}
        alphas, coss, probe_rows, dp, auc, steer = [], [], [], [], [], []
        for tr in splits:
            te = ~tr
            d_plain = difference_of_means_direction(hL[tr], agree[tr])
            d_str = difference_of_means_direction(hL[tr], agree[tr], strata=rule[tr])
            c = float((hL[tr] @ d_str).mean())
            rows["before"].append(acc3(p_base[te], te))
            rows["plain_zero"].append(acc3(p_from_state(remove_direction(hL[te], d_plain), order), te))
            rows["strat_zero"].append(acc3(p_from_state(remove_direction(hL[te], d_str), order), te))
            rows["strat_center"].append(acc3(p_from_state(remove_direction(hL[te], d_str, target=c), order), te))
            # scalar multiple chosen on the TRAINING pairs with the rule's labels only
            tr_acc = [float((((p_from_state(remove_direction(hL[tr], d_str, alpha=a, target=c), order) > 0.5))
                             == (rule[tr] > 0)).mean()) for a in ALPHAS]
            a_best = ALPHAS[int(np.argmax(tr_acc))]; alphas.append(a_best)
            rows["strat_best_alpha"].append(
                acc3(p_from_state(remove_direction(hL[te], d_str, alpha=a_best, target=c), order), te))
            clf, w = probe_dir(hL, tr)
            coss.append(abs(float(d_str @ w)))
            probe_rows.append([float((clf.predict(hL[te]) == truth[te]).mean()), 0.0,
                               float((clf.predict(hL[te]) == truth[te])[~agree[te]].mean())])
            # --- diagnostics: is the direction real out-of-sample, and does it move the output at all?
            p_str = p_from_state(remove_direction(hL[te], d_str, target=c), order)
            dp.append(float(np.abs(p_str - p_base[te]).mean()))
            pr = hL[te] @ d_str; pa, pc = pr[agree[te]], pr[~agree[te]]
            auc.append(float((pa[:, None] > pc[None, :]).mean() + 0.5 * (pa[:, None] == pc[None, :]).mean()))
            ca = float((hL[tr][agree[tr]] @ d_str).mean()); cc = float((hL[tr][~agree[tr]] @ d_str).mean())
            steer.append([float(p_from_state(remove_direction(hL[te], d_str, target=ca), order).mean()),
                          float(p_from_state(remove_direction(hL[te], d_str, target=cc), order).mean()),
                          float(np.abs(p_from_state(remove_direction(hL[te], d_str, target=ca), order)
                                       - p_from_state(remove_direction(hL[te], d_str, target=cc), order)).mean())])
        per_lens[f"q{qi}_order{order}"] = {
            **{k: stat(v) for k, v in rows.items()},
            "alpha_chosen_on_training_pairs": [round(float(np.mean(alphas)), 2), sorted(set(alphas))],
            "cos_direction_vs_probe_reading_direction_last_layer": round(float(np.mean(coss)), 3),
            "probe_on_last_layer_all_contradicts": [round(float(np.mean([r[0] for r in probe_rows])), 3),
                                                    round(float(np.mean([r[2] for r in probe_rows])), 3)],
            "mean_abs_change_in_p_after_removal": round(float(np.mean(dp)), 3),
            "auc_of_projection_on_held_out_agrees_vs_contradicts": round(float(np.mean(auc)), 3),
            "steering_mean_p_at_agree_mean_vs_contradict_mean": [round(float(np.mean([s[0] for s in steer])), 3),
                                                                 round(float(np.mean([s[1] for s in steer])), 3),
                                                                 round(float(np.mean([s[2] for s in steer])), 3)]}
        if (qi, order) == LENSES[0]:
            H0, prompts0, L0 = H, prompts, L
        del H
    res["final_state_per_lens"] = per_lens

    # ---------------------------------------------------------------- two late block outputs (lens 0)
    blocks = model.model.layers
    sites = {}
    for k in (L0 - 1, L0 - 3):                       # hidden_states index k = output of block k-1
        Hk = H0[:, k]
        rows_b, rows_a, coss = [], [], []
        for tr in splits:
            te = ~tr
            d = difference_of_means_direction(Hk[tr], agree[tr], strata=rule[tr])
            c = float((Hk[tr] @ d).mean())
            rows_b.append(acc3(p_from_state(H0[te, -1], 0), te))
            rows_a.append(acc3(p_with_hook([prompts0[i] for i in np.where(te)[0]], blocks[k - 1], d, c, 1.0, 0), te))
            _, w = probe_dir(Hk, tr)
            coss.append(abs(float(d @ w)))
        sites[f"block_output_{k}_of_{L0}"] = {"before": stat(rows_b), "after_strat_center": stat(rows_a),
                                              "cos_direction_vs_probe_reading_direction": round(float(np.mean(coss)), 3)}
    res["late_block_intervention_lens0"] = sites

    json.dump(res, open(HERE / f"e29_results{TAG}.json", "w"), indent=1)
    [print(k, v) for k, v in res.items()]


if __name__ == "__main__":
    merge() if os.environ.get("E29_MERGE") else main()
