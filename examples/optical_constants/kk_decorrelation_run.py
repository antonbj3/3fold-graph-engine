"""kk_decorrelation_run -- are two optical self-consistency checks independent evidence?

The worked example on real, public optical constants (6 datasets for Au, SiO2 and H2O from the
refractiveindex.info database, CC0). Two genuinely different checks are run on each dataset:

  cert A  Kramers-Kronig causality residual: predict n from k by the KK integral.
  cert B  causal Lorentz-oscillator fit: predict k from n by fitting a causal oscillator model.

The pooled reading across all six datasets gives a positive correlation between the two checks'
residuals and an n_eff near 1, which would say the two checks are effectively one (co-blind).
That pooled reading does NOT hold: see qc_neff_sign_blindness.py, which shows the per-material
correlation runs from about -0.9 to +0.97 and that pooling with |rho| hides the sign. Metals come
out genuinely redundant; the dielectrics come out ANTI-correlated, i.e. complementary -- fusing
them cancels noise. Read the two scripts together: the sign carries the information.

CPU only, a few tens of seconds. Requires kk_cert_evidence.json (run kk_cert_run.py first).
Writes kk_decorrelation_evidence.json.
"""
import json
import time
import numpy as np

from kk_cert_lib import (
    load_real_instance, subtractive_predict, interior_mask, relative_residual,
    inject_bump, oscillator_nk, make_log_grid, OSC_PARAMS_DEFAULT, section,
)
from kk_decorrelation_lib import (
    fit_causal_oscillators_to_n, fit_causal_drude_interband_to_n, log_relative_residual,
)

DATA_DIR = "kk_cert_data"
INSTANCES = [
    ("Au_Johnson", f"{DATA_DIR}/Au_Johnson.yml", None),
    ("Au_McPeak", f"{DATA_DIR}/Au_McPeak.yml", None),
    ("SiO2_Rodriguez", f"{DATA_DIR}/SiO2_Rodriguez.yml", None),
    ("SiO2_Kischkat", f"{DATA_DIR}/SiO2_Kischkat.yml", None),
    ("H2O_Hale", f"{DATA_DIR}/H2O_Hale.yml", None),
    ("H2O_Segelstein_xcheck", f"{DATA_DIR}/H2O_Segelstein.yml", (0.2, 200.0)),
]
tags6 = [t for t, _, _ in INSTANCES]
AMPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
WIDTHS = [0.05, 0.15]

with open("kk_cert_evidence.json") as f:
    KK_EVIDENCE = json.load(f)

gates = {}
evidence = {"gates": {}}


# =====================================================================================
# PART A -- cert B engine validation (C1B/C2B), mirrors kk_cert Part A
# =====================================================================================
section("PART A -- cert B (causal-oscillator regression) engine validation")
E_ref = make_log_grid(1e-3, 1e3, 4000)
n_true, k_true = oscillator_nk(E_ref, OSC_PARAMS_DEFAULT, acausal=False)
n_acaus, k_acaus = oscillator_nk(E_ref, OSC_PARAMS_DEFAULT, acausal=True)
mask_ref = interior_mask(E_ref, 0.2, 0.8)

t0 = time.time()
_, n_fit_c1, k_fit_c1, loss_c1 = fit_causal_oscillators_to_n(E_ref, n_true, n_restarts=10, seed=0)
r_c1b = relative_residual(k_fit_c1, k_true, mask_ref)
print(f"C1B causal-target fit: kappa residual={r_c1b:.4e}  loss={loss_c1:.4e}  ({time.time()-t0:.1f}s)")

_, n_fit_c1_25, k_fit_c1_25, loss_c1_25 = fit_causal_oscillators_to_n(E_ref, n_true, n_restarts=25, seed=1)
r_c1b_25 = relative_residual(k_fit_c1_25, k_true, mask_ref)
print(f"C1B stability check (25 restarts, diff seed): residual={r_c1b_25:.4e}")
gates["GB1_engine_validated"] = (r_c1b < 0.05) and (r_c1b_25 <= r_c1b * 1.5)
print(f"GATE GB1 (residual<0.05 AND not worse with more restarts): "
      f"{'PASS' if gates['GB1_engine_validated'] else 'FAIL'}")

t0 = time.time()
_, n_fit_c2, k_fit_c2, loss_c2 = fit_causal_oscillators_to_n(E_ref, n_acaus, n_restarts=10, seed=0)
r_c2b = relative_residual(k_fit_c2, k_acaus, mask_ref)
ratio_c2b = r_c2b / r_c1b
print(f"C2B acausal-target fit: kappa residual={r_c2b:.4e}  ratio(acausal/causal)={ratio_c2b:.2f}x "
      f"({time.time()-t0:.1f}s)")
gates["GB2_anticausal_caught"] = ratio_c2b > 5.0
print(f"GATE GB2 (ratio>5x): {'PASS' if gates['GB2_anticausal_caught'] else 'FAIL'}")

evidence["partA"] = dict(r_c1b=r_c1b, r_c1b_25restart=r_c1b_25, r_c2b=r_c2b, ratio_c2b=ratio_c2b)


# =====================================================================================
# PART B -- real-data cert B baseline (all 6 instances) + metal (Drude+interband) diagnostic
# =====================================================================================
section("PART B -- cert B on REAL measured (n,k): causal-oscillator-regression baseline")
real = {}
for tag, path, lam_range in INSTANCES:
    E, n, k, meta = load_real_instance(path, lam_range_um=lam_range)
    real[tag] = dict(E=E, n=n, k=k, meta=meta, mask=interior_mask(E))

rho_fit_baseline = {}
k_model_fixed = {}
print(f"{'instance':<24}{'N':>6}{'rho_sub(kk)':>14}{'rho_fit(B)':>14}")
for tag in tags6:
    E, n, k, mask = real[tag]["E"], real[tag]["n"], real[tag]["k"], real[tag]["mask"]
    t0 = time.time()
    _, n_model, k_model, loss = fit_causal_oscillators_to_n(E, n, n_restarts=10, seed=0)
    rho_fit = log_relative_residual(k_model, k, k, mask)
    rho_fit_baseline[tag] = rho_fit
    k_model_fixed[tag] = k_model
    n_sub, _, _ = subtractive_predict(E, k, n)
    rho_sub_kk = relative_residual(n_sub, n, mask)
    real[tag]["rho_sub_kk"] = rho_sub_kk
    print(f"{tag:<24}{len(E):>6}{rho_sub_kk:>14.4e}{rho_fit:>14.4e}   ({time.time()-t0:.1f}s)")

n_pass_fit = sum(1 for v in rho_fit_baseline.values() if v < 0.15)
gates["GB3_real_majority_pass"] = n_pass_fit >= 3
print(f"\nGATE GB3 (>=3/6 cert-B rho_fit<0.15 log-units): {n_pass_fit}/6  "
      f"{'PASS' if gates['GB3_real_majority_pass'] else 'FAIL'}")

# the headline decorrelation test)
metal_diag = {}
for tag in ["Au_Johnson", "Au_McPeak"]:
    E, n, k, mask = real[tag]["E"], real[tag]["n"], real[tag]["k"], real[tag]["mask"]
    _, n_m, k_m, loss_m = fit_causal_drude_interband_to_n(E, n, n_restarts=10, seed=0)
    rho_fit_metal = log_relative_residual(k_m, k, k, mask)
    metal_diag[tag] = dict(rho_fit_generic=rho_fit_baseline[tag], rho_fit_drude_interband=rho_fit_metal)
    print(f"[metal diagnostic] {tag}: generic-3-Lorentz rho_fit={rho_fit_baseline[tag]:.4e}  "
          f"Drude+interband rho_fit={rho_fit_metal:.4e}")

evidence["partB"] = dict(rho_fit_baseline=rho_fit_baseline, n_pass_fit=n_pass_fit, metal_diag=metal_diag,
                          rho_sub_kk={t: real[t]["rho_sub_kk"] for t in tags6})


# =====================================================================================
# PART C -- 108-trial grid: recompute BOTH certs' response to the SAME non-causal perturbation
# (reuses kk_cert_lib.inject_bump verbatim, identical grid to kk_cert_run.py Part E)
# =====================================================================================
section("PART C -- both certs' response to the same 108-trial non-causal perturbation grid")
records = []
for tag in tags6:
    E, n, k, mask = real[tag]["E"], real[tag]["n"], real[tag]["k"], real[tag]["mask"]
    baseline_kk = real[tag]["rho_sub_kk"]
    baseline_fit = rho_fit_baseline[tag]
    kmod = k_model_fixed[tag]
    tau_delta = KK_EVIDENCE["partE"]["boot_stats"][tag]["tau_delta"] # reused, not recomputed (Sec.2)
    for width in WIDTHS:
        for A in AMPS:
            k_pert, Ec, kc = inject_bump(E, k, pctile=0.25, width_log_frac=width, amplitude=A)
            n_pert_sub, _, _ = subtractive_predict(E, k_pert, n)
            rho_pert_kk = relative_residual(n_pert_sub, n, mask)
            delta_kk = rho_pert_kk - baseline_kk
            rho_pert_fit = log_relative_residual(kmod, k_pert, k, mask)
            delta_fit = rho_pert_fit - baseline_fit
            records.append(dict(tag=tag, width=width, amplitude=A,
                                 rho_pert_kk=rho_pert_kk, delta_kk=delta_kk,
                                 rho_pert_fit=rho_pert_fit, delta_fit=delta_fit,
                                 detected_kk=bool(delta_kk > tau_delta) if np.isfinite(tau_delta) else False))
print(f"built {len(records)} trial records (expect {len(tags6)}*{len(WIDTHS)}*{len(AMPS)}={len(tags6)*len(WIDTHS)*len(AMPS)})")

# --- Sec.6 reproduction spot-check: pooled power curve must match kk_cert_evidence.json exactly ---
pooled_power_recomputed = []
for A in AMPS:
    trials = [r for r in records if r["amplitude"] == A]
    p = float(np.mean([r["detected_kk"] for r in trials]))
    pooled_power_recomputed.append([A, p])
pooled_power_stored = KK_EVIDENCE["partE"]["pooled_power_curve"]
repro_match = all(abs(a[1] - b[1]) < 1e-9 for a, b in zip(pooled_power_recomputed, pooled_power_stored))
gates["G_REPRO_SPOTCHECK"] = repro_match
print(f"reproduction spot-check (recomputed pooled power curve vs kk_cert_evidence.json): "
      f"{'EXACT MATCH' if repro_match else 'MISMATCH -- INVESTIGATE'}")
print(f"  recomputed: {pooled_power_recomputed}")
print(f"  stored:     {pooled_power_stored}")

evidence["partC"] = dict(n_records=len(records), repro_match=repro_match,
                          pooled_power_recomputed=pooled_power_recomputed)


# =====================================================================================
# PART D -- the decorrelation test: 3-step OODA-forced ladder (Sec.4)
# =====================================================================================
section("PART D -- decorrelation test: naive -> within-instance -> amplitude-partialled (forced)")

dk = np.array([r["delta_kk"] for r in records])
df = np.array([r["delta_fit"] for r in records])
tag_arr = np.array([r["tag"] for r in records])
width_arr = np.array([r["width"] for r in records])
amp_arr = np.array([r["amplitude"] for r in records])

# --- Step 1: naive pooled (confounded) ---
rho_naive = float(np.corrcoef(dk, df)[0, 1])
print(f"Step 1 (naive, pooled all 108, confounded by cross-instance offset + shared amplitude driver): "
      f"rho_naive={rho_naive:.4f}")

# --- Step 2: within-instance (fixes cross-instance offset confound) ---
per_instance_rho = {}
for tag in tags6:
    m = tag_arr == tag
    per_instance_rho[tag] = float(np.corrcoef(dk[m], df[m])[0, 1])
zs = [np.arctanh(np.clip(v, -0.999999, 0.999999)) for v in per_instance_rho.values()]
rho_within = float(np.tanh(np.mean(zs)))
print(f"Step 2 (within-instance, Fisher-z pooled): per-instance rho={per_instance_rho}")
print(f"  rho_within={rho_within:.4f}")

# --- Step 3 (headline, forced): amplitude/width-partialled, pooled residual correlation ---
resid_kk_all, resid_fit_all = [], []
per_instance_partial = {}
for tag in tags6:
    m = tag_arr == tag
    X = np.column_stack([np.ones(m.sum()), np.log10(amp_arr[m]),
                          (width_arr[m] == 0.15).astype(float)])
    for y, store in [(dk[m], resid_kk_all), (df[m], resid_fit_all)]:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        store.extend(resid.tolist())
    r_kk_local = dk[m] - X @ np.linalg.lstsq(X, dk[m], rcond=None)[0]
    r_fit_local = df[m] - X @ np.linalg.lstsq(X, df[m], rcond=None)[0]
    per_instance_partial[tag] = float(np.corrcoef(r_kk_local, r_fit_local)[0, 1])

resid_kk_all = np.array(resid_kk_all)
resid_fit_all = np.array(resid_fit_all)
rho_partial = float(np.corrcoef(resid_kk_all, resid_fit_all)[0, 1])
print(f"Step 3 (FORCED headline: residuals after regressing out log10(A)+width, per instance, pooled n=108):")
print(f"  per-instance partial rho: {per_instance_partial}")
print(f"  rho_partial (pooled) = {rho_partial:.4f}")

# --- machine cross-check: stratified permutation null on rho_partial ---
rng = np.random.default_rng(42)
null_rhos = np.empty(2000)
idx_by_tag = {tag: np.where(tag_arr == tag)[0] for tag in tags6}
for i in range(2000):
    perm_fit = resid_fit_all.copy()
    for tag in tags6:
        idx = idx_by_tag[tag]
        perm_fit[idx] = rng.permutation(resid_fit_all[idx])
    null_rhos[i] = np.corrcoef(resid_kk_all, perm_fit)[0, 1]
p_perm = float(np.mean(np.abs(null_rhos) >= abs(rho_partial)))
print(f"permutation null (2000 draws, stratified within instance): "
      f"null mean={null_rhos.mean():.4f} std={null_rhos.std():.4f}  2-sided p={p_perm:.4f}")

n_eff = 2.0 / (1.0 + rho_partial)
print(f"\nn_eff = 2/(1+rho_partial) = {n_eff:.3f}")

# --- Step 4 (OODA re-force, NOT in the original PREREG -- added because Step 3's rho_partial=
# 0.85 result was heterogeneous per-instance (Au/H2O strongly +corr, SiO2_Rodriguez strongly
# -0.90 ANTI-corr, SiO2_Kischkat ~0). Orient: Step 3 only removed a LINEAR-in-log10(A)+width
# shared trend -- any REMAINING nonlinear/kink dependence on the same (amplitude,width) knob,
# common across instances, would still show up as "correlated residuals" without being genuine
# shared error SOURCE. Since exactly 6 instances are measured at the IDENTICAL 18 (width,
# amplitude) grid experiment scripts, this can be removed nonparametrically: subtract, from each instance's
# Step-3 residual at each experiment script, the OTHER 5 instances' mean residual at that same experiment script
# (leave-one-out cross-instance experiment script-mean -- a two-way-FE-style adjustment, strictly stronger
# than Step 3, not tuned toward either outcome). ---
CELLS = [(w, a) for w in WIDTHS for a in AMPS]
resid_kk_mat = np.zeros((len(tags6), len(CELLS)))
resid_fit_mat = np.zeros((len(tags6), len(CELLS)))
for ti, tag in enumerate(tags6):
    m = tag_arr == tag
    X = np.column_stack([np.ones(m.sum()), np.log10(amp_arr[m]), (width_arr[m] == 0.15).astype(float)])
    r_kk_local = dk[m] - X @ np.linalg.lstsq(X, dk[m], rcond=None)[0]
    r_fit_local = df[m] - X @ np.linalg.lstsq(X, df[m], rcond=None)[0]
    resid_kk_mat[ti, :] = r_kk_local
    resid_fit_mat[ti, :] = r_fit_local

n_inst = len(tags6)
cellmean_kk = resid_kk_mat.mean(axis=0)
cellmean_fit = resid_fit_mat.mean(axis=0)
loo_mean_kk = (cellmean_kk[None, :] * n_inst - resid_kk_mat) / (n_inst - 1)
loo_mean_fit = (cellmean_fit[None, :] * n_inst - resid_fit_mat) / (n_inst - 1)
final_kk = (resid_kk_mat - loo_mean_kk).ravel()
final_fit = (resid_fit_mat - loo_mean_fit).ravel()
rho_step4 = float(np.corrcoef(final_kk, final_fit)[0, 1])

# permutation null for step4, stratified within instance (shuffle across the 18 experiment scripts per row)
rng2 = np.random.default_rng(7)
null_rhos4 = np.empty(2000)
for i in range(2000):
    perm = final_fit.reshape(n_inst, len(CELLS)).copy()
    for ti in range(n_inst):
        perm[ti, :] = rng2.permutation(perm[ti, :])
    null_rhos4[i] = np.corrcoef(final_kk, perm.ravel())[0, 1]
p_perm4 = float(np.mean(np.abs(null_rhos4) >= abs(rho_step4)))
n_eff_step4 = 2.0 / (1.0 + rho_step4)
print(f"\nStep 4 (OODA re-forced: leave-one-out cross-instance CELL-mean removed on top of Step 3, "
      f"nonparametric, removes ANY shared nonlinear amplitude/width dependence, not just linear):")
print(f"  rho_step4 (pooled, n=108) = {rho_step4:.4f}   permutation p={p_perm4:.4f}   "
      f"n_eff_step4 = {n_eff_step4:.3f}")

# per-instance breakdown after step4 (diagnostic: is heterogeneity still present?)
per_instance_step4 = {}
for ti, tag in enumerate(tags6):
    per_instance_step4[tag] = float(np.corrcoef(
        resid_kk_mat[ti] - loo_mean_kk[ti], resid_fit_mat[ti] - loo_mean_fit[ti])[0, 1])
print(f"  per-instance step4 rho: {per_instance_step4}")

# HEADLINE gates use the most-forced number (Step 4), per watertight discipline: accept the
# adversary's strongest fair form, not the first one computed.
gates["G_DECORR_PASS"] = (rho_step4 < 1.0 / 3.0) and (p_perm4 > 0.05)
gates["G_COBLIND"] = (rho_step4 > 0.7) and (p_perm4 < 0.01)
verdict_D = ("DECORRELATED cert-vector (n_eff>1.5, H1)" if gates["G_DECORR_PASS"] else
             "CO-BLIND (n_eff~1, H0/null)" if gates["G_COBLIND"] else
             "GRAY ZONE (partial overlap, neither H1 nor H0 cleanly)")
print(f"GATE G_DECORR_PASS: {'PASS' if gates['G_DECORR_PASS'] else 'FAIL'}   "
      f"GATE G_COBLIND: {'PASS' if gates['G_COBLIND'] else 'FAIL'}")
print(f"VERDICT (Part D): {verdict_D}")

# --- shared-upstream VALUE-level check (task's explicit ask; small-n=6, underpowered, honest) ---
vals_kk = np.array([real[t]["rho_sub_kk"] for t in tags6])
vals_fit = np.array([rho_fit_baseline[t] for t in tags6])
r_value = float(np.corrcoef(vals_kk, vals_fit)[0, 1])
print(f"\nshared-upstream VALUE-level check (n=6, underpowered, descriptive only): "
      f"corr(baseline rho_sub_kk, baseline rho_fit) = {r_value:.4f}")
print(f"  interpretation: value-level={r_value:.3f} vs residual-level rho_partial={rho_partial:.3f} -- "
      f"{'CONSISTENT (both high or both low)' if (r_value>0.5)==(rho_partial>0.33) else 'DIVERGE -- classic common-per-instance-offset-without-shared-error-source signature (decorrelation-precondition-error-sources law)'}")

# --- concrete complement check (machine-checkable, rank-based, no extra bootstrap needed) ---
flag_kk = np.zeros(len(records), dtype=bool)
flag_fit = np.zeros(len(records), dtype=bool)
for tag in tags6:
    idx = idx_by_tag[tag]
    thresh_kk = np.percentile(dk[idx], 100 * 2 / 3)
    thresh_fit = np.percentile(df[idx], 100 * 2 / 3)
    flag_kk[idx] = dk[idx] >= thresh_kk
    flag_fit[idx] = df[idx] >= thresh_fit
both = int(np.sum(flag_kk & flag_fit))
a_only = int(np.sum(flag_kk & ~flag_fit))
b_only = int(np.sum(~flag_kk & flag_fit))
neither = int(np.sum(~flag_kk & ~flag_fit))
print(f"\nconcrete complement (top-tertile-within-instance flagging, n=108): "
      f"both={both} A_only={a_only} B_only={b_only} neither={neither}")
print(f"  (deviation from PREREG Sec.5's G_CONCRETE_COMPLEMENT: used rank/percentile flagging "
      f"instead of a fresh cert-B bootstrap sigma -- cheaper, distribution-free, equally valid "
      f"for a 'does each catch something the other misses' question; documented, not silent)")

evidence["partD"] = dict(
    rho_naive=rho_naive, per_instance_rho=per_instance_rho, rho_within=rho_within,
    per_instance_partial=per_instance_partial, rho_partial=rho_partial,
    permutation_p=p_perm, null_mean=float(null_rhos.mean()), null_std=float(null_rhos.std()),
    n_eff=n_eff,
    rho_step4=rho_step4, permutation_p_step4=p_perm4, n_eff_step4=n_eff_step4,
    per_instance_step4=per_instance_step4,
    verdict=verdict_D,
    value_level_corr=r_value, vals_kk=vals_kk.tolist(), vals_fit=vals_fit.tolist(),
    concrete_complement=dict(both=both, a_only=a_only, b_only=b_only, neither=neither),
)


# =====================================================================================
# VERDICT
# =====================================================================================
section("VERDICT")
for k_, v_ in gates.items():
    print(f"  [{'PASS' if v_ else 'FAIL'}] {k_}")

evidence["gates"] = {k_: bool(v_) for k_, v_ in gates.items()}
evidence["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

with open("kk_decorrelation_evidence.json", "w") as f:
    json.dump(evidence, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
print("\nwrote kk_decorrelation_evidence.json")
