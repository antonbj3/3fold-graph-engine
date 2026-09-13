"""MULTI-PATENT CLUSTER inverse — individual patents under-determined, the CLUSTER jointly recoverable (an agent worktree).

D's framing: F/G/H run their own patent CLUSTERS. The reason a cluster matters is identifiability: several patents share
the SAME underlying design but each claims a different measured property, so any one patent is usually under-determined
(few claims, more unknowns) while the STACK of the cluster's claims pins the shared design. This is the joint inverse —
concatenate the cluster's forwards over the shared unknowns and recover once. The eigen-spectrum gate shows the deficiency
fall as patents accrue: the cluster is what turns an unrecoverable design into a recoverable one.

Shared design x=(a,b,c); three patents each disclose ONE nonlinear property: P1=a²+b, P2=b²+c, P3=c²+a.

GATES (null/control each):
 (G0) CLUSTER — three patents share design (a,b,c); each contributes ONE claim (1 observable, 3 unknowns individually).
 (G1) ONE PATENT ⇒ UNDER-DETERMINED — serve_inverse on a single patent returns a REGION (deficiency 2), never a false point.
 (G2) CLUSTER ⇒ RECOVERABLE — the joint inverse over all three patents recovers the shared (a,b,c) as a trustworthy point (<2%).
 (G3) DEFICIENCY FALLS 2→1→0 — adding patents one at a time drives the eigen-spectrum deficiency 2→1→0: each independent
      patent removes one flat direction. The cluster is what makes the design identifiable. NULL = staying with one patent.

Run: python3 r.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from graph_engine.inverse_design.persona_design_gi_service import serve_inverse, identifiability

X_TRUE = np.array([1.2, 0.8, 1.5])
bnds = ([0.1, 0.1, 0.1], [3.0, 3.0, 3.0]); x0 = np.ones(3)
patents = [
    lambda x: np.array([x[0] ** 2 + x[1]]),                        # P1: a²+b
    lambda x: np.array([x[1] ** 2 + x[2]]),                        # P2: b²+c
    lambda x: np.array([x[2] ** 2 + x[0]]),                        # P3: c²+a
]


def cluster_forward(k):                                            # concatenate the first k patents' claims over shared x
    return lambda x: np.concatenate([patents[i](x) for i in range(k)])


def main():
    print("=" * 98)
    print("MULTI-PATENT CLUSTER inverse — individual patents under-determined, the CLUSTER jointly recoverable")
    print("=" * 98)
    g0 = len(patents) == 3 and all(len(p(X_TRUE)) == 1 for p in patents)
    print(f"\n(G0) CLUSTER — 3 patents share design (a,b,c)=({X_TRUE[0]},{X_TRUE[1]},{X_TRUE[2]}); each 1 claim (1 obs, 3 unknowns alone): {'PASS' if g0 else 'FAIL'}")

    r1 = serve_inverse(patents[0], patents[0](X_TRUE), x0, bnds)
    g1 = (not r1["trustworthy"]) and r1["status"] == "under-determined" and r1["deficiency"] == 2 and r1["region"] is not None
    print(f"\n(G1) ONE PATENT ⇒ UNDER — serve_inverse(P1) status='{r1['status']}', trustworthy={r1['trustworthy']}, deficiency={r1['deficiency']} → REGION: {'PASS' if g1 else 'FAIL'}")

    fwd3 = cluster_forward(3)
    r3 = serve_inverse(fwd3, fwd3(X_TRUE), x0, bnds)
    err = float(np.max(np.abs(r3["x_hat"] - X_TRUE) / X_TRUE))
    g2 = r3["trustworthy"] and err < 0.02
    print(f"\n(G2) CLUSTER ⇒ RECOVERABLE — joint inverse over 3 patents: trustworthy={r3['trustworthy']}, recovered (a,b,c)=({r3['x_hat'][0]:.2f},{r3['x_hat'][1]:.2f},{r3['x_hat'][2]:.2f}), err {100*err:.1f}%: {'PASS' if g2 else 'FAIL'}")

    defs = [identifiability(cluster_forward(k), x0)["deficiency"] for k in (1, 2, 3)]
    g3 = defs == [2, 1, 0]
    print(f"\n(G3) DEFICIENCY FALLS — eigen-spectrum deficiency as patents accrue: 1→{defs[0]}, 2→{defs[1]}, 3→{defs[2]} (each patent removes a flat direction): {'PASS' if g3 else 'FAIL'}")

    allok = g0 and g1 and g2 and g3
    print("\n" + "=" * 98)
    if allok:
        print("VERDICT: a patent CLUSTER is recoverable where its members are not — one patent pins only 1 of 3 design directions (a")
        print("  2-flat REGION, point refused), but stacking the cluster's claims over the shared design drives the eigen-spectrum deficiency")
        print(f"  2→1→0 and recovers (a,b,c) to {100*err:.1f}%. THAT is why F/G/H run clusters, not single patents — and the joint inverse is just")
        print(f"  serve_inverse on the concatenated forward, the same gate that refuses the under-determined single-patent point.")
    else:
        print(f"VERDICT: NOT all pass — G0 {g0} G1 {g1} G2 {g2} G3 {g3}. Fix at SOURCE.")
    print("=" * 98)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
