"""The optical worked example: the QC re-analysis runs and reproduces its four gates."""
import json
import subprocess
import sys
from pathlib import Path

EX = Path(__file__).resolve().parent.parent / "examples" / "optical_constants"


def test_qc_reanalysis_reproduces_its_gates(tmp_path):
    r = subprocess.run([sys.executable, str(EX / "qc_neff_sign_blindness.py")],
                       capture_output=True, text=True, cwd=str(EX))
    assert r.returncode == 0, r.stdout + r.stderr
    ev = json.loads((EX / "qc_neff_sign_blindness_evidence.json").read_text())
    gates = ev["gates"]
    assert gates["G1_pooling_confound_sign_flip"] is True
    assert gates["G2_abs_rho_neff_is_sign_blind"] is True
    assert gates["G3_not_all_materials_coblind_sio2_anticomplementary"] is True
