#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dataset_profiler.py — automatic inner-structure identification per dataset, at probe cost.

Design goal: with thousands of candidate datasets you must quickly identify the inner structure of
each one, which values belong in the graph, and how it combines with model internals.

Per source (input = metadata text + optionally a small sample/probe):
  (a) STRUCTURE: columns/dtypes/units from table samples (HTML/JSON), file-type census,
                      pairing relations (image<->pose, param<->outcome, time series, mesh<->photo, LR<->HR, ...)
  (b) IMAGING TYPE against the twin taxonomy (6 families, the same as the acquisition_scheduler category priors):
                      process-inversion / real-cad-pair / sensor-calibrated / perfect-gt-synthetic /
                      cross-modal / field-physics
  (c) GRAPH VALUES: decision-bearing fields -- ground-truth keys, closed-loop tuples, calibration anchors
  (d) MODEL-MATCH: rule table (MODEL_MATCH_RULES) against the model-reservoir substrate
                      (reports/probes/model_reservoir_protocol.json + sam2_cad_anchor_pilot.json)

Modes:
  --selftest             >=3 constructed cases (asserts)
  --retrodict            decisive number 1: reproduce a hand-written schema for one source
  --scale                decisive number 2: all DB sources, ms/dataset, type distribution, model match
  --probe N              N live probes (<2MB each); measure the confidence lift vs metadata alone
  --one NAME             profile one named DB source

Output per dataset: data/dataset_profiles/<id>.profile.json
The source DB ($DATA_SOURCES_DB, default ./data_sources.db) is opened READ-ONLY. It never writes to an evidence ledger.
"""
from __future__ import annotations

import argparse
import html as _html
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata

BASE = os.environ.get("DATASET_PROFILER_BASE", os.getcwd())
SOURCES_DB = os.environ.get("DATA_SOURCES_DB", os.path.join(BASE, "data_sources.db"))   # READ-ONLY
PROFILE_DIR = os.path.join(BASE, "data", "dataset_profiles")
IM_DIR = os.environ.get("RETRODICT_DIR", os.path.join(BASE, "data", "acquired", "retrodict"))
HAND_SCHEMA = os.environ.get("RETRODICT_SCHEMA", os.path.join(IM_DIR, "extracted_closed_loop_schema.json"))
REPORT_PATH = os.path.join(BASE, "reports", "probes", "dataset_profiler_v0.json")

PROBE_BYTE_CAP = 2 * 1024 * 1024      # hard rule: <2MB per source probe
PROBE_TIMEOUT_S = 20
PROBE_MAX_SOURCES = 15                # hard rule: at most 15 sources touched over the network

# ---------------------------------------------------------------- (a) STRUKTUR --

# File-type census: known dataset extensions grouped into modality classes.
EXT_CLASS = {
    "table":   {"csv", "tsv", "parquet", "feather", "xlsx", "xls"},
    "cad":     {"step", "stp", "iges", "igs", "dxf", "dwg", "brep", "fcstd", "sldprt"},
    "mesh":    {"stl", "obj", "ply", "off", "glb", "gltf", "usd", "usda", "usdz", "fbx", "dae", "3mf"},
    "image":   {"jpg", "jpeg", "png", "tif", "tiff", "bmp", "exr", "webp", "dng", "raw", "gif"},
    "video":   {"mp4", "avi", "mov", "mkv", "webm"},
    "pointcloud": {"pcd", "las", "laz", "e57", "xyz", "pts"},
    "array":   {"npz", "npy", "hdf5", "h5", "mat", "nc", "grib", "grib2", "zarr", "fits",
                "segy", "sgy", "nii"},
    "annot":   {"json", "xml", "yaml", "yml", "coco"},
    "logbag":  {"bag", "db3", "mcap", "ulog", "tlog"},
    "archive": {"rar", "zip", "tar", "gz", "7z", "tgz"},
    "audio":   {"wav", "flac", "mp3"},
    "doc":     {"md", "txt", "pdf"},
}
_ALL_EXTS = sorted({e for s in EXT_CLASS.values() for e in s}, key=len, reverse=True)
_EXT_RE = re.compile(r"\b[\w\-./]{0,64}\.(" + "|".join(_ALL_EXTS) + r")\b", re.IGNORECASE)

# Physical unit tokens (for column typing and field-physics evidence).
UNIT_TOKENS = [
    "°c", "degc", "ºc", "celsius", "kelvin", "mm2", "mm²", "cm2", "mm", "cm", "µm", "um", "nm",
    "bar", "kpa", "mpa", "gpa", "pa", "hz", "khz", "ghz", "px", "pixel", "m/s", "mm/s", "rad",
    "deg", "n·m", "nm torque", "newton", "kg", "g/", "gram", "kwh", "joule", " watt", " ms ",
    " s ", "sec", "volt", " amp", "db ", "lux", "kelvin",
]

# Pairing relations: (name, requires ALL groups) -- each group = at least one regex matches.
PAIRING_RULES = [
    ("image<->pose", [r"\b(image|photo|rgb|frame|picture|camera)\w*",
                      r"\b(pose|extrinsic|6\s*-?do?f|trajectory|odometry|groundtruth pose|camera pose)"]),
    ("param<->outcome", [r"\b(process parameter|parameter configuration|doe\b|design of experiments|"
                         r"process condition|machining parameter|print(ing)? parameter)",
                         r"\b(defect|outcome|quality|viability|label|wear|porosity|tensile|failure)"]),
    ("time-series", [r"\b(\d+\s*k?hz|time[- ]?series|imu\b|sampling rate|timestamps?|"
                     r"accelerometer|vibration|acoustic emission|waveform)"]),
    ("mesh<->photo", [r"\b(cad|mesh|\bstep\b|\bstl\b|scan2cad|3d model|geometry|shape)",
                      r"\b(image|photo|rgb|picture|video|scan(ned)?)"]),
    ("lr<->hr", [r"\b(low[- ]?res|lr[-/ ]hr|super[- ]?resolution|\bsr pair|zoom pair|upscal)"]),
    ("depth-gt", [r"\b(depth map|depth ground|rgb-?d\b|depth gt|kinect|tof camera|"
                  r"depth annotation|metric depth)"]),
    ("segmentation-gt", [r"\b(segmentation|instance mask|coco[- ]format|pixel[- ]level|"
                         r"semantic label|annotation.{0,20}mask|mask.{0,20}annotation)"]),
    ("calib-anchor", [r"\b(intrinsic|extrinsic|calibrat|camera matrix|distortion|rig\b|"
                      r"sensor model|boresight)"]),
    ("field-array", [r"\b(velocity field|pressure field|temperature field|flow field|wavefield|"
                     r"radar reflectivity|reanalysis|gridded|voxel grid|volumetric)"]),
    ("material-response", [r"\b(brdf|btf\b|svbrdf|reflectance|albedo|roughness map|spectral|"
                           r"gonio|scattering)"]),
    ("articulation", [r"\b(articulat|joint (angle|state|limit)|kinematic|urdf|part mobility|"
                      r"degrees? of freedom)"]),
    ("caption<->visual", [r"\b(caption|transcript|asr\b|narration|text annotation|"
                          r"language instruction)"]),
]
_PAIRING_COMPILED = [(n, [re.compile(p, re.IGNORECASE) for p in groups]) for n, groups in PAIRING_RULES]

# ------------------------------------------------- (b) AVBILDNINGS-TYP (taxonomi) --

FAMILIES = ["process-inversion", "real-cad-pair", "sensor-calibrated",
            "perfect-gt-synthetic", "cross-modal", "field-physics"]

# DB-kategori → (familj, bas-konfidens). Samma kategorinamn som acquisition_scheduler
# CATEGORY_PRIOR; the base confidence is a PRIOR (documented, CONSTRUCTED); the text evidence
# below is what is measured. Categories with an ambiguous family get a lower base.
CATEGORY_FAMILY = {
    "fabrication-twin":        ("process-inversion", 0.55),
    "industrial-inspection":   ("process-inversion", 0.40),
    "cad-scale-gt":            ("real-cad-pair", 0.55),
    "sim2real-twin":           ("real-cad-pair", 0.45),
    "cad-assembly":            ("real-cad-pair", 0.45),
    "cad-brep":                ("real-cad-pair", 0.45),
    "3d-scene-scanned":        ("real-cad-pair", 0.40),
    "heritage-3d":             ("real-cad-pair", 0.40),
    "real-estate-floorplan":   ("real-cad-pair", 0.35),
    "articulated-kinematic":   ("real-cad-pair", 0.35),
    "sensor-calibrated":       ("sensor-calibrated", 0.55),
    "niche-sensor":            ("sensor-calibrated", 0.45),
    "geo-registered":          ("sensor-calibrated", 0.40),
    "egocentric":              ("sensor-calibrated", 0.35),
    "sr-realpair":             ("sensor-calibrated", 0.40),
    "satellite-sr":            ("sensor-calibrated", 0.40),
    "microscopy":              ("sensor-calibrated", 0.35),
    "material-brdf":           ("sensor-calibrated", 0.40),
    "textile-material":        ("sensor-calibrated", 0.30),
    "robot-manipulation":      ("sensor-calibrated", 0.35),
    "assembly-manipulation":   ("sensor-calibrated", 0.30),
    "perfect-gt-synthetic":    ("perfect-gt-synthetic", 0.60),
    "impossible-gt-synthetic": ("perfect-gt-synthetic", 0.60),
    "cross-modal":             ("cross-modal", 0.55),
    "video-corpus":            ("cross-modal", 0.45),
    "public-domain-video":     ("cross-modal", 0.40),
    "sport-broadcast":         ("cross-modal", 0.35),
    "patent-drawing":          ("cross-modal", 0.35),
    "structured-commons":      ("cross-modal", 0.35),
    "measured-physics":        ("field-physics", 0.50),
    "weather-cloud":           ("field-physics", 0.55),
    "atmospheric-physics":     ("field-physics", 0.55),
    "seismic":                 ("field-physics", 0.55),
    "fluid-physics":           ("field-physics", 0.55),
    "potential-field-bathymetry": ("field-physics", 0.50),
    "insar-deformation":       ("field-physics", 0.50),
    "gpr-subsurface":          ("field-physics", 0.50),
    "digital-rock":            ("field-physics", 0.45),
    "well-log-borehole":       ("field-physics", 0.45),
    "astronomy":               ("field-physics", 0.45),
    "underwater-marine":       ("field-physics", 0.35),
    "agri-forestry-construction": ("field-physics", 0.30),
}
DEFAULT_FAMILY_BASE = 0.25   # unknown category -> weak prior on cross-modal

# Text evidence per family: regexes; each hit adds EVIDENCE_STEP up to EVIDENCE_CAP.
FAMILY_EVIDENCE = {
    "process-inversion": [
        r"\bprocess parameter", r"\bdefect", r"\bdoe\b|design of experiments", r"injection mold",
        r"\bmachining|milling|turning|drilling", r"tool wear", r"in[- ]situ", r"layer[- ]wise",
        r"additive manufactur|3d print|\bwaam\b|powder bed", r"weld", r"process[–-]quality",
        r"closed[- ]loop", r"viability", r"cycle time", r"energy consumption",
    ],
    "real-cad-pair": [
        r"\bcad\b", r"scan2cad", r"\bstep\b|\bstl\b|b-?rep", r"3d scan", r"reference (geometry|model)",
        r"real.{0,25}(cad|model|mesh)|(cad|model|mesh).{0,25}real", r"alignment", r"scale ground truth",
        r"photogrammetr", r"multi[- ]view.{0,30}(object|reconstruction)", r"as[- ]built",
    ],
    "sensor-calibrated": [
        r"intrinsic", r"extrinsic", r"calibrat", r"\bimu\b", r"lidar", r"rolling shutter",
        r"camera parameters", r"\brig\b", r"vicon|mocap|motion capture", r"ground[- ]?truth trajectory",
        r"time[- ]synchron", r"exposure|iso\b|raw sensor",
    ],
    "perfect-gt-synthetic": [
        r"synthetic", r"render(ed|ing)", r"blender|unreal|unity|omniverse|isaac",
        r"simulat(ed|ion).{0,30}(ground truth|gt\b|dataset)", r"ray[- ]?trac", r"procedural",
        r"perfect (ground truth|labels)", r"domain randomi",
    ],
    "cross-modal": [
        r"caption", r"\basr\b|transcript", r"text[- ](annotation|description|pair)",
        r"language", r"audio[- ]visual", r"narration", r"instructional video", r"subtitles",
    ],
    "field-physics": [
        r"weather|cloud|precipitation|atmospher", r"seismic|earthquake|waveform",
        r"velocity field|pressure field|temperature field|flow field", r"reanalysis|era5",
        r"radar|sonar|bathymetr", r"\bpde\b|navier|turbulen", r"insar|deformation",
        r"borehole|well log|subsurface", r"satellite|remote sensing", r"magnet(ic|ometer)|gravity",
        r"\bctd\b|oceanograph", r"micro-?ct|tomograph",
    ],
}
_FAMILY_EVIDENCE_C = {f: [re.compile(p, re.IGNORECASE) for p in ps] for f, ps in FAMILY_EVIDENCE.items()}
EVIDENCE_STEP = 0.05
EVIDENCE_CAP = 0.35
# Structural bonuses: pairing -> family (+bonus) -- structure outweighs keywords.
PAIRING_FAMILY_BONUS = {
    "param<->outcome": ("process-inversion", 0.15),
    "mesh<->photo": ("real-cad-pair", 0.10),
    "calib-anchor": ("sensor-calibrated", 0.12),
    "field-array": ("field-physics", 0.12),
    "caption<->visual": ("cross-modal", 0.10),
}
CONF_CAP = 0.95

# ------------------------------------------------------------- (d) MODEL-MATCH --
# Regeltabell mot model-reservoir-substratet:
# reports/probes/model_reservoir_protocol.json (SAM2.1 / DINOv3 / VLM / SAM2-small ×-anchor-rader)
# reports/probes/sam2_cad_anchor_pilot.json (SAM2×CAD-GT-mask-ankaret, del-prior-fyndet)
# mode: CALIBRATE = datasetets GT kan kalibrera modellens prior (TPR/FPR→LLR-vikt);
# CONSUME = the model/twin substrate can consume the dataset as a corpus/anchor source.
# trigger: predicate over (pairings, ext classes, text).
MODEL_MATCH_RULES = [
    {"model": "SAM2.1 (hiera-large, on-disk)", "mode": "CALIBRATE",
     "rule": "segmenterbara objekt + segmentations-GT ELLER GT-mesh → mask-IoU-ankare (sam2_cad_anchor_pilot-protokollet)",
     "pred": lambda P, E, T: ("segmentation-gt" in P) or ("mesh<->photo" in P and ("mesh" in E or "cad" in E))},
    {"model": "mono-depth-prior", "mode": "CALIBRATE",
     "rule": "depth-GT / RGB-D / metrisk depth → depth-belief-kalibrering",
     "pred": lambda P, E, T: "depth-gt" in P},
    {"model": "material-leg (BRDF-prior)", "mode": "CALIBRATE",
     "rule": "BRDF/reflectance/spektral GT → material-belief-kalibrering",
     "pred": lambda P, E, T: "material-response" in P},
    {"model": "camera-state-prior (camera_state_v0)", "mode": "CALIBRATE",
     "rule": "intrinsics/extrinsics/IMU/pose-GT → kamera/pose-kalibrering",
     "pred": lambda P, E, T: ("calib-anchor" in P) or ("image<->pose" in P)},
    {"model": "upscaler/SR-cert (certified_superres)", "mode": "CALIBRATE",
     "rule": "genuine LR/HR or zoom pairs -> super-resolution belief anchor",
     "pred": lambda P, E, T: "lr<->hr" in P},
    {"model": "VLM-cross-check", "mode": "CALIBRATE",
     "rule": "human captions/text paired with visual data -> caption-belief calibration (cross-channel, not primary)",
     "pred": lambda P, E, T: "caption<->visual" in P},
    {"model": "DINOv3-retrieval (ViT-S)", "mode": "CONSUME",
     "rule": "bildkorpus → embedding-NN-retrieval / implicit del/material-segmentering",
     "pred": lambda P, E, T: ("image" in E) or ("video" in E) or bool(re.search(r"\bimage|photo|picture|video", T, re.I))},
    {"model": "fabrication-twin-surrogat (process-inversion)", "mode": "CONSUME",
     "rule": "parameter/outcome closed-loop tuple -> inverse process model / twin training",
     "pred": lambda P, E, T: "param<->outcome" in P},
    {"model": "physics-sim-residual (field-solver priors)", "mode": "CONSUME",
     "rule": "measured field arrays/time series with physical units -> sim-residual calibration",
     "pred": lambda P, E, T: ("field-array" in P) or ("time-series" in P and "array" in E)},
    {"model": "articulation-prior (kinematik-leg)", "mode": "CONSUME",
     "rule": "leder/artikulations-annotationer → kinematik-prior-konsumtion",
     "pred": lambda P, E, T: "articulation" in P},
]


# ====================================================================== core ====

def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"(?s)<(script|style)[^>]*>.*?</\1>", " ",
                                                     re.sub(r"(?s)<[^>]+>", " ", text))))


def extract_html_table_columns(raw_html: str):
    """Extract column/description/unit from HTML tables whose header mentions Column/Parameter."""
    cols = []
    for tbl in re.findall(r"(?is)<table[^>]*>(.*?)</table>", raw_html or ""):
        rows = re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", tbl)
        if not rows:
            continue
        header = _strip_html(rows[0]).lower()
        if not re.search(r"\b(column|parameter|variable|field)\b", header):
            continue
        for row in rows[1:]:
            cells = [_strip_html(c).strip() for c in re.findall(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>", row)]
            cells = [c for c in cells if c]
            if not cells:
                continue
            name = cells[0]
            if len(name) > 40 or not re.match(r"^[\w .%°ºµ/()\[\]-]+$", name):
                continue
            entry = {"col": name}
            if len(cells) >= 2:
                entry["desc"] = cells[1][:120]
            if len(cells) >= 3:
                entry["unit"] = cells[-1][:40]
            cols.append(entry)
    # dedupe on the EXACT column name (case-sensitive: Tinj and tinj are different columns)
    seen, out = set(), []
    for c in cols:
        if c["col"] not in seen:
            seen.add(c["col"])
            out.append(c)
    return out


def extract_structure(text: str, raw_html: str = ""):
    """(a) STRUKTUR: filtyps-census, kolumner+enheter, parnings-relationer."""
    census_ext = {}
    for m in _EXT_RE.finditer(text):
        ext = m.group(1).lower()
        census_ext[ext] = census_ext.get(ext, 0) + 1
    census_class = {}
    for cls, exts in EXT_CLASS.items():
        n = sum(census_ext.get(e, 0) for e in exts)
        if n:
            census_class[cls] = n
    columns = extract_html_table_columns(raw_html or text)
    low = " " + text.lower() + " "
    units = sorted({u.strip() for u in UNIT_TOKENS if u in low})
    pairings = []
    for name, groups in _PAIRING_COMPILED:
        if all(any(g.search(text) for _ in [0]) and g.search(text) for g in groups):
            pairings.append(name)
    return {"file_census_ext": census_ext, "file_census_class": census_class,
            "columns": columns, "units_seen": units, "pairings": pairings}


def classify_family(category: str, text: str, structure: dict):
    """(b) AVBILDNINGS-TYP: kategori-prior + text-evidens + struktur-bonus → familj+konfidens."""
    scores = {f: 0.0 for f in FAMILIES}
    fam0, base = CATEGORY_FAMILY.get((category or "").strip(), (None, 0.0))
    if fam0:
        scores[fam0] += base
    else:
        scores["cross-modal"] += DEFAULT_FAMILY_BASE
    evidence_hits = {}
    for fam, regs in _FAMILY_EVIDENCE_C.items():
        hits = [r.pattern for r in regs if r.search(text)]
        evidence_hits[fam] = len(hits)
        scores[fam] += min(EVIDENCE_CAP, EVIDENCE_STEP * len(hits))
    for pairing in structure["pairings"]:
        fb = PAIRING_FAMILY_BONUS.get(pairing)
        if fb:
            scores[fb[0]] += fb[1]
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top, second = ranked[0], ranked[1]
    conf = min(CONF_CAP, top[1])
    return {"family": top[0], "confidence": round(conf, 3),
            "margin_over_second": round(top[1] - second[1], 3),
            "runner_up": second[0], "scores": {k: round(v, 3) for k, v in scores.items()},
            "category_prior_used": {"category": category, "family": fam0, "base": base},
            "evidence_hits": evidence_hits}


# Which unit patterns type a column as a parameter vs an outcome (for closed-loop tuples).
_PARAM_UNIT = re.compile(r"(°c|ºc|degc|bar|kpa|mpa|\bs\b|sec|mm/s|rpm|hz|%|volt|amp)", re.I)
_OUTCOME_HINT = re.compile(r"(mm2|mm²|area|label|viability|defect|wear|class|score|weight|energy|cycle)", re.I)
_ID_HINT = re.compile(r"(\bid\b|identifier|number|batch|index)", re.I)


def extract_graph_values(structure: dict, family: str, text: str):
    """(c) GRAPH VALUES: the decision-bearing fields, i.e. what goes into the graph."""
    gv = {"gt_keys": [], "closed_loop_tuple": None, "calibration_anchors": [], "notes": []}
    params, outcomes, ids = [], [], []
    for c in structure["columns"]:
        blob = " ".join(str(c.get(k, "")) for k in ("col", "desc", "unit"))
        if _ID_HINT.search(c["col"]) or _ID_HINT.search(c.get("desc", "")):
            ids.append(c["col"])
        elif _OUTCOME_HINT.search(blob):
            outcomes.append(c["col"])
        elif _PARAM_UNIT.search(c.get("unit", "")) or re.search(r"(pressure|temperature|time|speed|feed)",
                                                                c.get("desc", ""), re.I):
            params.append(c["col"])
    if params and outcomes:
        gv["closed_loop_tuple"] = {"param_vector": params, "outcome_vector": outcomes, "id_keys": ids}
    P = structure["pairings"]
    if "segmentation-gt" in P:
        gv["gt_keys"].append("instance-segmentation-masks")
    if "depth-gt" in P:
        gv["gt_keys"].append("depth-gt")
    if "image<->pose" in P:
        gv["gt_keys"].append("pose-gt")
    if "material-response" in P:
        gv["gt_keys"].append("material-brdf-gt")
    if "lr<->hr" in P:
        gv["gt_keys"].append("lr-hr-pairs")
    if "articulation" in P:
        gv["gt_keys"].append("articulation-gt")
    if "cad" in structure["file_census_class"] or "mesh" in structure["file_census_class"]:
        gv["calibration_anchors"].append("geometry-file(CAD/mesh)")
    if "calib-anchor" in P:
        gv["calibration_anchors"].append("sensor-calibration(intrinsics/extrinsics)")
    if "caption<->visual" in P:
        gv["gt_keys"].append("human-caption-anchor")
    if "field-array" in P:
        gv["gt_keys"].append("measured-field-arrays")
    if "time-series" in P:
        gv["gt_keys"].append("timestamped-series")
    if family == "perfect-gt-synthetic":
        gv["notes"].append("synthetic ground truth is perfect by construction; its anchor value is model deviation, not truth about the world")
    return gv


def match_models(structure: dict, text: str):
    """(d) MODEL-MATCH: regeltabellen mot struktur+text."""
    P, E = set(structure["pairings"]), set(structure["file_census_class"])
    out = []
    for r in MODEL_MATCH_RULES:
        try:
            if r["pred"](P, E, text):
                out.append({"model": r["model"], "mode": r["mode"], "rule": r["rule"]})
        except Exception:
            pass
    return out


def slugify(name: str) -> str:
    """Injective id key: readable slug + 6-hex sha1 of the EXACT name (measured collisions:
    ScanNet vs ScanNet++ and Mid-Air vs MID-Air collide without the suffix; at thousands of sources
    a silent profile overwrite poisons the aggregate)."""
    import hashlib
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    base = re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-") or "unnamed"
    return base + "-" + hashlib.sha1(name.encode()).hexdigest()[:6]


def profile_dataset(name: str, category: str, text: str, raw_html: str = "",
                    probe_level: str = "db-metadata-only", band=None):
    """The full profile for ONE dataset. text = plain metadata text; raw_html = optional HTML sample.
    band = callable(profile) -> profile applied before serving (three-band admission + value
    masking). When None, the module `dataset_band_port` must provide apply_band; the call is
    fail-closed, so no profile is served without banding."""
    structure = extract_structure(text, raw_html)
    mapping = classify_family(category, text, structure)
    graph_values = extract_graph_values(structure, mapping["family"], text)
    models = match_models(structure, text)
    profile = {"id": slugify(name), "name": name, "probe_level": probe_level,
               "struktur": structure, "avbildningstyp": mapping,
               "graf_varden": graph_values, "model_matcher": models}
    # INTERPOSITION: the band port (AUTO_ADMIT/QUARANTINE/REJECT)
    # every served profile is banded + value-masked (never-clean, REJECT included) HERE. Lazy
    # import (the band port imports this module). Fail-CLOSED: if the port cannot be loaded NO
    # profile is served (a stop is better than unmasked/unbanded serving). Provide a module
    # `dataset_band_port` exposing apply_band(profile) -> profile.
    if band is None:
        from dataset_band_port import apply_band as band   # noqa: F811  (fail-closed if absent)
    return band(profile)


# ==================================================================== DB & IO ====

def load_sources():
    db = sqlite3.connect(f"file:{SOURCES_DB}?mode=ro", uri=True)   # READ-ONLY, hard rule
    cur = db.execute("select name, category, provider, scale, distributed, license, "
                     "license_class, sensor_metadata, relevance, url, notes from sources")
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur]
    db.close()
    return rows


def row_text(r: dict) -> str:
    return " | ".join(str(r.get(k) or "") for k in
                      ("name", "category", "provider", "scale", "distributed",
                       "sensor_metadata", "relevance", "notes", "url"))


def write_profile(profile: dict):
    os.makedirs(PROFILE_DIR, exist_ok=True)
    path = os.path.join(PROFILE_DIR, profile["id"] + ".profile.json")
    with open(path, "w") as f:
        json.dump(profile, f, ensure_ascii=False, indent=1)
    return path


# ================================================================= retrodiktion ====

def run_retrodict():
    """Decisive number 1: profile the reference metadata and compare it field by field against a
    hand-extracted schema (paths come from $RETRODICT_DIR / $RETRODICT_SCHEMA)."""
    hand = json.load(open(HAND_SCHEMA))
    texts, htmls = [], []
    for fn in ("zenodo_20322729_metadata_full.json", "zenodo_20309380_companion_metadata_full.json"):
        d = json.load(open(os.path.join(IM_DIR, fn)))
        desc = d.get("metadata", {}).get("description", "")
        htmls.append(desc)
        texts.append(_strip_html(json.dumps(d.get("metadata", {}), ensure_ascii=False)))
    text, raw_html = " ".join(texts), " ".join(htmls)
    prof = profile_dataset("Zenodo Injection-Molding Defect", "fabrication-twin", text, raw_html,
                           probe_level="acquired-metadata-sample")
    write_profile(prof)

    got_cols = {c["col"] for c in prof["struktur"]["columns"]}
    hand_cols = {c["col"] for c in hand["csv_columns"]}
    col_overlap = sorted(got_cols & hand_cols)
    hand_units = {c["col"]: c["unit"] for c in hand["csv_columns"]}
    got_units = {c["col"]: c.get("unit", "") for c in prof["struktur"]["columns"]}
    unit_match = [c for c in col_overlap
                  if hand_units[c].lower().replace("°", "deg").replace("²", "2") in
                  got_units.get(c, "").lower().replace("°", "deg").replace("º", "deg").replace("²", "2")
                  or got_units.get(c, "").lower() in ("int",) and hand_units[c] == "int"]

    clt = prof["graf_varden"]["closed_loop_tuple"] or {"param_vector": [], "outcome_vector": []}
    hand_params = set(hand["closed_loop_tuple_map"]["param_vector"])
    hand_outcomes = set(hand["closed_loop_tuple_map"]["outcome_vector"]) - {"viability_label"}
    param_hit = sorted(set(clt["param_vector"]) & hand_params)
    outcome_hit = sorted(set(clt["outcome_vector"]) & hand_outcomes)
    hand_files = {"csv", "step", "rar", "zip"}
    file_hit = sorted(hand_files & set(prof["struktur"]["file_census_ext"]))
    fields_total = len(hand_cols) + len(hand_params) + len(hand_outcomes) + len(hand_files)
    fields_hit = len(col_overlap) + len(param_hit) + len(outcome_hit) + len(file_hit)
    return {
        "hand_schema": HAND_SCHEMA,
        "csv_columns": {"hand": len(hand_cols), "profiler": len(got_cols),
                        "overlap": len(col_overlap), "overlap_frac": round(len(col_overlap) / len(hand_cols), 4),
                        "missing": sorted(hand_cols - got_cols), "extra": sorted(got_cols - hand_cols)},
        "units_matched_on_overlap": {"n": len(unit_match), "of": len(col_overlap)},
        "param_vector": {"hand": len(hand_params), "hit": len(param_hit), "missing": sorted(hand_params - set(param_hit))},
        "outcome_vector": {"hand": len(hand_outcomes), "hit": len(outcome_hit), "missing": sorted(hand_outcomes - set(outcome_hit))},
        "file_types": {"hand": sorted(hand_files), "hit": file_hit},
        "family": prof["avbildningstyp"]["family"],
        "family_expected": "process-inversion",
        "closed_loop_tuple_detected": prof["graf_varden"]["closed_loop_tuple"] is not None,
        "overall_field_overlap": round(fields_hit / fields_total, 4),
        "fields_hit": fields_hit, "fields_total": fields_total,
        "profile_path": os.path.join(PROFILE_DIR, prof["id"] + ".profile.json"),
    }


# ==================================================================== skaltest ====

def run_scale(write_profiles=True):
    """Decisive number 2: all DB sources, text only (no network), ms/dataset + distributions."""
    rows = load_sources()
    t0 = time.perf_counter()
    profiles = []
    for r in rows:
        profiles.append(profile_dataset(r["name"], r["category"], row_text(r),
                                        probe_level="db-metadata-only"))
    elapsed = time.perf_counter() - t0
    ms_per = elapsed * 1000.0 / max(1, len(rows))
    if write_profiles:
        for p in profiles:
            write_profile(p)
    fam_dist, conf_by_fam, model_counts, pairing_counts = {}, {}, {}, {}
    for p in profiles:
        f = p["avbildningstyp"]["family"]
        fam_dist[f] = fam_dist.get(f, 0) + 1
        conf_by_fam.setdefault(f, []).append(p["avbildningstyp"]["confidence"])
        for m in p["model_matcher"]:
            key = f'{m["model"]} [{m["mode"]}]'
            model_counts[key] = model_counts.get(key, 0) + 1
        for pr in p["struktur"]["pairings"]:
            pairing_counts[pr] = pairing_counts.get(pr, 0) + 1
    return {
        "n_sources": len(rows),
        "elapsed_s": round(elapsed, 4),
        "ms_per_dataset": round(ms_per, 4),
        "extrapolated_10000_datasets_minutes": round(ms_per * 10000 / 60000.0, 3),
        "family_distribution": dict(sorted(fam_dist.items(), key=lambda kv: -kv[1])),
        "mean_confidence_by_family": {f: round(sum(v) / len(v), 3) for f, v in conf_by_fam.items()},
        "model_match_counts": dict(sorted(model_counts.items(), key=lambda kv: -kv[1])),
        "pairing_counts": dict(sorted(pairing_counts.items(), key=lambda kv: -kv[1])),
        "n_with_closed_loop_tuple": sum(1 for p in profiles if p["graf_varden"]["closed_loop_tuple"]),
        "profiles_written": len(profiles) if write_profiles else 0,
        "profile_dir": PROFILE_DIR,
    }, profiles


# =================================================================== live-prober ====

def _fetch(url: str):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "dataset-profiler/0.1"})
    with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S) as resp:
        return resp.read(PROBE_BYTE_CAP)


def probe_source(r: dict):
    """Fetch a small structure sample (<2MB): Zenodo API / GitHub API / raw page."""
    url = (r.get("url") or "").strip()
    try:
        m = re.search(r"zenodo\.org/(?:records?|record)/(\d+)", url)
        if m:
            raw = _fetch(f"https://zenodo.org/api/records/{m.group(1)}").decode("utf-8", "replace")
            d = json.loads(raw)
            files = " ".join(f.get("key", "") for f in d.get("files", []))
            desc = d.get("metadata", {}).get("description", "")
            return _strip_html(json.dumps(d.get("metadata", {}), ensure_ascii=False)) + " " + files, desc, len(raw)
        m = re.search(r"github\.com/([\w.-]+)/([\w.-]+)", url)
        if m:
            owner, repo = m.group(1), m.group(2).removesuffix(".git")
            raw = _fetch(f"https://api.github.com/repos/{owner}/{repo}/contents/").decode("utf-8", "replace")
            listing = " ".join(x.get("path", "") for x in json.loads(raw) if isinstance(x, dict))
            try:
                readme = _fetch(f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/README.md"
                                ).decode("utf-8", "replace")
            except Exception:
                readme = ""
            return listing + " " + readme, readme, len(raw) + len(readme)
        raw = _fetch(url)
        html_text = raw.decode("utf-8", "replace")
        return _strip_html(html_text), html_text, len(raw)
    except Exception as e:
        return None, None, f"ERROR: {type(e).__name__}: {e}"


def run_probes(n: int, profiles_by_name: dict):
    """5-10 live probes: measure the confidence lift (probe vs metadata alone)."""
    rows = load_sources()
    go = [r for r in rows if (r.get("license_class") or "").lower() in
          {"permissive", "cc0-pd", "cc-by", "open", "open-gov", "open-source", "odc-by"}
          and (r.get("url") or "").startswith("http")]
    # prioritise sources whose URL yields cheap file listings (zenodo/github), then the rest
    go.sort(key=lambda r: (0 if re.search(r"zenodo\.org/records?/\d+", r["url"] or "")
                           else 1 if "github.com/" in (r["url"] or "") else 2, r["name"]))
    picked, results, total_bytes = go[:min(n, PROBE_MAX_SOURCES)], [], 0
    for r in picked:
        base_prof = profile_dataset(r["name"], r["category"], row_text(r),
                                    probe_level="db-metadata-only")
        text, raw_html, nbytes = probe_source(r)
        if text is None:
            results.append({"name": r["name"], "url": r["url"], "error": nbytes})
            continue
        total_bytes += nbytes
        probed = profile_dataset(r["name"], r["category"], row_text(r) + " " + text,
                                 raw_html or "", probe_level="db+live-probe")
        write_profile(probed)
        results.append({
            "name": r["name"], "url": r["url"], "probe_bytes": nbytes,
            "conf_db_only": base_prof["avbildningstyp"]["confidence"],
            "conf_with_probe": probed["avbildningstyp"]["confidence"],
            "conf_lift": round(probed["avbildningstyp"]["confidence"]
                               - base_prof["avbildningstyp"]["confidence"], 3),
            "family_db_only": base_prof["avbildningstyp"]["family"],
            "family_with_probe": probed["avbildningstyp"]["family"],
            "new_pairings": sorted(set(probed["struktur"]["pairings"])
                                   - set(base_prof["struktur"]["pairings"])),
            "new_file_classes": sorted(set(probed["struktur"]["file_census_class"])
                                       - set(base_prof["struktur"]["file_census_class"])),
            "n_columns_found": len(probed["struktur"]["columns"]),
            "new_model_matches": sorted({m["model"] for m in probed["model_matcher"]}
                                        - {m["model"] for m in base_prof["model_matcher"]}),
        })
    ok = [x for x in results if "conf_lift" in x]
    return {"n_probed": len(picked), "n_ok": len(ok), "total_probe_bytes": total_bytes,
            "byte_cap_per_source": PROBE_BYTE_CAP,
            "mean_conf_lift": round(sum(x["conf_lift"] for x in ok) / len(ok), 4) if ok else None,
            "results": results}


# ===================================================================== selftest ====

SELFTEST_CASES = [
    {"name": "ST-injection", "category": "fabrication-twin",
     "text": ("Injection molding dataset. Process parameters varied via design of experiments (DoE). "
              "Quantitative_defects_and_process_parameters.csv, Mold_geometry.step, Images_dataset.rar. "
              "Pixel-level segmentation masks, COCO-format annotations. Defect areas in mm2. "
              "Process viability label per part. 6000x4000 px images of injected parts."),
     "html": ("<table><tr><th>Column</th><th>Description</th><th>Unit</th></tr>"
              "<tr><td>Tinj</td><td>Injection temperature</td><td>degC</td></tr>"
              "<tr><td>Flash</td><td>defect area</td><td>mm2</td></tr></table>"),
     "expect_family": "process-inversion",
     "expect_pairings": ["param<->outcome", "segmentation-gt"],
     "expect_model": "SAM2.1 (hiera-large, on-disk)",
     "expect_closed_loop": True},
    {"name": "ST-visualinertial", "category": "sensor-calibrated",
     "text": ("Visual-inertial dataset: stereo camera images at 20 Hz, IMU at 200 Hz, "
              "camera intrinsics and extrinsics from calibration, ground-truth trajectory from Vicon "
              "motion capture. data.bag rosbag files with timestamps. Ground truth pose per frame."),
     "html": "", "expect_family": "sensor-calibrated",
     "expect_pairings": ["calib-anchor", "image<->pose", "time-series"],
     "expect_model": "camera-state-prior (camera_state_v0)", "expect_closed_loop": False},
    {"name": "ST-synthdepth", "category": "perfect-gt-synthetic",
     "text": ("Synthetic dataset rendered in Blender with ray-traced global illumination: RGB images "
              "with perfect ground truth depth map per frame, instance segmentation masks, "
              "camera pose. scene_0001.png, depth_0001.exr, masks.json."),
     "html": "", "expect_family": "perfect-gt-synthetic",
     "expect_pairings": ["depth-gt", "segmentation-gt"],
     "expect_model": "mono-depth-prior", "expect_closed_loop": False},
    {"name": "ST-brdf", "category": "material-brdf",
     "text": ("Measured BRDF database: gonio-reflectometer measurements of isotropic materials, "
              "spectral reflectance tables, roughness map fits. brdf_material_042.mat files."),
     "html": "", "expect_family": "sensor-calibrated",
     "expect_pairings": ["material-response"],
     "expect_model": "material-leg (BRDF-prior)", "expect_closed_loop": False},
]


def run_selftest_deep():
    """Selftest for the L2/L3 analysers -- constructed payloads, no network calls."""
    import io, struct, tempfile
    checks = {}
    # PNG-analysator
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (8, 6)).save(buf, "PNG")
        info = analyze_image_bytes(buf.getvalue())
        checks["png"] = (info["w"], info["h"], info["channels"]) == (8, 6, 3)
    except ImportError:
        checks["png"] = analyze_image_bytes(
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", 8, 6) + bytes([8, 2]) + b"\x00" * 8
        )["w"] == 8
    # CSV: numerik + null + klassbalans
    csvb = b"T,label,note\n200,good,\n210,bad,x\n205,good,\n,good,\n"
    t = analyze_csv_bytes(csvb)
    checks["csv_numeric"] = t["numeric_cols"]["T"]["median"] == 205.0 and t["numeric_cols"]["T"]["min"] == 200.0
    checks["csv_null"] = t["null_fraction"]["T"] == 0.25
    checks["csv_classes"] = t["class_balance"]["label"]["good"] == 0.75
    # LAS
    lasb = (b"~Version\nVERS. 2.0:\n~Well\nNULL. -999.25:\n~Curve\nDEPT.m : depth\nGR.gAPI : gamma\n"
            b"~ASCII\n100.0 50.0\n101.0 -999.25\n102.0 70.0\n")
    la = analyze_las_bytes(lasb)
    checks["las"] = (len(la["curves"]) == 2 and la["null_fraction"]["GR"] == 0.3333
                     and la["numeric_cols"]["GR"]["max"] == 70.0)
    # PLY: sluten tetraeder = watertight; en yta borttagen = ej
    tet = ("ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\n"
           "property float z\nelement face {n}\nproperty list uchar int vertex_indices\nend_header\n"
           "0 0 0\n1 0 0\n0 1 0\n0 0 1\n")
    f_closed = "3 0 1 2\n3 0 3 1\n3 1 3 2\n3 0 2 3\n"
    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, "t.ply")
        open(p1, "w").write(tet.format(n=4) + f_closed)
        p2 = os.path.join(td, "o.ply")
        open(p2, "w").write(tet.format(n=3) + "\n".join(f_closed.splitlines()[:3]) + "\n")
        m1, m2 = analyze_mesh_file(p1), analyze_mesh_file(p2)
        checks["ply_watertight"] = m1["watertight"] is True and m1["n_faces"] == 4
        checks["ply_open"] = m2["watertight"] is False and m2["boundary_edges"] == 3
    # TFRecord framing + key scan + embedded PNG
    from PIL import Image as _I
    b2 = io.BytesIO()
    _I.new("L", (4, 4)).save(b2, "PNG")
    payload = b"video" + b2.getvalue() + b"segmentations"
    rec = struct.pack("<Q", len(payload)) + b"\x00" * 4 + payload + b"\x00" * 4
    tf = parse_tfrecords(rec * 2, key_probes=("video", "segmentations", "depth"))
    checks["tfrecord"] = (tf["n_complete_records"] == 2 and tf["keys_seen"] == ["segmentations", "video"]
                          and len(tf["images"]) == 2 and tf["images"][0]["w"] == 4)
    # goal-relative decisions: a BRDF source is relevant for one goal and not another; a segmentable CAD source is the opposite
    _id_band = (lambda pr: pr)
    pA = profile_dataset("stA", "material-brdf", "measured brdf reflectance roughness map gonio spectral",
                         band=_id_band)
    pB = profile_dataset("stB", "cad-scale-gt",
                         "cad mesh model.ply real photo rgb images instance segmentation mask annotation",
                         band=_id_band)
    checks["goal_A"] = goal_decision("A", pA)["relevant"] and not goal_decision("A", pB)["relevant"]
    checks["goal_B"] = goal_decision("B", pB)["relevant"] and not goal_decision("B", pA)["relevant"]
    return {"all_pass": all(checks.values()), "checks": checks}


def run_selftest():
    results, ok = [], True
    for c in SELFTEST_CASES:
 # the selftest exercises the PROFILING analysers; banding is a serving-time policy and is
 # bypassed with an identity band so the suite needs no admission port
        p = profile_dataset(c["name"], c["category"], c["text"], c["html"], probe_level="selftest",
                            band=lambda pr: pr)
        checks = {
            "family": (p["avbildningstyp"]["family"] == c["expect_family"],
                       p["avbildningstyp"]["family"]),
            "pairings": (all(x in p["struktur"]["pairings"] for x in c["expect_pairings"]),
                         p["struktur"]["pairings"]),
            "model": (c["expect_model"] in {m["model"] for m in p["model_matcher"]},
                      sorted(m["model"] for m in p["model_matcher"])),
            "closed_loop": ((p["graf_varden"]["closed_loop_tuple"] is not None) == c["expect_closed_loop"],
                            bool(p["graf_varden"]["closed_loop_tuple"])),
        }
        passed = all(v[0] for v in checks.values())
        ok &= passed
        results.append({"case": c["name"], "pass": passed,
                        "checks": {k: {"pass": v[0], "got": v[1]} for k, v in checks.items()}})
    deep = run_selftest_deep()
    ok &= deep["all_pass"]
    return {"all_pass": ok, "n_cases": len(SELFTEST_CASES), "cases": results,
            "deep_analyzers": deep}


# ================================================================ L2/L3 (deep) ====
# L0 = DB metadata (default). L1 = live probe <2MB (run_probes).
# L2 = payload sample: fetch a SMALL representative sample (10-50MB class), measure the actual
# distributions/resolutions/geometry + label sanity + ground-truth-promise verification.
# L3 = model-anchored: run a REAL model channel (a segmentation model from an offline cache) on the
# source images and measure the coupling strength directly (IoU vs GT mask / stability).
#
# DECLARED LIMIT: L2 on an arbitrary source needs a fetch plan (where the payload lives). The plan
# is per-source data in DEEP_SOURCES (6 curated sources across the families); sources without a plan
# get {"l2": "no-fetch-plan"} instead of a silent error.

SAMPLES_DIR = os.path.join(BASE, "data", "acquired", "samples")
DEEP_REPORT = os.path.join(BASE, "reports", "probes", "dataset_profiler_deep.json")
HF_HOME_DEFAULT = os.environ.get("HF_HOME", os.path.join(BASE, "hf_cache"))
L2_TOTAL_BYTE_CAP = 500 * 1024 * 1024          # hard rule: at most 500MB downloaded in total
DISK_FLOOR_BYTES = 12 * 1024**3                # root-disk-golv
_UA = {"User-Agent": "dataset-profiler/0.2"}


def _disk_ok():
    import shutil
    return shutil.disk_usage("/").free >= DISK_FLOOR_BYTES


def _http_read(url: str, max_bytes: int, rng: str = None, timeout: int = 90) -> bytes:
    import urllib.request
    h = dict(_UA)
    if rng:
        h["Range"] = rng
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(max_bytes)


class _RangeFile:
    """File-like object over HTTP Range -- lets zipfile read ONLY the central directory +
    valda medlemmar ur ett stort arkiv (payload-sampling utan full nedladdning).
    A simple readahead cache (512KB blocks) keeps the request count down."""
    BLOCK = 512 * 1024

    def __init__(self, url: str):
        import urllib.request
        self.url, self.pos, self.bytes_fetched = url, 0, 0
        req = urllib.request.Request(url, method="HEAD", headers=_UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            cl = r.headers.get("Content-Length")
        if cl is None:
            raise RuntimeError("no Content-Length; Range sampling impossible")
        self.length = int(cl)
        self._cache_start, self._cache = -1, b""

    def seekable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.length + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.length - self.pos
        n = min(n, self.length - self.pos)
        if n <= 0:
            return b""
        cs, ce = self._cache_start, self._cache_start + len(self._cache)
        if not (cs <= self.pos and self.pos + n <= ce):
            fetch_n = max(n, self.BLOCK)
            end = min(self.length - 1, self.pos + fetch_n - 1)
            self._cache = _http_read(self.url, fetch_n + 16,
                                     rng=f"bytes={self.pos}-{end}")
            self._cache_start = self.pos
            self.bytes_fetched += len(self._cache)
        off = self.pos - self._cache_start
        out = self._cache[off:off + n]
        self.pos += len(out)
        return out


def zip_sample(url: str, member_pred, dest_dir: str, max_members=40,
               per_member_cap=20 * 1024 * 1024):
    """Range-sample selected members from a remote zip. member_pred(name) -> bool.
    Returnerar (paths, bytes_fetched, n_members_total, namelist_sample)."""
    import zipfile
    rf = _RangeFile(url)
    zf = zipfile.ZipFile(rf)
    names = zf.namelist()
    picked = [n for n in names if member_pred(n)][:max_members]
    os.makedirs(dest_dir, exist_ok=True)
    paths = []
    for n in picked:
        info = zf.getinfo(n)
        if info.file_size > per_member_cap:
            continue
        data = zf.open(n).read()
        p = os.path.join(dest_dir, n.replace("/", "__"))
        with open(p, "wb") as f:
            f.write(data)
        paths.append(p)
    return paths, rf.bytes_fetched, len(names), names[:200]


def fetch_to(url: str, dest: str, max_bytes: int):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    data = _http_read(url, max_bytes)
    with open(dest, "wb") as f:
        f.write(data)
    return dest, len(data)


# ------------------------------------------------------------- payload-analysatorer --

def analyze_image_bytes(b: bytes):
    """PNG/JPEG header -> {format,w,h,channels,bit_depth}. Uses PIL when available, else a plain parse."""
    try:
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(b))
        ch = {"L": 1, "P": 1, "I;16": 1, "I": 1, "RGB": 3, "RGBA": 4, "LA": 2}.get(im.mode, len(im.getbands()))
        bd = 16 if im.mode in ("I;16", "I") else 8
        return {"format": (im.format or "?").lower(), "w": im.width, "h": im.height,
                "channels": ch, "bit_depth": bd, "mode": im.mode}
    except Exception:
        pass
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        import struct
        w, h = struct.unpack(">II", b[16:24])
        bit, ct = b[24], b[25]
        ch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(ct, 0)
        return {"format": "png", "w": w, "h": h, "channels": ch, "bit_depth": bit}
    if b[:3] == b"\xff\xd8\xff":
        i = 2
        while i < len(b) - 9:
            if b[i] != 0xFF:
                i += 1
                continue
            m = b[i + 1]
            if m in (0xC0, 0xC1, 0xC2):
                import struct
                h, w = struct.unpack(">HH", b[i + 5:i + 9])
                return {"format": "jpeg", "w": w, "h": h, "channels": b[i + 9], "bit_depth": b[i + 4]}
            seg = int.from_bytes(b[i + 2:i + 4], "big")
            i += 2 + seg
    return None


def _num_stats(vals):
    if not vals:
        return None
    vs = sorted(vals)
    n = len(vs)
    mean = sum(vs) / n
    var = sum((v - mean) ** 2 for v in vs) / n
    return {"n": n, "min": round(vs[0], 6), "max": round(vs[-1], 6),
            "median": round(vs[n // 2], 6), "mean": round(mean, 6), "var": round(var, 6)}


def analyze_csv_bytes(b: bytes, max_rows=20000, null_tokens=("", "na", "nan", "null", "none")):
    """Tabular payload: numeric column distributions, null fraction, class balance (low cardinality)."""
    import csv, io
    text = b.decode("utf-8", "replace")
    delim = ";" if text[:2000].count(";") > text[:2000].count(",") else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))[:max_rows + 1]
    if len(rows) < 2:
        return None
    header, data = rows[0], rows[1:]
    ncol = len(header)
    out = {"n_rows_sampled": len(data), "n_cols": ncol, "delimiter": delim,
           "numeric_cols": {}, "null_fraction": {}, "class_balance": {}}
    for j, col in enumerate(header):
        vals = [r[j].strip() if j < len(r) else "" for r in data]
        nulls = sum(1 for v in vals if v.lower() in null_tokens)
        out["null_fraction"][col] = round(nulls / len(vals), 4)
        nums = []
        for v in vals:
            try:
                nums.append(float(v.replace(",", ".")))
            except ValueError:
                pass
        if len(nums) >= 0.8 * (len(vals) - nulls) and nums:
            out["numeric_cols"][col] = _num_stats(nums)
        else:
            uniq = {}
            for v in vals:
                if v.lower() not in null_tokens:
                    uniq[v] = uniq.get(v, 0) + 1
            if 1 <= len(uniq) <= 20:
                tot = sum(uniq.values())
                out["class_balance"][col] = {k: round(c / tot, 4)
                                             for k, c in sorted(uniq.items(), key=lambda kv: -kv[1])}
    return out


def analyze_las_bytes(b: bytes, max_rows=30000):
    """LAS 2.0 well log: curves (mnemonic/unit) + numeric distributions + NULL fraction."""
    text = b.decode("latin-1", "replace")
    lines = text.splitlines()
    section, curves, null_val, rows = None, [], -999.25, []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("~"):
            section = s[1].upper()
            continue
        if section == "W" and s.upper().startswith("NULL"):
            m = re.search(r"NULL\s*\.\s*([\-\d.eE+]+)", s)
            if m:
                null_val = float(m.group(1))
        elif section == "C":
            m = re.match(r"([\w:()\-/]+)\s*\.(\S*)", s)
            if m:
                curves.append({"mnemonic": m.group(1), "unit": m.group(2)})
        elif section == "A" and len(rows) < max_rows:
            try:
                rows.append([float(x) for x in s.split()])
            except ValueError:
                pass
    if not curves or not rows:
        return None
    ncur = len(curves)
    cols = {c["mnemonic"]: [] for c in curves}
    nulls = {c["mnemonic"]: 0 for c in curves}
    for r in rows:
        for j in range(min(ncur, len(r))):
            mn = curves[j]["mnemonic"]
            if abs(r[j] - null_val) < 1e-6:
                nulls[mn] += 1
            else:
                cols[mn].append(r[j])
    class_balance = {}
    for mn, v in cols.items():           # coded label columns = low cardinality
        uniq = set(v)
        if v and 1 <= len(uniq) <= 20 and all(float(x).is_integer() for x in list(uniq)[:20]):
            cb = {}
            for x in v:
                cb[str(int(x))] = cb.get(str(int(x)), 0) + 1
            class_balance[mn] = {k: round(c / len(v), 4)
                                 for k, c in sorted(cb.items(), key=lambda kv: -kv[1])}
    return {"curves": curves, "null_value": null_val, "n_rows_sampled": len(rows),
            "numeric_cols": {mn: _num_stats(v) for mn, v in cols.items() if v},
            "class_balance": class_balance,
            "null_fraction": {mn: round(nulls[mn] / len(rows), 4) for mn in nulls}}


def analyze_mesh_file(path: str, max_faces=400000):
    """PLY (ascii/binary_LE) & STL (binary/ascii): vertex/face-count, bbox, watertight
    (closed 2-manifold at edge level: every undirected edge is shared by exactly 2 faces)."""
    import struct
    with open(path, "rb") as f:
        head = f.read(64 * 1024)
    verts, faces = [], []
    if head[:3] == b"ply":
        hdr_end = head.find(b"end_header\n")
        if hdr_end < 0:
            return None
        hdr = head[:hdr_end].decode("ascii", "replace").splitlines()
        fmt = next((l.split()[1] for l in hdr if l.startswith("format")), "")
        n_v = n_f = 0
        vprops, cur = [], None
        for l in hdr:
            t = l.split()
            if not t:
                continue
            if t[0] == "element":
                cur = t[1]
                if t[1] == "vertex":
                    n_v = int(t[2])
                elif t[1] == "face":
                    n_f = int(t[2])
            elif t[0] == "property" and cur == "vertex" and t[1] != "list":
                vprops.append(t[1])
        with open(path, "rb") as f:
            f.seek(hdr_end + len(b"end_header\n"))
            body = f.read()
        if fmt == "ascii":
            toks = body.decode("ascii", "replace").split("\n")
            for i in range(min(n_v, len(toks))):
                p = toks[i].split()
                if len(p) >= 3:
                    verts.append(tuple(float(x) for x in p[:3]))
            for i in range(n_v, min(n_v + n_f, len(toks))):
                p = toks[i].split()
                if p and p[0] == "3" and len(p) >= 4:
                    faces.append(tuple(int(x) for x in p[1:4]))
        elif "binary_little_endian" in fmt:
            sz = {"float": 4, "float32": 4, "double": 8, "uchar": 1, "uint8": 1,
                  "char": 1, "short": 2, "ushort": 2, "int": 4, "uint": 4,
                  "int32": 4, "uint32": 4}
            types = []
            for l in hdr:
                t = l.split()
                if t and t[0] == "property" and t[1] != "list":
                    types.append(t[1])
            vstride = sum(sz.get(t, 4) for t in types[:len(vprops)])
            off = 0
            for i in range(n_v):
                x, y, z = struct.unpack_from("<fff", body, off)
                verts.append((x, y, z))
                off += vstride
            for i in range(min(n_f, max_faces)):
                cnt = body[off]
                off += 1
                idx = struct.unpack_from("<" + "i" * cnt, body, off)
                off += 4 * cnt
                if cnt == 3:
                    faces.append(idx)
                else:
                    for k in range(1, cnt - 1):
                        faces.append((idx[0], idx[k], idx[k + 1]))
        else:
            return {"format": "ply-" + fmt, "n_vertices": n_v, "n_faces": n_f, "watertight": None}
    elif head[:5].lower() == b"solid" and b"facet" in head[:2000]:
        txt = open(path, "rb").read().decode("ascii", "replace")
        vs = re.findall(r"vertex\s+([\-\d.eE+]+)\s+([\-\d.eE+]+)\s+([\-\d.eE+]+)", txt)
        vmap = {}
        for i in range(0, len(vs) - 2, 3):
            tri = []
            for k in range(3):
                key = tuple(round(float(c), 6) for c in vs[i + k])
                tri.append(vmap.setdefault(key, len(vmap)))
            faces.append(tuple(tri))
        verts = [None] * len(vmap)
        for k, ix in vmap.items():
            verts[ix] = k
    else:
        with open(path, "rb") as f:
            f.seek(80)
            n_tri = struct.unpack("<I", f.read(4))[0]
            vmap = {}
            for i in range(min(n_tri, max_faces)):
                rec = f.read(50)
                if len(rec) < 50:
                    break
                tri = []
                for k in range(3):
                    x, y, z = struct.unpack_from("<fff", rec, 12 + 12 * k)
                    key = (round(x, 6), round(y, 6), round(z, 6))
                    tri.append(vmap.setdefault(key, len(vmap)))
                faces.append(tuple(tri))
            verts = [None] * len(vmap)
            for k, ix in vmap.items():
                verts[ix] = k
    if not faces:
        return {"n_vertices": len(verts), "n_faces": 0, "watertight": False}
    edges = {}
    for f3 in faces:
        for a, bq in ((f3[0], f3[1]), (f3[1], f3[2]), (f3[2], f3[0])):
            e = (a, bq) if a < bq else (bq, a)
            edges[e] = edges.get(e, 0) + 1
    watertight = all(c == 2 for c in edges.values())
    xs = [v[0] for v in verts if v], [v[1] for v in verts if v], [v[2] for v in verts if v]
    bbox = [round(max(a) - min(a), 4) if a else 0 for a in xs]
    return {"n_vertices": len(verts), "n_faces": len(faces), "watertight": watertight,
            "n_edges": len(edges), "boundary_edges": sum(1 for c in edges.values() if c == 1),
            "bbox_extent": bbox}


def parse_tfrecords(b: bytes, key_probes=()):
    """TFRecord-raming: [len u64le][crc4][data][crc4]. Returnerar antal HELA records,
    embedded PNG/JPEG blobs (with header parse) and which feature keys appear in the bytes."""
    import struct
    recs, off = [], 0
    while off + 12 <= len(b):
        ln = struct.unpack_from("<Q", b, off)[0]
        if off + 12 + ln + 4 > len(b) or ln > len(b):
            break
        recs.append(b[off + 12: off + 12 + ln])
        off += 12 + ln + 4
    images = []
    keys_seen = set()
    for r in recs:
        for kp in key_probes:
            if kp.encode() in r:
                keys_seen.add(kp)
        i = 0
        while True:
            p = r.find(b"\x89PNG\r\n\x1a\n", i)
            j = r.find(b"\xff\xd8\xff", i)
            if p < 0 and j < 0:
                break
            if p >= 0 and (j < 0 or p < j):
                end = r.find(b"IEND", p)
                if end < 0:
                    break
                blob = r[p:end + 8]
                i = end + 8
            else:
                end = r.find(b"\xff\xd9", j + 3)
                if end < 0:
                    break
                blob = r[j:end + 2]
                i = end + 2
            info = analyze_image_bytes(blob)
            if info:
                info["_bytes"] = blob
                images.append(info)
    return {"n_complete_records": len(recs), "images": images, "keys_seen": sorted(keys_seen)}


def analyze_h5_file(path: str, max_elems=2_000_000):
    """HDF5 payload: dataset name/shape/dtype + numeric distributions per channel (guarded on h5py)."""
    try:
        import h5py
        import numpy as np
    except ImportError:
        return None
    out = {"datasets": []}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                d = {"name": name, "shape": list(obj.shape), "dtype": str(obj.dtype)}
                if obj.size and obj.size <= max_elems and obj.dtype.kind in "fiu":
                    a = np.asarray(obj[()], dtype=float)
                    if a.ndim == 1:
                        a = a[:, None]
                    if a.ndim == 2:
                        d["per_channel"] = [_num_stats([float(x) for x in a[::max(1, len(a) // 5000), j]])
                                            for j in range(min(a.shape[1], 8))]
                out["datasets"].append(d)
        f.visititems(visit)
    return out if out["datasets"] else None


def analyze_dir_payload(sample_dir: str):
    """Generic payload structure for a sample directory: census + image/geometry/tabular statistics."""
    out = {"files": 0, "bytes": 0, "by_ext": {}, "images": [], "meshes": [], "tables": [],
           "arrays": [], "tfrecords": [], "resolutions": {}, "formats": {}}
    for root, _, files in os.walk(sample_dir):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            sz = os.path.getsize(p)
            out["files"] += 1
            out["bytes"] += sz
            ext = fn.rsplit(".", 1)[-1].lower() if "." in fn else "?"
            out["by_ext"][ext] = out["by_ext"].get(ext, 0) + 1
            if ext in EXT_CLASS["image"]:
                info = analyze_image_bytes(open(p, "rb").read(4 * 1024 * 1024))
                if info:
                    key = f'{info["w"]}x{info["h"]}x{info.get("channels", "?")}'
                    out["resolutions"][key] = out["resolutions"].get(key, 0) + 1
                    out["formats"][info["format"]] = out["formats"].get(info["format"], 0) + 1
                    out["images"].append({"file": fn, **{k: v for k, v in info.items() if k != "_bytes"}})
            elif ext in ("ply", "stl", "obj"):
                m = analyze_mesh_file(p)
                if m:
                    out["meshes"].append({"file": fn, **m})
            elif ext in ("csv", "tsv"):
                t = analyze_csv_bytes(open(p, "rb").read(30 * 1024 * 1024))
                if t:
                    out["tables"].append({"file": fn, **t})
            elif ext == "las":
                t = analyze_las_bytes(open(p, "rb").read(30 * 1024 * 1024))
                if t:
                    out["tables"].append({"file": fn, **t})
            elif ext in ("h5", "hdf5"):
                a = analyze_h5_file(p)
                if a:
                    out["arrays"].append({"file": fn, **a})
            elif ".tfrecord" in fn.lower():
                tf = parse_tfrecords(open(p, "rb").read(64 * 1024 * 1024),
                                     key_probes=("video", "segmentations", "depth", "instances",
                                                 "camera", "forward_flow", "normal", "bboxes"))
                res = {}
                for im in tf["images"]:
                    key = f'{im["w"]}x{im["h"]}x{im.get("channels", "?")}@{im["format"]}'
                    res[key] = res.get(key, 0) + 1
                out["tfrecords"].append({"file": fn, "n_complete_records": tf["n_complete_records"],
                                         "keys_seen": tf["keys_seen"], "embedded_images": res})
    out["images"] = out["images"][:12]
    return out


# --------------------------------------------------------- DEEP_SOURCES (fetch plans) --
# 6 curated public sources spread over the families and over two concrete goals:
# GOAL A "calibrate the material/BRDF channel": ambientCG (primary), SkyFinder (negative control)
# GOAL B "anchor a part-decomposition prior against real CAD geometry": BOP/icbin (primary),
# Kubric (synthetic GT), SkyFinder (field-image control)
# All were verified directly fetchable via a files API / direct link BEFORE selection (a source
# whose records list 0 files via the API is EXCLUDED).

def _fetch_skyfinder(d):
    fs = []
    p, n = fetch_to("https://zenodo.org/api/records/5884485/files/858.zip/content",
                    os.path.join(d, "858.zip"), 12 * 1024 * 1024)
    fs.append((p, n))
    p2, n2 = fetch_to("https://zenodo.org/api/records/5884485/files/skyfinder_masks.zip/content",
                      os.path.join(d, "masks.zip"), 2 * 1024 * 1024)
    fs.append((p2, n2))
    import zipfile
    for z, sub in ((p, "img"), (p2, "mask")):
        with zipfile.ZipFile(z) as zf:
            names = [x for x in zf.namelist() if not x.endswith("/")]
            keep = names[:8] if sub == "img" else names
            for nm in keep:
                data = zf.open(nm).read()
                out = os.path.join(d, sub + "__" + nm.replace("/", "__"))
                open(out, "wb").write(data)
    os.remove(p)
    return sum(x[1] for x in fs)


def _fetch_bop_icbin(d):
    total = 0
    p, n = fetch_to("https://huggingface.co/datasets/bop-benchmark/icbin/resolve/main/icbin_models.zip",
                    os.path.join(d, "icbin_models.zip"), 4 * 1024 * 1024)
    total += n
    import zipfile
    with zipfile.ZipFile(p) as zf:
        for nm in zf.namelist():
            if nm.endswith((".ply", ".json")):
                open(os.path.join(d, nm.replace("/", "__")), "wb").write(zf.open(nm).read())
    os.remove(p)
    url = "https://huggingface.co/datasets/bop-benchmark/icbin/resolve/main/icbin_test_bop19.zip"
    want = re.compile(r"test/000001/(scene_gt\.json|scene_camera\.json|scene_gt_info\.json|"
                      r"rgb/00000[0-3]\.\w+|mask_visib/00000[0-3]_\d{6}\.png|depth/00000[0-3]\.\w+)$")
    paths, fetched, n_total, _ = zip_sample(url, lambda nm: bool(want.search(nm)), d, max_members=40)
    total += fetched
    return total


def _fetch_kubric(d):
    total = 0
    base = "https://storage.googleapis.com/kubric-public/tfds/movi_c/256x256/1.0.0/"
    for small in ("features.json", "dataset_info.json"):
        _, n = fetch_to(base + small, os.path.join(d, small), 2 * 1024 * 1024)
        total += n
    # ONE movi_c record (256x256, all channels) measured 27 030 620 B, so a 30MB prefix yields
    # >=1 COMPLETE record (24 frames + 24 seg + depth/flow) for the payload parse.
    data = _http_read(base + "movi_c-test.tfrecord-00000-of-00256",
                      30 * 1024 * 1024, rng="bytes=0-31457279")
    open(os.path.join(d, "movi_c_shard0_head.tfrecord.part"), "wb").write(data)
    return total + len(data)


def _fetch_ambientcg(d):
    import urllib.request, zipfile
    api = ("https://ambientcg.com/api/v2/full_json?type=Material&limit=3"
           "&include=downloadData&sort=popular")
    raw = _http_read(api, 4 * 1024 * 1024)
    total = len(raw)
    assets = json.loads(raw).get("foundAssets", [])
    for a in assets[:3]:
        dls = (a.get("downloadFolders", {}).get("default", {})
               .get("downloadFiletypeCategories", {}).get("zip", {}).get("downloads", []))
        one_k = next((x for x in dls if x.get("attribute") == "1K-JPG"), None)
        if not one_k:
            continue
        p = os.path.join(d, a["assetId"] + "_1K.zip")
        _, n = fetch_to(one_k["downloadLink"], p, 20 * 1024 * 1024)
        total += n
        with zipfile.ZipFile(p) as zf:
            for nm in zf.namelist():
                if nm.lower().endswith((".jpg", ".png")):
                    open(os.path.join(d, a["assetId"] + "__" + os.path.basename(nm)), "wb"
                         ).write(zf.open(nm).read())
        os.remove(p)
    return total


def _fetch_force2020(d):
    url = ("https://zenodo.org/api/records/4351156/files/"
           "LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip/content")
    paths, fetched, n_total, names = zip_sample(url, lambda nm: nm.lower().endswith(".las"),
                                                d, max_members=3, per_member_cap=25 * 1024 * 1024)
    open(os.path.join(d, "_namelist.json"), "w").write(json.dumps({"n_members": n_total,
                                                                   "sample": names[:50]}))
    return fetched


def _fetch_bosch_cnc(d):
    import urllib.request
    total = 0
    for sub in ("good", "bad"):
        try:
            raw = _http_read(f"https://api.github.com/repos/boschresearch/CNC_Machining/"
                             f"contents/data/M01/OP07/{sub}", 2 * 1024 * 1024)
            items = [x for x in json.loads(raw) if x["name"].endswith(".h5")][:3]
        except Exception:
            items = []
        for it in items:
            _, n = fetch_to(it["download_url"], os.path.join(d, sub + "__" + it["name"]),
                            8 * 1024 * 1024)
            total += n
    return total


DEEP_SOURCES = [
    {"db_name": "SkyFinder", "fetch": _fetch_skyfinder, "l3": "sam2-vs-skymask",
     "goal_tags": {"A": "negative-control", "B": "field-image-control"}},
    {"db_name": "BOP suite (LINEMOD/YCB-V/T-LESS/HOPE)", "fetch": _fetch_bop_icbin,
     "l3": "sam2-vs-gt-mask", "goal_tags": {"A": "irrelevant", "B": "primary"}},
    {"db_name": "Kubric", "fetch": _fetch_kubric, "l3": "sam2-vs-synth-seg",
     "goal_tags": {"A": "irrelevant", "B": "synthetic-GT"}},
    {"db_name": "ambientCG", "fetch": _fetch_ambientcg, "l3": None,
     "goal_tags": {"A": "primary", "B": "irrelevant"}},
    {"db_name": "FORCE 2020 Well Log & Lithofacies", "fetch": _fetch_force2020, "l3": None,
     "goal_tags": {"A": "irrelevant", "B": "irrelevant"}},
    {"db_name": "Bosch CNC_Machining", "fetch": _fetch_bosch_cnc, "l3": None,
     "goal_tags": {"A": "irrelevant", "B": "irrelevant"}},
]


def payload_text_summary(payload: dict, extra: str = "") -> str:
    """Render payload findings to text so the L0 machinery (family/pairing/model match) can be
    re-run on L2 evidence -- the L0->L2 diff is then mechanical, not narrated."""
    bits = [extra]
    for ext, n in payload.get("by_ext", {}).items():
        bits.append(f"{n} x file.{ext}")
    for r, n in payload.get("resolutions", {}).items():
        bits.append(f"{n} images at {r} px resolution")
    for m in payload.get("meshes", []):
        bits.append(f'mesh {m["file"]} vertices={m.get("n_vertices")} faces={m.get("n_faces")} '
                    f'watertight={m.get("watertight")}')
    for t in payload.get("tables", []):
        cols = list(t.get("numeric_cols", {}).keys())
        bits.append("numeric columns: " + " ".join(cols[:40]))
        if t.get("class_balance"):
            bits.append("label classes: " + " ".join(t["class_balance"].keys()))
        if t.get("curves"):
            bits.append("well log curves: " + " ".join(c["mnemonic"] for c in t["curves"]))
    # NOTE: only MEASURED names/shapes are rendered -- no regex-friendly extra words. The payload
    # text is evidence, not a channel for smuggling keywords through.
    for a in payload.get("arrays", []):
        for ds in a.get("datasets", [])[:6]:
            bits.append(f'hdf5 dataset {ds["name"]} shape={ds["shape"]} dtype={ds["dtype"]}')
    for tfr in payload.get("tfrecords", []):
        bits.append(f'tfrecord feature keys: {" ".join(tfr["keys_seen"])}; embedded images '
                    + " ".join(f"{n} at {k}" for k, n in tfr["embedded_images"].items()))
    return " | ".join(b for b in bits if b)


def gt_promise_checks(db_name: str, l0_profile: dict, payload: dict, sample_dir: str):
    """Ground-truth-promise verification: do the L0 profile's promises (gt_keys, pairings, family
    i VERKLIG payload? Returnerar lista av {promise, held, evidence}."""
    checks = []
    gt = set(l0_profile["graf_varden"]["gt_keys"])
    P = set(l0_profile["struktur"]["pairings"])
    files = []
    for root, _, fns in os.walk(sample_dir):
        files += fns
    fl = " ".join(files).lower()
    # payload-internal keys (tfrecord features, hdf5 dataset names) count as payload evidence
    fl += " " + " ".join(k for tfr in payload.get("tfrecords", []) for k in tfr.get("keys_seen", []))
    fl += " " + " ".join(ds["name"].lower() for a in payload.get("arrays", [])
                         for ds in a.get("datasets", []))

    def add(promise, held, evidence):
        checks.append({"promise": promise, "held": bool(held), "evidence": str(evidence)[:200]})

    if "instance-segmentation-masks" in gt or "segmentation-gt" in P:
        add("segmentation-masks-in-payload", "mask" in fl, f"mask files: {fl.count('mask')} name matches")
    if "pose-gt" in gt or "image<->pose" in P:
        held = ("scene_gt" in fl) or ("pose" in fl) or ("cam_r" in fl) or ("camera" in fl)
        add("pose-gt-in-payload", held,
            "scene_gt/pose/camera channel in payload" if held else "missing in sample")
    if "mesh<->photo" in P or "geometry-file(CAD/mesh)" in l0_profile["graf_varden"]["calibration_anchors"]:
        add("geometry-files-in-payload", bool(payload.get("meshes")) or any(
            e in payload.get("by_ext", {}) for e in ("ply", "stl", "obj", "step")),
            f'meshes={len(payload.get("meshes", []))}')
    if l0_profile["graf_varden"]["closed_loop_tuple"]:
        has_tab = bool(payload.get("tables"))
        add("closed-loop-tuple-in-payload", has_tab and any(
            t.get("numeric_cols") for t in payload["tables"]), f"tabeller={len(payload.get('tables', []))}")
    if "material-brdf-gt" in gt or "material-response" in P:
        maps = {m for m in ("color", "normal", "roughness", "displacement", "ambientocclusion")
                if m in fl}
        add("brdf-map-set-complete", {"color", "normal", "roughness"} <= maps, f"maps={sorted(maps)}")
    if "measured-field-arrays" in gt or "timestamped-series" in gt:
        add("field/series-arrays-in-payload", any(e in payload.get("by_ext", {})
                                                  for e in ("h5", "las", "nc", "npz", "csv")),
            f'exts={list(payload.get("by_ext", {}).keys())}')
    fam = l0_profile["avbildningstyp"]["family"]
    if fam == "perfect-gt-synthetic":
        add("synthetic-gt-channels-present", ("segmentation" in fl) or ("features.json" in fl),
            "features/segmentation-kanaler i payload")
    # DISCOVERIES (not L0 promises): payload facts L0 never promised -- this is L2's own value and
    # must enter the decision even when the metadata was silent (e.g. 30+ mask files unpromised).
    promised = {c["promise"] for c in checks}
    if "segmentation-masks-in-payload" not in promised and ("mask" in fl):
        add("segmentation-masks-in-payload[DISCOVERED]", True,
            f"mask files found in payload without an L0 promise: {fl.count('mask')} name matches")
    if any(tfr.get("keys_seen") for tfr in payload.get("tfrecords", [])):
        keys = sorted({k for tfr in payload["tfrecords"] for k in tfr["keys_seen"]})
        if "segmentations" in keys and "synthetic-gt-channels-present" not in promised:
            add("segmentation-masks-in-payload[DISCOVERED]", True, f"tfrecord-nycklar: {keys}")
    for m in payload.get("meshes", []):
        if m.get("watertight") is False:
            add("mesh-watertight[DISCOVERED]", False,
                f'{m["file"]}: boundary_edges={m.get("boundary_edges")} av {m.get("n_edges")} '
                f'(near-closed but NOT watertight -- a ground-truth nuance for simulation use)')
    return checks


def run_level2(entry: dict, rows_by_name: dict, budget_state: dict, prev_text: str = ""):
    """L2 for ONE source: fetch a payload sample into data/acquired/samples/<slug>/, analyse it and
    diff against L0. prev_text = cumulative L1 text (a level must NEVER drop a lower level's
    evidence, otherwise the flips measure information loss instead of information gain)."""
    r = rows_by_name[entry["db_name"]]
    slug = slugify(r["name"])
    d = os.path.join(SAMPLES_DIR, slug)
    os.makedirs(d, exist_ok=True)
    if not _disk_ok():
        return {"error": "disk floor: <12GB free on /, fetch refused"}
    l0 = profile_dataset(r["name"], r["category"], row_text(r), probe_level="db-metadata-only")
    t0 = time.perf_counter()
    marker = os.path.join(d, "_fetched.json")
    if os.path.exists(marker):
        fetched = 0                       # sample already on disk (idempotent re-run)
    else:
        if budget_state["bytes"] >= L2_TOTAL_BYTE_CAP:
            return {"error": "total byte cap of 500MB reached"}
        fetched = entry["fetch"](d)
        budget_state["bytes"] += fetched
        with open(marker, "w") as f:
            json.dump({"bytes_fetched": fetched, "ts": time.time()}, f)
    t_fetch = time.perf_counter() - t0
    t1 = time.perf_counter()
    payload = analyze_dir_payload(d)
    l2_text = (prev_text or row_text(r)) + " | PAYLOAD: " + payload_text_summary(payload)
    l2_prof = profile_dataset(r["name"], r["category"], l2_text, probe_level="payload-sample-L2")
    t_analyze = time.perf_counter() - t1
    checks = gt_promise_checks(r["name"], l0, payload, d)
    diff = {
        "family_l0": l0["avbildningstyp"]["family"],
        "family_l2": l2_prof["avbildningstyp"]["family"],
        "family_flip": l0["avbildningstyp"]["family"] != l2_prof["avbildningstyp"]["family"],
        "confidence_l0": l0["avbildningstyp"]["confidence"],
        "confidence_l2": l2_prof["avbildningstyp"]["confidence"],
        "pairings_added": sorted(set(l2_prof["struktur"]["pairings"]) - set(l0["struktur"]["pairings"])),
        "pairings_lost": sorted(set(l0["struktur"]["pairings"]) - set(l2_prof["struktur"]["pairings"])),
        "model_matches_added": sorted({m["model"] for m in l2_prof["model_matcher"]}
                                      - {m["model"] for m in l0["model_matcher"]}),
        "closed_loop_l0": l0["graf_varden"]["closed_loop_tuple"] is not None,
        "closed_loop_l2_payload_confirmed": any(t.get("numeric_cols") for t in payload.get("tables", [])),
    }
    return {"sample_dir": d, "bytes_fetched": fetched, "bytes_on_disk": payload["bytes"],
            "fetch_s": round(t_fetch, 3), "analyze_s": round(t_analyze, 3),
            "payload": {k: v for k, v in payload.items() if k != "images"} |
                       {"images_sample": payload["images"][:6]},
            "gt_promise_checks": checks,
            "n_promises": len(checks), "n_promises_held": sum(1 for c in checks if c["held"]),
            "diff_vs_l0": diff, "l2_profile_id": l2_prof["id"], "_l2_prof": l2_prof}


# --------------------------------------------------------------------- L3 (SAM2) --

def _load_sam2(points_per_side=16):
    os.environ.setdefault("HF_HOME", HF_HOME_DEFAULT)
    import glob as _g
    import torch  # noqa
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    ckpts = _g.glob(os.path.join(os.environ["HF_HOME"], "coordinator",
                                 "models--facebook--sam2.1-hiera-large", "snapshots", "*",
                                 "sam2.1_hiera_large.pt"))
    if not ckpts:
        raise RuntimeError("SAM2.1 checkpoint missing in the HF_HOME cache")
    model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", ckpts[0], device="cuda")
    return SAM2AutomaticMaskGenerator(model, points_per_side=points_per_side)


def _iou(a, b):
    import numpy as np
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def _best_iou_per_gt(amg_masks, gt_masks):
    """Per ground-truth instance: best IoU over the model's automatic masks (per unit, not just the mean)."""
    out = []
    for g in gt_masks:
        best = 0.0
        for m in amg_masks:
            v = _iou(m["segmentation"], g)
            if v > best:
                best = v
        out.append(round(best, 4))
    return out


def run_level3(entry: dict, l2_block: dict, max_images=4):
    """L3: a real segmentation-model channel on the source images; coupling strength measured directly."""
    import numpy as np
    from PIL import Image
    mode = entry.get("l3")
    if not mode:
        return {"skipped": "ingen L3-plan (modalitet passar ej modellkanalen)"}
    d = l2_block.get("sample_dir")
    t0 = time.perf_counter()
    amg = _load_sam2()
    t_load = time.perf_counter() - t0
    pairs, note = [], ""
    if mode == "sam2-vs-gt-mask":                     # BOP icbin: rgb + mask_visib per instans
        rgbs = sorted(f for f in os.listdir(d) if "__rgb__" in f)[:max_images]
        for rgb in rgbs:
            frame_id = rgb.split("__")[-1].split(".")[0]
            gts = sorted(f for f in os.listdir(d)
                         if "__mask_visib__" in f and f.split("__")[-1].startswith(frame_id + "_"))
            if gts:
                pairs.append((os.path.join(d, rgb),
                              [os.path.join(d, g) for g in gts]))
        note = "GT = BOP mask_visib per objekt-instans (verklig scen, CAD-modell finns)"
    elif mode == "sam2-vs-skymask":                   # SkyFinder: kamerabilder + statisk himmelsmask
        masks = [f for f in os.listdir(d) if f.startswith("mask__") and "858" in f]
        imgs = sorted(f for f in os.listdir(d) if f.startswith("img__"))[:max_images]
        if masks:
            pairs = [(os.path.join(d, i), [os.path.join(d, masks[0])]) for i in imgs]
        note = "GT = binary sky mask per camera (static, shared by all frames)"
    elif mode == "sam2-vs-synth-seg":                 # Kubric: JPEG-frames + 8-bit seg-PNG ur tfrecord
        part = os.path.join(d, "movi_c_shard0_head.tfrecord.part")
        feats = []
        fj = os.path.join(d, "features.json")
        if os.path.exists(fj):
            feats = re.findall(r'"([a-z_]+)"\s*:', open(fj).read())
        tf = parse_tfrecords(open(part, "rb").read(),
                             key_probes=("video", "segmentations", "depth", "instances"))
        # MEASURED (from the shard prefix): frames are RGB PNG (channels=3), segmentations are
        # 8-bit single-channel PNG, depth/flow are 16-bit. Pairing = k-th RGB with k-th seg PNG.
        frames = [im for im in tf["images"] if im["format"] == "png" and im.get("channels") == 3]
        segs = [im for im in tf["images"] if im["format"] == "png"
                and im.get("bit_depth", 8) == 8 and im.get("channels", 3) == 1]
        note = (f"CONSTRUCTED pairing assumption (declared): k-th RGB PNG (frame) <-> k-th 8-bit "
                f"single-channel PNG (segmentation) within the same record; rgb={len(frames)} seg8={len(segs)}; "
                f"keys in records: {tf['keys_seen']}")
        for k in range(0, min(len(frames), len(segs), max_images)):
            fp = os.path.join(d, f"_frame{k}.png")
            open(fp, "wb").write(frames[k]["_bytes"])
            sp = os.path.join(d, f"_seg{k}.png")
            open(sp, "wb").write(segs[k]["_bytes"])
            pairs.append((fp, [sp]))
    results = []
    t_infer = 0.0
    for img_path, gt_paths in pairs:
        im = np.array(Image.open(img_path).convert("RGB"))
        t1 = time.perf_counter()
        masks = amg.generate(im)
        t_infer += time.perf_counter() - t1
        gt_list = []
        for gp in gt_paths:
            g = np.array(Image.open(gp))
            if g.ndim == 3:
                g = g[..., 0]
            if g.shape[:2] != im.shape[:2]:
                g = np.array(Image.fromarray(g).resize((im.shape[1], im.shape[0]), Image.NEAREST))
            ids = [v for v in np.unique(g) if v != 0]
            if len(ids) > 1 and mode == "sam2-vs-synth-seg":
                for v in ids:
                    gt_list.append(g == v)
            else:
                gm = g > 0
                gt_list.append(gm if gm.mean() < 0.5 or mode != "sam2-vs-skymask" else gm)
        best = _best_iou_per_gt(masks, gt_list)
        if mode == "sam2-vs-skymask" and best and max(best) < 0.3:
            inv = _best_iou_per_gt(masks, [~g for g in gt_list])
            if max(inv) > max(best):
                best = inv
                note += " | polarity: the INVERTED mask gave a higher IoU (declared)"
        results.append({"image": os.path.basename(img_path), "n_amg_masks": len(masks),
                        "n_gt_instances": len(gt_list), "best_iou_per_gt": best,
                        "pred_iou_dist": _num_stats([round(m["predicted_iou"], 4) for m in masks]),
                        "stability_dist": _num_stats([round(m["stability_score"], 4) for m in masks])})
    all_ious = [v for r0 in results for v in r0["best_iou_per_gt"]]
    return {"mode": mode, "note": note, "model_load_s": round(t_load, 2),
            "infer_s_total": round(t_infer, 2), "n_images": len(results),
            "per_image": results, "coupling_iou": _num_stats(all_ious),
            "per_unit_iou_all": all_ious,
            "coupling_band": ("strong" if all_ious and sorted(all_ious)[len(all_ious) // 2] >= 0.5
                              else "weak" if all_ious and max(all_ious) >= 0.3 else
                              "none" if all_ious else "no-pairs")}


# ------------------------------------------------------ goal-relative decisions + curve --
# The resolution requirement is GOAL-RELATIVE. Two concrete goals:
# A: calibrate the material/BRDF channel for twin rendering
# B: anchor a part-decomposition prior against real CAD geometry
# The decision VECTORS below are CONSTRUCTED (declared); the values in them are measured per level.

def goal_decision(goal: str, prof: dict, l2: dict = None, l3: dict = None):
    P = set(prof["struktur"]["pairings"])
    models = {m["model"] for m in prof["model_matcher"]}
    if goal == "A":
        relevant = ("material-response" in P) or ("material-leg (BRDF-prior)" in models)
        maps_ok = None
        if l2:
            maps_ok = any(c["promise"] == "brdf-map-set-complete" and c["held"]
                          for c in l2.get("gt_promise_checks", []))
        verdict = ("calibratable" if relevant and maps_ok else
                   "candidate" if relevant else "irrelevant")
        return {"relevant": relevant, "payload_verified": maps_ok, "verdict": verdict}
    if goal == "B":
        sam2_match = any(m.startswith("SAM2") for m in models)
        mask_gt = None
        if l2:
            mask_gt = any(c["promise"] in ("segmentation-masks-in-payload",
                                           "segmentation-masks-in-payload[DISCOVERED]",
                                           "synthetic-gt-channels-present") and c["held"]
                          for c in l2.get("gt_promise_checks", []))
        band = (l3 or {}).get("coupling_band")
        verdict = (f"anchor-{band}" if band in ("strong", "weak", "none") else
                   "candidate" if sam2_match and (mask_gt is not False) else "irrelevant")
        return {"relevant": sam2_match, "mask_gt": mask_gt, "coupling_band": band,
                "verdict": verdict}
    fam = prof["avbildningstyp"]["family"]
    return {"verdict": f"{fam}|" + ",".join(sorted(models))}


def run_deep(max_level=3):
    """The whole battery: L0->L1->L2->L3 over DEEP_SOURCES; decision changes per (goal x level),
    the cost curve and the goal-relative depth allocation. Writes DEEP_REPORT."""
    rows = load_sources()
    rows_by_name = {r["name"]: r for r in rows}
    budget = {"bytes": 0}
    per_source, curve_rows = {}, []
    for entry in DEEP_SOURCES:
        name = entry["db_name"]
        r = rows_by_name[name]
        rec = {"goal_tags": entry["goal_tags"], "levels": {}}
        t0 = time.perf_counter()
        l0 = profile_dataset(r["name"], r["category"], row_text(r), probe_level="db-metadata-only")
        rec["levels"]["L0"] = {"cost_ms": round((time.perf_counter() - t0) * 1000, 3),
                               "bytes": 0, "profile": {
                                   "family": l0["avbildningstyp"]["family"],
                                   "confidence": l0["avbildningstyp"]["confidence"],
                                   "pairings": l0["struktur"]["pairings"],
                                   "models": sorted(m["model"] for m in l0["model_matcher"])}}
        decisions = {g: {"L0": goal_decision(g, l0)} for g in ("A", "B", "generic")}
        prof_l1 = l0
        l1_text = row_text(r)
        if max_level >= 1:
            t0 = time.perf_counter()
            text, raw_html, nbytes = probe_source(r)
            if text is not None:
                l1_text = row_text(r) + " " + text
                prof_l1 = profile_dataset(r["name"], r["category"], l1_text,
                                          raw_html or "", probe_level="db+live-probe")
            rec["levels"]["L1"] = {"cost_ms": round((time.perf_counter() - t0) * 1000, 1),
                                   "bytes": nbytes if isinstance(nbytes, int) else 0,
                                   "probe_error": None if text is not None else str(nbytes),
                                   "profile": {
                                       "family": prof_l1["avbildningstyp"]["family"],
                                       "confidence": prof_l1["avbildningstyp"]["confidence"],
                                       "pairings": prof_l1["struktur"]["pairings"],
                                       "models": sorted(m["model"] for m in prof_l1["model_matcher"])}}
            for g in decisions:
                decisions[g]["L1"] = goal_decision(g, prof_l1)
        l2 = None
        if max_level >= 2:
            l2 = run_level2(entry, rows_by_name, budget, prev_text=l1_text)
            l2p = l2.pop("_l2_prof", None)      # kumulativ L2-profil (L1-text + payload-text)
            rec["levels"]["L2"] = l2
            if "error" not in l2 and l2p is not None:
                rec["levels"]["L2"]["profile"] = {
                    "family": l2p["avbildningstyp"]["family"],
                    "confidence": l2p["avbildningstyp"]["confidence"],
                    "pairings": l2p["struktur"]["pairings"],
                    "models": sorted(m["model"] for m in l2p["model_matcher"])}
                for g in decisions:
                    decisions[g]["L2"] = goal_decision(g, l2p, l2=l2)
        l3 = None
        if max_level >= 3 and l2 and "error" not in l2:
            try:
                l3 = run_level3(entry, l2)
            except Exception as e:
                l3 = {"error": f"{type(e).__name__}: {e}"}
            rec["levels"]["L3"] = {k: v for k, v in (l3 or {}).items() if k != "per_unit_iou_all"} \
                if l3 else None
            if l3 and "error" not in l3 and "skipped" not in l3:
                for g in decisions:
                    decisions[g]["L3"] = goal_decision(g, l2p, l2=l2, l3=l3)
        rec["decisions"] = decisions
        per_source[name] = rec
    # ---- the curve: per (goal x level), the share of sources whose decision CHANGES at that level
    levels = ["L0", "L1", "L2", "L3"]
    curve = {}
    for g in ("A", "B", "generic"):
        curve[g] = {}
        for i in range(1, len(levels)):
            prev_l, cur_l = levels[i - 1], levels[i]
            changed, evald = [], 0
            for name, rec in per_source.items():
                dv = rec["decisions"].get(g, {})
                if cur_l in dv and prev_l in dv:
                    evald += 1
                    # a flip means the VERDICT changes (the decision), not a helper field
                    # (None -> False is not a decision change; counting it inflated the curve)
                    if dv[cur_l].get("verdict") != dv[prev_l].get("verdict"):
                        changed.append(name)
            curve[g][f"{prev_l}->{cur_l}"] = {
                "n_evaluated": evald, "n_changed": len(changed), "changed_sources": changed,
                "frac_changed": round(len(changed) / evald, 3) if evald else None}
    cost = {}
    for lv in levels:
        ms = [rec["levels"][lv]["cost_ms"] for rec in per_source.values()
              if lv in rec["levels"] and isinstance(rec["levels"][lv], dict)
              and "cost_ms" in rec["levels"][lv]]
        by = [rec["levels"][lv].get("bytes", 0) for rec in per_source.values()
              if lv in rec["levels"] and isinstance(rec["levels"][lv], dict)]
        if lv == "L2":
            ms = [round((rec["levels"][lv].get("fetch_s", 0) + rec["levels"][lv].get("analyze_s", 0))
                        * 1000, 1) for rec in per_source.values()
                  if lv in rec["levels"] and "error" not in rec["levels"][lv]]
            by = [rec["levels"][lv].get("bytes_fetched", 0) for rec in per_source.values()
                  if lv in rec["levels"] and "error" not in rec["levels"][lv]]
        if lv == "L3":
            ms = [round(((rec["levels"][lv] or {}).get("model_load_s", 0)
                         + (rec["levels"][lv] or {}).get("infer_s_total", 0)) * 1000, 1)
                  for rec in per_source.values()
                  if rec["levels"].get(lv) and "error" not in rec["levels"][lv]
                  and "skipped" not in rec["levels"][lv]]
            by = [0] * len(ms)
        cost[lv] = {"per_source_ms": ms, "mean_ms": round(sum(ms) / len(ms), 2) if ms else None,
                    "total_bytes": sum(by)}
    return per_source, curve, cost, budget


# ========================================================================= main ====

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--retrodict", action="store_true")
    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--probe", type=int, default=0, help="number of live probes (<2MB each, max 15)")
    ap.add_argument("--one", type=str, default=None, help="profile ONE named DB source")
    ap.add_argument("--no-write", action="store_true", help="do not write profile files (dry run)")
    ap.add_argument("--level", type=int, default=0, choices=(0, 1, 2, 3),
                    help="profiling depth: 0=DB metadata (default), 1=+live probe, "
                         "2=+payload sample (needs a fetch plan in DEEP_SOURCES), 3=+model channel")
    ap.add_argument("--deep", action="store_true",
                    help="run the whole L0->L3 battery over DEEP_SOURCES and write "
                         "the deep report (curve + goal allocation)")
    ap.add_argument("--deep-max-level", type=int, default=3, choices=(1, 2, 3))
    args = ap.parse_args()

    out = {}
    if args.selftest:
        out["selftest"] = run_selftest()
        print(json.dumps(out["selftest"], ensure_ascii=False, indent=1))
        sys.exit(0 if out["selftest"]["all_pass"] else 1)
    if args.one:
        rows = [r for r in load_sources() if r["name"] == args.one]
        if not rows:
            print(f"source '{args.one}' is not in the DB", file=sys.stderr)
            sys.exit(2)
        r = rows[0]
        p = profile_dataset(r["name"], r["category"], row_text(r))
        if args.level >= 1:
            text, raw_html, nbytes = probe_source(r)
            if text is not None:
                p = profile_dataset(r["name"], r["category"], row_text(r) + " " + text,
                                    raw_html or "", probe_level="db+live-probe")
        if args.level >= 2:
            entry = next((e for e in DEEP_SOURCES if e["db_name"] == r["name"]), None)
            if entry is None:
                p["l2"] = {"error": "no-fetch-plan: the source has no fetch plan in DEEP_SOURCES"}
            else:
                l2 = run_level2(entry, {r["name"]: r}, {"bytes": 0})
                l2.pop("_l2_prof", None)
                p["l2"] = l2
                if args.level >= 3 and "error" not in l2:
                    l3 = run_level3(entry, l2)
                    p["l3"] = {k: v for k, v in l3.items() if k != "per_unit_iou_all"}
                if "error" not in l2:
                    p["probe_level"] = f"payload-sample-L{args.level}"
        print(json.dumps(p, ensure_ascii=False, indent=1))
        if not args.no_write:
            print("->", write_profile(p), file=sys.stderr)
    if args.deep:
        per_source, curve, cost, budget = run_deep(max_level=args.deep_max_level)
        print(json.dumps({"curve": curve, "cost": cost, "bytes_total": budget["bytes"]},
                         ensure_ascii=False, indent=1))
        os.makedirs(os.path.dirname(DEEP_REPORT), exist_ok=True)
        with open(DEEP_REPORT + ".raw.json", "w") as f:
            json.dump({"per_source": per_source, "curve": curve, "cost": cost,
                       "bytes_total": budget["bytes"]}, f, ensure_ascii=False, indent=1, default=str)
        print("->", DEEP_REPORT + ".raw.json", file=sys.stderr)
    if args.retrodict:
        out["retrodict"] = run_retrodict()
        print(json.dumps(out["retrodict"], ensure_ascii=False, indent=1))
    if args.scale:
        scale, profiles = run_scale(write_profiles=not args.no_write)
        out["scale"] = scale
        print(json.dumps(scale, ensure_ascii=False, indent=1))
    if args.probe:
        out["probes"] = run_probes(args.probe, {})
        print(json.dumps(out["probes"], ensure_ascii=False, indent=1))
    if not any([args.selftest, args.retrodict, args.scale, args.probe, args.one, args.deep]):
        ap.print_help()


if __name__ == "__main__":
    main()
