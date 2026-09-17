"""Shared loaders for the engine experiments. Public data: SNAP cit-HepTh / cit-HepPh
(arXiv citation graphs, KDD Cup 2003). Files are fetched to $HUNT_DATA (default: ./data next to this file, git-ignored)."""
import gzip, os, urllib.request
from pathlib import Path
import numpy as np

DATA = Path(os.environ.get("HUNT_DATA", Path(__file__).resolve().parent / "data"))

def snap_citations(name: str):
    """(edges int64 m×2 [citing, cited], month int per node, ids). Month is read from the arXiv id
    (YYMMnnn), counted from 1992-01; the SNAP dates file covers under half of the nodes."""
    f = DATA / f"cit-{name}.txt.gz"
    if not f.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://snap.stanford.edu/data/cit-{name}.txt.gz", f)
    raw = np.array([l.split() for l in gzip.open(f, "rt") if l[0] != "#"], dtype=np.int64)
    ids, inv = np.unique(raw, return_inverse=True)
    edges = inv.reshape(raw.shape)
    yy, mm = ids // 100000, (ids // 1000) % 100
    year = np.where(yy >= 90, 1900 + yy, 2000 + yy)
    month = (year - 1992) * 12 + np.clip(mm, 1, 12) - 1
    return edges, month, ids

def largest_component(n, edges):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    A = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    keep = lab == np.bincount(lab).argmax()
    remap = -np.ones(n, np.int64); remap[keep] = np.arange(keep.sum())
    e = remap[edges]; e = e[(e >= 0).all(1)]
    return int(keep.sum()), e, keep

def simple_undirected(edges):
    e = np.sort(edges, axis=1); e = e[e[:, 0] != e[:, 1]]
    return np.unique(e, axis=0)
