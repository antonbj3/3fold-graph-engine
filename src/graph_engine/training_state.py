"""Snapshot existing compiled Graph beliefs for an injected numeric learner.

No model framework is imported and no graph store is created. Rows preserve
the compiled finite family, frozen partition, folded claim likelihood and
physical probe identities. The arrays describe states, not additional evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .compiled_regime import CompiledRegime


def _frozen_array(value):
    array=np.asarray(value,dtype='<f8')
    return np.frombuffer(array.tobytes(),dtype='<f8').reshape(array.shape)


@dataclass(frozen=True)
class CompiledTrainingBatch:
    """An immutable export from live regimes at one point in time.

    Use :meth:`from_regimes`; mixed partitions or family priors are refused.
    ``natural`` is (N,2,Q), with claims folded into A, and ``family_mass`` is the
    PRIOR required by the native decoder, never the already-updated posterior.
    All rows share that family and geometry. Original regime updates do not
    change this snapshot. Raw claims/history are not included; retain the
    original evidence separately if later recalibration is required.
    Likelihood constants are omitted: this exports predictive training states,
    not an absolute-evidence objective for fitting sensor calibration.
    """
    example_ids: tuple[str,...]
    natural: np.ndarray
    cells: np.ndarray
    family_mass: tuple[float,float,float]
    evidence_ids: tuple[tuple[str,...],...]
    fingerprint: str

    @classmethod
    def from_regimes(cls, examples):
        rows=tuple(examples)
        if not rows:
            raise ValueError('at least one named compiled regime is required')
        ids=[]; states=[]; evidence=[]; cells=None; prior=None
        for row in rows:
            if not isinstance(row,(tuple,list)) or len(row)!=2:
                raise ValueError('each example must be (unique string id, CompiledRegime)')
            key,regime=row
            if not isinstance(key,str) or not key or key in ids:
                raise ValueError('example ids must be unique nonempty strings')
            if not isinstance(regime,CompiledRegime):
                raise TypeError('training snapshots require CompiledRegime instances')
            current_cells=regime.cells
            current_prior=tuple(float(x) for x in regime.transition_prior()['mass'])
            if cells is None:
                cells=current_cells;prior=current_prior
            elif not np.array_equal(cells,current_cells) or prior!=current_prior:
                raise ValueError('training rows must have identical partitions and family priors')
            statistics=regime.natural_statistics()
            states.append(np.stack((statistics['kernel_a'],statistics['kernel_b'])))
            ids.append(key);evidence.append(tuple(sorted(regime.evidence_ids)))
        natural=_frozen_array(states);cells=_frozen_array(cells)
        if not np.all(np.isfinite(natural)):
            raise ValueError('compiled natural parameters must be finite')
        descriptor=json.dumps({'schema':1,'ids':ids,'prior':prior,'evidence':evidence,
                               'natural_shape':natural.shape,'cells_shape':cells.shape},
                              sort_keys=True,separators=(',',':')).encode()
        digest=hashlib.sha256(descriptor+cells.tobytes()+natural.tobytes()).hexdigest()
        return cls(tuple(ids),natural,cells,prior,tuple(evidence),digest)

    def indices_for(self, example_ids):
        """Explicit row identity for a reordered/minibatched learner input.

        Repetitions support sampling with replacement. Unknown IDs raise; the
        caller must pair feature rows with these SAME example IDs.
        """
        if isinstance(example_ids,str):
            raise ValueError('provide a sequence of example ids, not one string')
        lookup={key:i for i,key in enumerate(self.example_ids)}
        try:
            return np.array([lookup[key] for key in example_ids],dtype=np.int64)
        except (KeyError,TypeError) as exc:
            raise ValueError('unknown training example id') from exc
