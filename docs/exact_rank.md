# Optional exact rational rank

Three APIs accept `exact_rank=True`: `MarginNet.estimate`,
`PrecisionForm.add_measurement`, and `lineage_information`. Defaults retain the
existing floating-point cutoffs. The exact path uses rational elimination and
a Moore–Penrose inverse, so a small nonzero rational direction is not discarded
solely because it falls below a relative rank tolerance.

```python
from fractions import Fraction
import numpy as np
from graph_engine.precision_form import PrecisionForm

form = PrecisionForm.zeros(2)
form.add_measurement(
    np.eye(2), [1, 2], [[1, 0], [0, Fraction(1, 10**12)]],
    exact_rank=True,
)
assert form.J[1, 1] == 1e12
```

Integers, `Fraction`, finite `Decimal`, and finite binary floating values are
interpreted as their represented rational values. A binary float is not replaced
with a guessed decimal fraction. Full rank modulo a prime proves full rank over
the rationals. A deficient modular rank is inconclusive; after two primes the
implementation uses exact elimination to decide the rank.

`MarginNet` constructs the rational covariance before floating Gram products,
retaining supplied rational tree and shared-source weights. A shared-source
model requiring irrational square roots uses the existing floating path instead.
The same fallback applies to unsupported covariance entries; opting in does not
guarantee that every input model admits a rational calculation.

In `PrecisionForm`, the option controls the measurement covariance inverse and
its update. `A` and `y` retain their existing float conversion; later form reads
still use the form's tolerance. Exact intermediate inverse entries are contracted
before conversion to public floats, avoiding overflow when the final update is
representable. Both updated arrays are checked before mutation; nonfinite final
updates raise `ValueError` and leave the form unchanged. `MarginNet` likewise
avoids converting a huge intermediate information value before calculating a
small, representable uncertainty.

Public estimates and form state remain floating point. This is not an
outward-rounded probability certificate or a positive-semidefinite covariance
validator. Existing covariance-model assumptions still apply. Rational arithmetic
can be substantially slower and use more memory on large or high-bit-size
matrices; no performance advantage is claimed. The intended benefit is retaining
valid small directions and obtaining consistent rank and disagreement degrees
of freedom in supported rational models.
