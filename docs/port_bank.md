# Prospective source-port banks

`graph_engine.port_bank.PortBank` loads an explicit versioned source-port contract,
checks its byte hash, canonical quantity types, AND lists, forward direction,
validity, costs, Gaussian shapes, and the declared structural row construction.
Gaussian coordinates in this version are **dimensionless artifact proxies**.
They have no calibration to physical uncertainty or worker success. Source
validity, unresolved laws, and empirical readiness remain explicit in the bank.

```python
from graph_engine.port_bank import PortBank

bank = PortBank.load("BANK.json", expected_sha256="<frozen SHA256>")
chosen = bank.form().best_set(bank.candidates(block_ids), 3, Q=bank.Q)
available, acquisition_order, missing = bank.execution([c.id for c in chosen])
build = bank.best_executable_set(block_ids, 3)
if build is not None:
    action = bank.action(build)  # native triple_bundle, structural currency
    assert action.meta["bank"] == bank.reference()
```

`execution` establishes **schema-level port reachability**, not implementation of
the source laws. The application must execute and independently test the selected
source program before reporting PASS. `best_executable_set` exhaustively searches
small pools, checks target reachability, and values acquired rows from the same
bank. It returns `None` when no target build exists and enforces `max_sets`.
The generic `PrecisionForm.best_set(Q)` continues to price Gaussian information:
positive Gaussian value can exist without any forward target closure.

The example source bank uses correctly directed artifact input/output types;
its physical payload units are separate from proxy units. A declared diagnostic
signed parity domain is separate from physical source claims. Feeds, certifies,
same-operator, and contradiction labels never become derivative signs.

`worker_graph` requires the exact bank reference and three canonical member IDs.
Signed claims require integer ±1, a registered composition domain and instance,
explicit validity contained in that domain, endpoints, and evidence. It validates
before passing claims to `Federation.add_graph`; `federation` is the convenience
consumer. The caller still owns independent evidence review. A manifest hash is
an identity check, not authentication or scientific acceptance.

The J3 pilot's exact rational conditioning control agrees with all 816 target
values within 7.4e-11 bits, and ordinary forward closure matches all reachability
labels. The strongest equally informed oracle ties the selector. This is a useful
execution guard and prospective contract, not a new selection theorem or an
empirical ranking advantage.
