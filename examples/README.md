# KATE examples

Runnable demonstrations of the pipeline on small synthetic systems, so each stage can be
inspected without a large trajectory or a GPU. Run them from the repository root after
installing (`bash install_kate.sh`, or `pip install -e ".[kinetics]"`).

```bash
python examples/demo_pathbound.py
python examples/demo_kinetic_codec.py
python examples/demo_kate.py
python examples/demo_bound_loss.py
```

| Example | What it shows |
|---|---|
| [`demo_pathbound.py`](demo_pathbound.py) | The path-space bound `KL(path) = ensemble + transition`. Two dynamics with the same stationary distribution give an ensemble term near zero and a large transition term, the case that motivates KATE. |
| [`demo_kinetic_codec.py`](demo_kinetic_codec.py) | The classical MSM-as-entropy-coder path: discretize, estimate the reversible MSM, and code the discrete-state sequence against the Markov entropy rate. |
| [`demo_kate.py`](demo_kate.py) | End-to-end flow-based codec on a synthetic trajectory: TICA collective variables, the normalizing flow, reweighted frame selection, entropy coding, the retained MSM, and the artifact accounting. |
| [`demo_bound_loss.py`](demo_bound_loss.py) | The differentiable path-bound loss, used when the bound is optimized directly rather than only reported. |

Each script prints the intermediate quantities (bound terms, implied timescales, artifact
size) so the numbers can be checked against the method. The synthetic generators are seeded,
so the output is reproducible.
