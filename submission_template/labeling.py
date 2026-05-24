"""Fisher-information adaptive-labeling acquisition (Lecture 4 §sec-fisher-design).

For a Bernoulli outcome with predicted probability p, the Fisher information
the response carries about the latent ability is p * (1 - p) -- maximized at
p = 0.5 and 0 at p in {0, 1}. So the most informative labels to spend our
budget on are the items the model is most uncertain about. The platform calls
this function once per candidate (per round, per data category), sorts by
returned score, reveals the top-K labels per category, and passes those
labeled inputs into predict() as the `labeled` argument.

`acquisition_function` runs BEFORE predict() in each round, so there is no
`labeled` argument and no per-round calibration yet. We just call into the
same NCF + IRT-via-head pipeline that predict() uses, passing labeled=None.

If anything goes wrong, returning a non-finite value would discard ALL scores
for the round (the platform's documented fallback). We catch and return 0.0
on any error so a single bad candidate doesn't poison the round.
"""

from __future__ import annotations

import math

from model import predict


def acquisition_function(input: dict) -> float:
    try:
        p = predict(input, labeled=None)
        info = p * (1.0 - p)
    except Exception:
        return 0.0
    if not math.isfinite(info):
        return 0.0
    return float(info)
