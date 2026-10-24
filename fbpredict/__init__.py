"""fbpredict - a small, honest football prediction research pipeline.

The project predicts two markets from opening Asian-handicap / price-pool style
odds (see README.md and tip.jpg):

    * AH  - does the favourite cover the opening handicap?
    * O/U - does the match go over the opening total line?

and is allowed to SKIP a match when neither market clears the edge threshold.

Design rules that this package deliberately enforces
----------------------------------------------------
1. No leakage: every rolling feature is shifted by one match (``.shift(1)``)
   and the backtest only ever shows the model matches from *before* the day
   being predicted.
2. Every fetched payload is archived verbatim together with ``fetched_at`` so
   the raw evidence of "I fetched this myself" is reproducible.
3. The model is frozen (pickled + hashed + git commit recorded) before any
   forward test. Changing it invalidates the forward test.
4. Small samples are reported as small samples. The report refuses to claim an
   edge when the test set is too small to support one.
"""

__version__ = "0.1.0"
