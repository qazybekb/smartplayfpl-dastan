"""Dastan — an open-source expected-points model for Fantasy Premier League.

Named after Dastan Satpayev, the first Kazakh footballer signed by an English
Premier League club.

    from dastan import data, mappings, model, metrics
    df = data.load()
    identities = mappings.load()
    feats = data.shipped_features(df)

Published by SmartPlayFPL (https://smartplayfpl.com). MIT licensed.
Built on the feature engineering of OpenFPL (https://github.com/daniegr/OpenFPL).
"""

__version__ = "1.0.0"
__all__ = ["data", "mappings", "model", "metrics", "predictor"]
