# BALLADEER Validator — Legacy v1.0.0

This directory preserves the original BALLADEER validator used for the
results reported in release v1.0.0.

The script is retained unchanged for reproducibility. Its clustering
procedure uses accuracy as a cluster-defining feature and subsequently
tests commission errors, creating partial outcome circularity.

The active validator in `scripts/HEALTHY_VALID_BALLADEER.py` implements
the corrected cluster-free and dimensional analysis introduced in v1.0.1.

Do not use this legacy version as the preferred current analysis.
