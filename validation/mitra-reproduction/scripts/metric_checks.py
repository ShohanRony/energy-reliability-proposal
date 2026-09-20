"""Hand-checkable synthetic fixtures. These are NOT CNN benchmark results."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.metrics import calibration

out = Path("results/metric-checks")
out.mkdir(parents=True, exist_ok=True)
fixture = {"confidence": [0.1, 0.2, 0.3, 0.9], "correct": [0, 0, 1, 1], "n_bins": 2}
(out / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
r = {"kind": "synthetic_metric_test_not_paper_reproduction", "numpy_version": np.__version__,
     "expected": {"equal_mass": 0.275, "equal_width": 0.125}}
for scheme in ("equal_mass", "equal_width"):
    r[scheme] = calibration(fixture["confidence"], fixture["correct"], 2, scheme)
    assert abs(r[scheme]["ece"] - r["expected"][scheme]) < 1e-12
(out / "metrics.json").write_text(json.dumps(r, indent=2) + "\n")
print(json.dumps(r, indent=2))
