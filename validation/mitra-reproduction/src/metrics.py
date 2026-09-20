"""NumPy-only, aggregate top-label ECE. Values are fractions, not percents."""
import numpy as np


def calibration(confidence, correct, n_bins=10, scheme="equal_mass"):
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidence.ndim != 1 or confidence.size == 0 or correct.shape != confidence.shape:
        raise ValueError("Expected nonempty, equally sized vectors")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Confidence must be finite and in [0,1]")
    if not np.isin(correct, [0, 1]).all() or n_bins < 1:
        raise ValueError("Correctness must be binary and bins positive")
    if scheme == "equal_mass":
        # Explicit reconstruction choice: stable sorting; ties may span bins.
        groups = np.array_split(np.argsort(confidence, kind="stable"), n_bins)
    elif scheme == "equal_width":
        # Paper Eq. 1 uses right-closed intervals. Include zero in first bin.
        edges = np.arange(1, n_bins + 1, dtype=float) / n_bins
        bins = np.searchsorted(edges, confidence, side="left")
        groups = [np.flatnonzero(bins == b) for b in range(n_bins)]
    else:
        raise ValueError(scheme)
    rows = []
    for b, idx in enumerate(groups):
        if not len(idx):
            rows.append(dict(bin=b, count=0, accuracy=None, confidence=None, contribution=0.0))
            continue
        acc, conf = float(correct[idx].mean()), float(confidence[idx].mean())
        rows.append(dict(bin=b, count=len(idx), accuracy=acc, confidence=conf,
                         min_confidence=float(confidence[idx].min()),
                         max_confidence=float(confidence[idx].max()),
                         contribution=len(idx) / len(correct) * abs(acc - conf)))
    return {"ece": sum(r["contribution"] for r in rows), "bins": rows,
            "scheme": scheme, "n_bins": n_bins, "n": len(correct)}


def from_probabilities(probabilities, labels):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels)
    if p.ndim != 2 or y.shape != (len(p),) or len(p) == 0:
        raise ValueError("Expected NxK probabilities and N labels")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probabilities")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probabilities must sum to one")
    if not np.equal(y, y.astype(int)).all() or np.any((y < 0) | (y >= p.shape[1])):
        raise ValueError("Invalid labels")
    correct = p.argmax(axis=1) == y
    confidence = p.max(axis=1)
    return {"n": len(y), "accuracy": float(correct.mean()),
            "equal_mass": calibration(confidence, correct, scheme="equal_mass"),
            "equal_width": calibration(confidence, correct, scheme="equal_width")}


def summarize_corruptions(rows):
    """Mean accuracy and mean per-corruption ECE; pooled ECE is separate."""
    if not rows:
        raise ValueError("No corruption observations")
    return {"n_corruptions": len(rows),
            "mpc": float(np.mean([r["accuracy"] for r in rows])),
            "mean_corruption_ece_equal_mass": float(np.mean([r["equal_mass"]["ece"] for r in rows])),
            "mean_corruption_ece_equal_width": float(np.mean([r["equal_width"]["ece"] for r in rows]))}
