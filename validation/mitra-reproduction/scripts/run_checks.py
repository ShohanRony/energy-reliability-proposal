import io
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

out = Path("results/metric-checks")
out.mkdir(parents=True, exist_ok=True)
stream = io.StringIO()
suite = unittest.defaultTestLoader.discover("tests")
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
(out / "unittest.txt").write_text(stream.getvalue())
record = {"tests_discovered": result.testsRun, "passed": result.testsRun - len(result.skipped) - len(result.failures) - len(result.errors),
          "skipped": [{"test": str(t), "reason": s} for t, s in result.skipped],
          "failures": len(result.failures), "errors": len(result.errors),
          "cnn_pipeline_validated": False}
(out / "unittest.json").write_text(json.dumps(record, indent=2) + "\n")
print(stream.getvalue()); print(json.dumps(record, indent=2))
sys.exit(0 if result.wasSuccessful() else 1)
