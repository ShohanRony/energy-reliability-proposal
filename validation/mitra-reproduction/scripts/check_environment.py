"""Record availability, not credentials or the full environment."""
import argparse
import concurrent.futures
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def probe(url):
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            r.read(1)
            return {"url": url, "status": r.status, "ok": True}
    except Exception as e:
        return {"url": url, "ok": False, "exception": type(e).__name__, "message": str(e)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/execution-attempt/environment.json")
    p.add_argument("--network", action="store_true")
    args = p.parse_args()
    r = {"utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
         "python": sys.version, "cpu_count": os.cpu_count(),
         "nvidia_smi_available": bool(shutil.which("nvidia-smi")),
         "modules": {m: importlib.util.find_spec(m) is not None for m in ("numpy", "torch", "torchvision")},
         "is_user_rtx3050": False}
    if Path("/proc/meminfo").exists():
        r["mem_total"] = Path("/proc/meminfo").read_text().splitlines()[0]
    if r["modules"]["torch"]:
        import torch
        r.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available())
    if r["nvidia_smi_available"]:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True)
        r["gpu_query"] = q.stdout
    if args.network:
        urls = ["https://arxiv.org/pdf/2405.20876", "https://pypi.org/simple/torch/",
                "https://github.com/Eric-mingjie/rethinking-network-pruning",
                "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"]
        with concurrent.futures.ThreadPoolExecutor(4) as ex: r["network"] = list(ex.map(probe, urls))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2))


if __name__ == "__main__": main()
