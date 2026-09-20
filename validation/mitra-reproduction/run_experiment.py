#!/usr/bin/env python3
"""VGG19-BN / CIFAR-10 / 30% conv-weight pruning reconstruction.

Training has NOT been executed in the delivered environment. See REPORT.md.
Requires the referenced Liu repository; no guessed VGG is silently substituted.
"""
import argparse
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True,
                   help="Local clean checkout of Eric-mingjie/rethinking-network-pruning")
    p.add_argument("--upstream-revision", help="Expected full 40-character commit; required after dependency checks")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--cifar-c", type=Path, help="Extracted CIFAR-10-C directory containing .npy files")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--download", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--recovery-control", action="store_true",
                   help="Also fine-tune an unpruned clone for 40 epochs; diagnostic extension")
    p.add_argument("--preflight-only", action="store_true", help="Check real CUDA batch, loss and backward pass; do not train")
    return p.parse_args()


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    import torchvision
    from torch import nn
    from torch.nn.utils import prune
    from torch.utils.data import DataLoader, Dataset
    from torchvision import datasets, transforms
    from src.metrics import from_probabilities, summarize_corruptions

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU substitution performed")
    if args.seed < 0: raise ValueError("Seed must be nonnegative")
    if not args.upstream_revision or len(args.upstream_revision) != 40:
        raise ValueError("Supply --upstream-revision with a full commit recorded from the inspected checkout")
    upstream = args.upstream.resolve()
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != args.upstream_revision: raise RuntimeError("Upstream commit mismatch")
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True)
    if dirty.strip(): raise RuntimeError("Use an unmodified upstream checkout to preserve provenance")
    model_root = upstream / "cifar" / "weight-level"
    if not (model_root / "models").is_dir(): raise FileNotFoundError(model_root / "models")
    sys.path.insert(0, str(model_root))
    upstream_models = importlib.import_module("models.cifar")
    if not Path(upstream_models.__file__).resolve().is_relative_to(model_root):
        raise RuntimeError("Imported a conflicting models package")
    factory = getattr(upstream_models, "vgg19_bn")
    # This constructor comes from the cited upstream, not torchvision's ImageNet VGG.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"): torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)

    def make_model():
        m = factory(num_classes=10)
        if sum(isinstance(x, nn.Conv2d) for x in m.modules()) != 16:
            raise RuntimeError("Expected 16 conv layers in the selected VGG-19; inspect upstream")
        return m.to(device)

    model = make_model()
    manifest = {"status": "started", "utc": datetime.now(timezone.utc).isoformat(),
        "kind": "independent_reconstruction_not_authors_exact_run",
        "configuration": {"architecture": "upstream_vgg19_bn", "seed": args.seed,
            "batch_size": 64, "baseline_epochs": 160, "recovery_epochs": 40,
            "sparsity": 0.3, "momentum": 0.9, "weight_decay": 0.0001,
            "normalization_mean": [0.4914, 0.4822, 0.4465],
            "normalization_std": [0.2023, 0.1994, 0.2010],
            "augmentation": "RandomCrop(32,padding=4); RandomHorizontalFlip(p=0.5)",
            "baseline_lr_one_based": "1-79:0.1;80-119:0.01;120-160:0.001",
            "fine_tune_lr": 0.001, "checkpoint_selection": "last_epoch",
            "optimizer_reset_on_prune": True, "primary_ece": "equal_mass_10_stable_sorted_ties_split",
            "secondary_ece": "equal_width_10_right_closed",
            "recovery_control": args.recovery_control, "precision": "FP32_no_TF32_no_AMP"},
        "upstream": {"url": "https://github.com/Eric-mingjie/rethinking-network-pruning",
                     "revision": revision, "model_source_sha256": {
                         str(p.relative_to(model_root)): sha256(p)
                         for p in sorted((model_root / "models").rglob("*.py"))}},
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__},
        "device": str(device), "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "note": "Unspecified choices are listed in docs/assumptions.md; defaults are not asserted author settings."}
    old_path = args.out / "manifest.json"
    if old_path.exists():
        old = json.loads(old_path.read_text())
        if not args.resume: raise RuntimeError("Output already exists; use a new --out or --resume")
        if old["configuration"] != manifest["configuration"] or old["upstream"] != manifest["upstream"]:
            raise RuntimeError("Cannot resume with different configuration or upstream source")
        if old["versions"] != manifest["versions"]: raise RuntimeError("Cannot resume across package versions")
    write_json(old_path, manifest)
    (args.out / "model.txt").write_text(str(model) + "\n")
    normalize = transforms.Normalize(manifest["configuration"]["normalization_mean"],
                                     manifest["configuration"]["normalization_std"])
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_set = datasets.CIFAR10(args.data, train=True, download=args.download, transform=train_transform)
    test_set = datasets.CIFAR10(args.data, train=False, download=args.download, transform=test_transform)
    manifest["dataset_sha256"] = {str(p.relative_to(args.data)): sha256(p)
                                  for p in sorted((args.data / "cifar-10-batches-py").glob("*")) if p.is_file()}
    write_json(old_path, manifest)
    generator = torch.Generator().manual_seed(args.seed)
    # workers=0 is the reproducible default, and supports Windows spawn.
    if args.workers != 0: raise ValueError("This reference runner requires --workers 0 for checkpoint-exact RNG replay")
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0,
                              generator=generator, pin_memory=device.type == "cuda")
    clean_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)
    loss_fn = nn.CrossEntropyLoss()
    if args.preflight_only:
        x, y = next(iter(train_loader))
        model.train()
        loss = loss_fn(model(x.to(device)), y.to(device))
        loss.backward()
        if device.type == "cuda": torch.cuda.synchronize()
        write_json(args.out / "preflight.json", {"status": "passed_forward_backward_only", "loss": float(loss),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "benchmark_result": False})
        manifest["status"] = "preflight_only"; write_json(old_path, manifest)
        return

    def train_phase(m, phase, epochs, fixed_lr=None):
        optimizer = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        ckpt = args.out / "checkpoints" / (phase + "_last.pt")
        ckpt.parent.mkdir(exist_ok=True)
        start = 1
        if args.resume and ckpt.exists():
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            m.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            start = state["epoch"] + 1
            torch.set_rng_state(state["torch_rng"])
            generator.set_state(state["loader_rng"])
            random.setstate(state["python_rng"])
            if device.type == "cuda": torch.cuda.set_rng_state_all(state["cuda_rng"])
        for epoch in range(start, epochs + 1):
            lr = fixed_lr if fixed_lr is not None else (0.1 if epoch < 80 else 0.01 if epoch < 120 else 0.001)
            for group in optimizer.param_groups: group["lr"] = lr
            m.train(); n = 0; loss_sum = 0.0; correct = 0; t0 = time.perf_counter()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = m(x); loss = loss_fn(logits, y)
                loss.backward(); optimizer.step()
                n += len(y); loss_sum += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
            if device.type == "cuda": torch.cuda.synchronize()
            record = {"phase": phase, "epoch": epoch, "lr": lr, "loss": loss_sum / n,
                      "train_accuracy": correct / n, "examples": n, "seconds": time.perf_counter() - t0}
            # One atomic record per epoch avoids duplicate log lines on resume.
            write_json(args.out / "epochs" / f"{phase}_{epoch:03}.json", record)
            state = {"model": m.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                     "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                     "python_rng": random.getstate(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
            temp = ckpt.with_suffix(".tmp"); torch.save(state, temp); temp.replace(ckpt)
            print(json.dumps(record), flush=True)
        return m

    def predict(m, loader):
        m.eval(); probs = []; labels = []
        with torch.inference_mode():
            for x, y in loader:
                probs.append(torch.softmax(m(x.to(device)), dim=1).cpu().numpy())
                labels.append(y.numpy())
        return np.concatenate(probs), np.concatenate(labels)

    def evaluate(m, phase, loader, tag):
        probs, labels = predict(m, loader)
        directory = args.out / "predictions" / phase; directory.mkdir(parents=True, exist_ok=True)
        target = directory / (tag + ".npz")
        np.savez_compressed(target, probabilities=probs, labels=labels,
                            sample_index=np.arange(len(labels), dtype=np.int64))
        result = from_probabilities(probs, labels)
        result["prediction_sha256"] = sha256(target)
        write_json(args.out / "metrics" / phase / (tag + ".json"), result)
        return result

    model = train_phase(model, "baseline", 160)
    baseline_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    results = {"baseline": {"clean": evaluate(model, "baseline", clean_loader, "clean")}}
    if args.cifar_c:
        corruption_names = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
            "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
            "elastic_transform", "pixelate", "jpeg_compression"]
        class CorruptionSet(Dataset):
            def __init__(self, name, severity):
                self.images = np.load(args.cifar_c / (name + ".npy"), mmap_mode="r", allow_pickle=False)
                labels = np.load(args.cifar_c / "labels.npy", mmap_mode="r", allow_pickle=False)
                if self.images.shape != (50000, 32, 32, 3): raise ValueError("Unexpected CIFAR-C image shape")
                if len(labels) not in (10000, 50000): raise ValueError("Unexpected CIFAR-C label count")
                self.start = (severity - 1) * 10000
                self.labels = labels[:10000] if len(labels) == 10000 else labels[self.start:self.start + 10000]
                if not np.array_equal(self.labels, np.asarray(test_set.targets)):
                    raise ValueError("CIFAR-C labels do not match the clean test ordering")
            def __len__(self): return 10000
            def __getitem__(self, i):
                return test_transform(np.array(self.images[self.start + i], copy=True)), int(self.labels[i])
        manifest["cifar_c_sha256"] = {name: sha256(args.cifar_c / name)
            for name in ["labels.npy"] + [x + ".npy" for x in corruption_names]}
        write_json(old_path, manifest)

    def evaluate_corruptions(m, phase):
        summaries = {}
        for severity in range(1, 6):
            rows = []; pooled_p = []; pooled_y = []
            for name in corruption_names:
                loader = DataLoader(CorruptionSet(name, severity), batch_size=64, shuffle=False, num_workers=0)
                tag = f"{name}_s{severity}"
                row = evaluate(m, phase, loader, tag); rows.append(row)
                with np.load(args.out / "predictions" / phase / (tag + ".npz")) as z:
                    pooled_p.append(z["probabilities"]); pooled_y.append(z["labels"])
            summary = summarize_corruptions(rows)
            pooled = from_probabilities(np.concatenate(pooled_p), np.concatenate(pooled_y))
            summary["pooled_ece_equal_mass"] = pooled["equal_mass"]["ece"]
            summary["pooled_ece_equal_width"] = pooled["equal_width"]["ece"]
            summaries[str(severity)] = summary
        return summaries

    if args.cifar_c: results["baseline"]["corruptions_by_severity"] = evaluate_corruptions(model, "baseline")
    del model
    phases = [("pruned30_recovered", True)]
    if args.recovery_control: phases.append(("unpruned_extra40_control", False))
    for phase, apply_pruning in phases:
        m = make_model(); m.load_state_dict(baseline_state)
        if apply_pruning:
            params = [(layer, "weight") for layer in m.modules() if isinstance(layer, nn.Conv2d)]
            prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=0.3)
            total = sum(layer.weight_mask.numel() for layer, _ in params)
            zeros = sum((layer.weight_mask == 0).sum().item() for layer, _ in params)
            write_json(args.out / "pruning.json", {"method": "global_L1_conv_only_fixed_mask",
                "target_fraction": 0.3, "masked_weights": zeros, "eligible_weights": total,
                "achieved_fraction": zeros / total, "mask_tie_break": "torch_global_unstructured"})
            assert zeros == round(total * 0.3)
        m = train_phase(m, phase, 40, fixed_lr=0.001)
        results[phase] = {"clean": evaluate(m, phase, clean_loader, "clean")}
        if args.cifar_c: results[phase]["corruptions_by_severity"] = evaluate_corruptions(m, phase)
        if apply_pruning:
            for layer, _ in params:
                assert torch.count_nonzero(layer.weight.detach()[layer.weight_mask == 0]).item() == 0
        del m
    write_json(args.out / "results.json", results)
    manifest["status"] = "completed_reconstruction_clean_and_corruption" if args.cifar_c else "completed_reconstruction_clean_only"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    write_json(old_path, manifest)
    print("Completed reconstruction; numerical agreement must be assessed separately.")


if __name__ == "__main__":
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except Exception as exc:
        record = {"status": "failed_or_blocked", "utc": datetime.now(timezone.utc).isoformat(),
                  "exception": type(exc).__name__, "message": str(exc),
                  "traceback": traceback.format_exc(), "model_results": None,
                  "command": sys.argv}
        write_json(args.out / "failure.json", record)
        print(json.dumps(record, indent=2), file=sys.stderr)
        sys.exit(1)
