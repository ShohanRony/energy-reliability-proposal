import argparse, json, time, sys
import torch, torchvision
from torch import nn
from torchvision import transforms

p = argparse.ArgumentParser()
p.add_argument('--arch', choices=['resnet18', 'mobilenet_v3_small'], required=True)
p.add_argument('--data', required=True)
p.add_argument('--epochs', type=int, default=5)
p.add_argument('--out', required=True)
p.add_argument('--seed', type=int, default=1)
a = p.parse_args()

torch.manual_seed(a.seed)
torch.backends.cudnn.benchmark = False
device = "cuda"

if a.arch == 'resnet18':
    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
else:
    model = torchvision.models.mobilenet_v3_small(weights=None, num_classes=10)

model = model.to(device)

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
ds = torchvision.datasets.CIFAR10(a.data, train=True, download=False, transform=transform)
loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()

print(f"arch={a.arch} params={sum(p.numel() for p in model.parameters()):,}", flush=True)
torch.cuda.reset_peak_memory_stats()

results = []
for epoch in range(1, a.epochs + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.perf_counter()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    row = dict(arch=a.arch, epoch=epoch, loss=total_loss/total, train_accuracy=correct/total,
               examples=total, seconds=t1-t0,
               gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    results.append(row)
    print(json.dumps(row), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")

print("DONE", flush=True)
