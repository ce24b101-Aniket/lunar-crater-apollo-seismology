"""Download CC-BY-4.0 NASA-IBM SomBench NAC crater benchmark files."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, urllib.request

REPO = "nasa-ibm-ai4science/Sombench-NAC-Crater-Detection"
ROOT = Path("work/data/Sombench-NAC-Crater-Detection")
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true&expand=false"
with urllib.request.urlopen(API, timeout=45) as r:
    entries = json.load(r)
keep = []
for e in entries:
    p = e.get("path", "")
    if e.get("type") != "file":
        continue
    if p == "README.md" or p in {"annotations_train.json", "annotations_val.json", "annotations_test.json", "splits.json"} or p.startswith("images/"):
        keep.append((p, int(e.get("size", 0))))
total = sum(s for _, s in keep)
print(f"Downloading {len(keep)} files ({total/1e6:.1f} MB) from {REPO}", flush=True)

def fetch(item):
    path, size = item
    dest = ROOT / path
    if dest.exists() and dest.stat().st_size == size:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Codex-lunar-research/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if tmp.stat().st_size != size:
        tmp.unlink(missing_ok=True)
        raise IOError(f"Size mismatch for {path}: expected {size}")
    tmp.replace(dest)
    return size

done = 0
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(fetch, x) for x in keep]
    for f in as_completed(futures):
        done += f.result()
        if done and done % (10 * 1024 * 1024) < 5 * 1024 * 1024:
            print(f"Downloaded about {done/1e6:.0f} MB", flush=True)
print(f"Dataset ready at {ROOT}; {len(keep)} files, {total/1e6:.1f} MB", flush=True)
