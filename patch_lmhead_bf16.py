#!/usr/bin/env python3
"""Dequantize a checkpoint's FP8-quantized lm_head -> BF16 so vLLM can load it.

WHY: the unsloth NVFP4 checkpoints (`Qwen3.8-27B-NVFP4`, and `Qwen3.6-27B-NVFP4`
before it) quantize lm_head to FP8 (`lm_head.weight` = F8_E4M3 + per-row BF16
`lm_head.weight_scale`). vLLM 0.22.1's `Qwen3_5` builds an UNquantized
ParallelLMHead (weight only), so loading errors with
`ValueError: no module or parameter named 'lm_head.weight_scale'` -> engine-core
crash-loop. This converts lm_head.weight to BF16 (= fp8_weight * scale) and removes
the scale, IN PLACE in the HF cache snapshot. The snapshot files are symlinks to
blobs, so replacing them leaves the original FP8 blob on disk as a backup.

Idempotent: a no-op if already patched.
RE-RUN AFTER ANY RE-DOWNLOAD of the model — the fix lives in the cached files,
NOT in serve.sh, so clearing/refreshing the HF cache brings the crash back.

Qwen3.8 ships its weights as ONE 22.5GB shard, so the shard is rewritten by
streaming tensor bytes straight from the source file (peak RSS stays well under
1GB) rather than loading it all into memory. Needs ~24GB free disk for the new
copy; the original blob is kept, so budget ~47GB if you keep both.

Run with a torch+safetensors python that supports float8 — e.g. the vLLM image.
STOP THE vLLM CONTAINER FIRST: it holds ~30GB of the GPU and this wants a few GB.

  docker compose down
  docker run --rm --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v "$PWD/patch_lmhead_bf16.py:/patch.py:ro" \
    --entrypoint /app/.venv/bin/python vllm-qwen38-nvfp4:latest /patch.py

Then: docker compose up -d   (first boot recompiles ~2-3 min, then cached).

  argv[1] / $SNAPSHOT_DIR : snapshot dir to patch  (default: auto-find in HF cache)
  argv[2] / $MODEL_REPO   : HF repo id to auto-find (default: the value below)
"""
import glob, json, os, struct, sys, torch
from safetensors import safe_open

DEFAULT_REPO = "unsloth/Qwen3.8-27B-NVFP4"
ROWS_PER_CHUNK = 16384          # lm_head dequant block size; keeps peak memory tiny
COPY_CHUNK = 64 << 20           # 64MB streaming copy buffer


def find_snapshot(repo: str) -> str:
    hub = os.environ.get("HF_HOME", "/root/.cache/huggingface")
    folder = "models--" + repo.replace("/", "--")
    pat = os.path.join(hub, "hub", folder, "snapshots", "*")
    snaps = [p for p in glob.glob(pat) if os.path.isdir(p)]
    if not snaps:
        sys.exit(f"no {repo} snapshot found under {pat}")
    return sorted(snaps)[-1]


def read_header(path: str):
    """Return (header_dict, data_start_offset). Offsets in the header are relative
    to data_start."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


REPO = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MODEL_REPO", DEFAULT_REPO)
SNAP = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.environ.get("SNAPSHOT_DIR") or find_snapshot(REPO)
)

idx_path = os.path.join(SNAP, "model.safetensors.index.json")
idx = json.load(open(idx_path))
wm = idx["weight_map"]

if "lm_head.weight_scale" not in wm:
    print("lm_head.weight_scale not in index — already patched (or not quantized). Nothing to do.")
    sys.exit(0)

shard_name = wm["lm_head.weight"]
shard = os.path.join(SNAP, shard_name)
if wm["lm_head.weight_scale"] != shard_name:
    sys.exit("lm_head.weight and lm_head.weight_scale live in different shards; unsupported")
print("snapshot:", SNAP)
print("shard:", shard_name, "->", os.path.realpath(shard))

header, data_start = read_header(shard)
meta = header.get("__metadata__")

# --- dequantize lm_head, in row chunks so nothing large is ever materialized ---
with safe_open(shard, framework="pt") as f:
    scale = f.get_tensor("lm_head.weight_scale")
    slc = f.get_slice("lm_head.weight")
    V, H = slc.get_shape()
    print(f"before: lm_head.weight {header['lm_head.weight']['dtype']} {(V, H)} "
          f"| scale {scale.dtype} {tuple(scale.shape)}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("compute on:", dev)
    scale = scale.to(dev).to(torch.float32).reshape(-1, 1)
    out = torch.empty((V, H), dtype=torch.bfloat16)   # 2 bytes/elem, CPU
    for lo in range(0, V, ROWS_PER_CHUNK):
        hi = min(lo + ROWS_PER_CHUNK, V)
        blk = slc[lo:hi].to(dev).to(torch.float32) * scale[lo:hi]
        out[lo:hi] = blk.to(torch.bfloat16).cpu()
        del blk
if dev == "cuda":
    torch.cuda.empty_cache()
print("after : lm_head.weight torch.bfloat16", (V, H), "| scale removed")

# --- rebuild the shard header with new offsets, then stream the data through ----
# Keep tensors in their original on-disk order so the output stays contiguous.
names = sorted(
    (k for k in header if k != "__metadata__"),
    key=lambda k: header[k]["data_offsets"][0],
)
lm_nbytes = V * H * 2
new_header = {}
cursor = 0
plan = []          # (name, source) where source is ("copy", begin, end) or ("lm_head",)
for name in names:
    if name == "lm_head.weight_scale":
        continue
    e = header[name]
    if name == "lm_head.weight":
        size, dtype, src = lm_nbytes, "BF16", ("lm_head",)
    else:
        b, en = e["data_offsets"]
        size, dtype, src = en - b, e["dtype"], ("copy", data_start + b, data_start + en)
    new_header[name] = {"dtype": dtype, "shape": e["shape"], "data_offsets": [cursor, cursor + size]}
    plan.append((name, src))
    cursor += size
if meta is not None:
    new_header["__metadata__"] = meta

hdr = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
hdr += b" " * (-len(hdr) % 8)          # safetensors wants the data 8-byte aligned

tmp = shard + ".patched.tmp"
print(f"writing {tmp} ({(len(hdr) + 8 + cursor) / 1e9:.2f} GB)...")
with open(shard, "rb") as src, open(tmp, "wb") as dst:
    dst.write(struct.pack("<Q", len(hdr)))
    dst.write(hdr)
    for name, how in plan:
        if how[0] == "lm_head":
            mv = memoryview(out.contiguous().view(torch.uint8).reshape(-1).numpy())
            for off in range(0, lm_nbytes, COPY_CHUNK):
                dst.write(mv[off:off + COPY_CHUNK])
            del mv
        else:
            _, begin, end = how
            src.seek(begin)
            left = end - begin
            while left:
                buf = src.read(min(COPY_CHUNK, left))
                if not buf:
                    sys.exit(f"unexpected EOF reading {name}")
                dst.write(buf)
                left -= len(buf)
    dst.flush()
    os.fsync(dst.fileno())

# Sanity-check the rewritten shard before it replaces the original.
with safe_open(tmp, framework="pt") as f:
    keys = set(f.keys())
    assert "lm_head.weight_scale" not in keys, "scale still present"
    t = f.get_slice("lm_head.weight")
    assert tuple(t.get_shape()) == (V, H), "lm_head shape changed"
    assert len(keys) == len(plan), f"tensor count {len(keys)} != {len(plan)}"
print("verified rewritten shard:", len(keys), "tensors")

os.replace(tmp, shard)                       # replaces the snapshot symlink; FP8 blob kept as backup

wm.pop("lm_head.weight_scale", None)
delta = lm_nbytes - V * H * 1 - V * 2        # weight fp8(1B)->bf16(2B); remove bf16 scale
if idx.get("metadata", {}).get("total_size"):
    idx["metadata"]["total_size"] = int(idx["metadata"]["total_size"]) + delta
itmp = idx_path + ".patched.tmp"
json.dump(idx, open(itmp, "w"), indent=2)
os.replace(itmp, idx_path)
print(f"patched shard + index (total_size {delta:+d} B). DONE — start the vLLM container.")
