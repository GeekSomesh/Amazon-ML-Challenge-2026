import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.environ.get("BER_DATA", os.path.join(ROOT, "dataset"))
WORK = os.environ.get("BER_WORK", os.path.join(ROOT, "work"))
OUTPUT = os.environ.get("BER_OUTPUT", os.path.join(ROOT, "output"))
os.makedirs(WORK, exist_ok=True)
N_THREADS = int(os.environ.get("BER_THREADS", os.cpu_count() or 8))


def norm_path(split, src):
    """'trainsim' = reduced Source-1 table + the train Source-2/3 pool (plus synthetic hard decoys
    when make_decoys.py has written trainsim-specific pool files)."""
    p = os.path.join(WORK, f"{split}_s{src}_norm.parquet")
    if split == "trainsim" and src != 1 and not os.path.exists(p):
        p = os.path.join(WORK, f"train_s{src}_norm.parquet")
    return p


def gt_path(split):
    return os.path.join(WORK, "trainsim_gt.parquet" if split == "trainsim" else "train_gt.parquet")
