
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data", default=str(ROOT / "data" / "training_long.parquet"))
    parser.add_argument("--encoder",
                        default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("loading data...", flush=True)
    df = pd.read_parquet(args.data)
    df = df.loc[df.label.isin([0.0, 1.0])].copy()
    subjects = sorted(df.subject_content.dropna().unique().tolist())
    items = sorted(df.item_content.dropna().unique().tolist())
    print(f"  unique subjects: {len(subjects)}", flush=True)
    print(f"  unique items:    {len(items)}", flush=True)

    print(f"loading encoder {args.encoder} on {args.device}...", flush=True)
    from sentence_transformers import SentenceTransformer
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("  cuda not available, falling back to cpu", flush=True)
        args.device = "cpu"
    enc = SentenceTransformer(args.encoder, device=args.device)
    enc.eval()

    t0 = time.time()
    print(f"encoding {len(subjects)} subjects...", flush=True)
    s_emb = enc.encode(subjects, batch_size=args.batch_size,
                       convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False).astype(np.float32)
    print(f"  done in {time.time() - t0:.1f}s  shape={s_emb.shape}", flush=True)

    t0 = time.time()
    print(f"encoding {len(items)} items...", flush=True)
    i_emb = enc.encode(items, batch_size=args.batch_size,
                       convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False).astype(np.float32)
    print(f"  done in {time.time() - t0:.1f}s  shape={i_emb.shape}", flush=True)

    out = {
        "encoder": args.encoder,
        "subjects": subjects,
        "items": items,
        "subj_emb": s_emb,
        "item_emb": i_emb,
    }
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    size = args.out.stat().st_size / 1024 / 1024
    print(f"saved {args.out} ({size:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
