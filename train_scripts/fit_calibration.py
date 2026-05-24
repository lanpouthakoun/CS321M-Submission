
from __future__ import annotations

import argparse
import importlib.util
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parent.parent


def load_module(sub_dir: Path):
    spec = importlib.util.spec_from_file_location("sub_model_cal", sub_dir / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(sub_dir))
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod


def cold_start_item_mask(items: pd.Series, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    unique = items.unique()
    n_val = int(round(len(unique) * val_frac))
    val_items = set(rng.choice(unique, size=n_val, replace=False).tolist())
    return items.isin(val_items).to_numpy()


def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    best_tau = 1.0; best_ll = float("inf")
    for tau in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                1.0, 1.1, 1.2, 1.5, 2.0, 3.0]:
        p = 1.0 / (1.0 + np.exp(-tau * z))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        if ll < best_ll:
            best_ll = ll; best_tau = tau
    return best_tau, best_ll


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-dir", required=True, type=Path)
    parser.add_argument("--data", default=str(ROOT / "data" / "training_long.parquet"))
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-per-bench", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("loading data...")
    df = pd.read_parquet(args.data)
    df = df.loc[df.label.isin([0.0, 1.0])].copy()
    df["label"] = df["label"].astype(np.float32)
    val_mask = cold_start_item_mask(df.item_content, args.val_frac, args.seed)
    df_va = df.loc[val_mask].reset_index(drop=True)
    print(f"  cold-start val rows: {len(df_va):,}")

    sampled = []
    for bench, g in df_va.groupby("benchmark"):
        if len(g) > args.max_per_bench:
            g = g.sample(n=args.max_per_bench, random_state=args.seed)
        sampled.append(g)
    df_va = pd.concat(sampled).reset_index(drop=True)
    print(f"  capped val rows: {len(df_va):,}")

    print(f"loading {args.sub_dir}...")
    M = load_module(args.sub_dir)

    M._clamp = lambda p, lo=0.0, hi=1.0: float(max(1e-6, min(1 - 1e-6, p)))

    print("running predict over val rows...")
    t0 = time.time()
    preds = np.empty(len(df_va), dtype=np.float32)
    for i, r in df_va.iterrows():
        p = M.predict({
            "subject_content": r.subject_content,
            "item_content": r.item_content,
            "benchmark": r.benchmark,
            "condition": r.condition,
        })
        preds[i] = p
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(df_va)}")
    print(f"  done in {time.time() - t0:.1f}s")

    z = safe_logit(preds)
    y = df_va.label.to_numpy()


    print(f"\n=== Per-benchmark temperature scaling ===")
    temp_calib: dict[str, dict] = {}
    print(f"{'bench':<20s}  {'n':>5s}  {'raw_ll':>8s}  {'tau':>5s}  {'cal_ll':>8s}")
    for bench, g in df_va.groupby("benchmark"):
        idx = g.index.to_numpy()
        zb = z[idx]; yb = y[idx]
        if len(zb) < 50 or yb.std() < 0.01:
            continue
        raw_ll = log_loss(yb, np.clip(preds[idx], 1e-6, 1 - 1e-6), labels=[0, 1])
        tau, ll = fit_temperature(zb, yb)
        temp_calib[bench] = {"tau": float(tau), "raw_ll": float(raw_ll), "cal_ll": float(ll), "n": int(len(zb))}
        print(f"{bench:<20s}  {len(zb):>5d}  {raw_ll:>8.4f}  {tau:>5.2f}  {ll:>8.4f}")

    tau_g, ll_g = fit_temperature(z, y)
    temp_calib["__global__"] = {"tau": float(tau_g), "raw_ll": float(log_loss(y, np.clip(preds, 1e-6, 1-1e-6))), "cal_ll": float(ll_g), "n": int(len(z))}
    print(f"{'__global__':<20s}  {len(z):>5d}  -        {tau_g:>5.2f}  {ll_g:>8.4f}")

    with open(args.sub_dir / "temp_calib.pkl", "wb") as f:
        pickle.dump(temp_calib, f)
    print(f"saved {args.sub_dir / 'temp_calib.pkl'}")


    print(f"\n=== Per-benchmark isotonic ===")
    iso_calib: dict[str, dict] = {}
    print(f"{'bench':<20s}  {'n':>5s}  {'raw_ll':>8s}  {'iso_ll':>8s}")
    for bench, g in df_va.groupby("benchmark"):
        idx = g.index.to_numpy()
        pb = preds[idx]; yb = y[idx]
        if len(pb) < 50 or yb.std() < 0.01:
            continue
        try:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(pb, yb)
            p_iso = iso.transform(pb)
            raw_ll = log_loss(yb, np.clip(pb, 1e-6, 1 - 1e-6), labels=[0, 1])
            iso_ll = log_loss(yb, np.clip(p_iso, 1e-6, 1 - 1e-6), labels=[0, 1])
        except Exception as e:
            print(f"{bench}: iso fit failed: {e}")
            continue

        grid = np.linspace(0.0, 1.0, 1001)
        cal_grid = iso.transform(grid)
        iso_calib[bench] = {"grid": grid, "cal_grid": cal_grid,
                            "raw_ll": float(raw_ll), "iso_ll": float(iso_ll),
                            "n": int(len(pb))}
        print(f"{bench:<20s}  {len(pb):>5d}  {raw_ll:>8.4f}  {iso_ll:>8.4f}")


    iso_g = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_g.fit(preds, y)
    grid = np.linspace(0.0, 1.0, 1001)
    iso_calib["__global__"] = {"grid": grid, "cal_grid": iso_g.transform(grid),
                                "raw_ll": float(log_loss(y, np.clip(preds, 1e-6, 1-1e-6))),
                                "iso_ll": float(log_loss(y, np.clip(iso_g.transform(preds), 1e-6, 1-1e-6))),
                                "n": int(len(preds))}

    with open(args.sub_dir / "iso_calib.pkl", "wb") as f:
        pickle.dump(iso_calib, f)
    print(f"saved {args.sub_dir / 'iso_calib.pkl'}")


    print(f"\n=== Summary (per-row val log-loss) ===")
    print(f"raw       : {log_loss(y, np.clip(preds, 1e-6, 1-1e-6)):.4f}")
    print(f"global τ  : {ll_g:.4f}")
    print(f"per-bench τ (weighted): {sum(c.get('cal_ll', c.get('raw_ll',0))*c['n'] for c in temp_calib.values() if 'tau' in c) / sum(c['n'] for c in temp_calib.values() if 'tau' in c):.4f}")
    print(f"global iso: {log_loss(y, np.clip(iso_g.transform(preds), 1e-6, 1-1e-6)):.4f}")


if __name__ == "__main__":
    main()
