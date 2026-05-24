from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent


class MIRTHeads(nn.Module):
    def __init__(self, emb_dim: int = 768, K: int = 8,
                 n_subjects: int = 0, n_items: int = 0, use_intercepts: bool = False):
        super().__init__()
        self.use_intercepts = use_intercepts
        self.subj_proj = nn.Linear(emb_dim, K, bias=False)
        self.item_proj = nn.Linear(emb_dim, K, bias=False)
        nn.init.normal_(self.subj_proj.weight, std=0.05)
        nn.init.normal_(self.item_proj.weight, std=0.05)
        if use_intercepts:
            self.b_subj = nn.Embedding(n_subjects, 1)
            self.b_item = nn.Embedding(n_items, 1)
            nn.init.zeros_(self.b_subj.weight)
            nn.init.zeros_(self.b_item.weight)

    def forward(self, u_emb: torch.Tensor, v_emb: torch.Tensor,
                s_idx: torch.Tensor = None, i_idx: torch.Tensor = None) -> torch.Tensor:
        U = self.subj_proj(u_emb)
        V = self.item_proj(v_emb)
        logit = (U * V).sum(dim=-1)
        if self.use_intercepts and s_idx is not None:
            logit = logit + self.b_subj(s_idx).squeeze(-1) + self.b_item(i_idx).squeeze(-1)
        return logit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default=str(ROOT / "data" / "training_long.parquet"))
    parser.add_argument("--embeddings",
                        default=str(ROOT / "experiments/jbd-hier-mcmc/embeddings.pkl"))
    parser.add_argument("--out", default=str(ROOT / "submission_mcmc/mirt_heads.pt"))
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-intercepts", action="store_true",
                        help="Add per-subject and per-item intercepts (helps factors specialize).")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("loading data...")
    df = pd.read_parquet(args.data)
    df = df.loc[df.label.isin([0.0, 1.0])].copy()
    df["label"] = df["label"].astype(np.float32)
    print(f"  rows: {len(df):,}")

    print(f"loading embeddings from {args.embeddings}...")
    with open(args.embeddings, "rb") as f:
        emb = pickle.load(f)
    subj_to_idx = {s: i for i, s in enumerate(emb["subjects"])}
    item_to_idx = {it: i for i, it in enumerate(emb["items"])}
    subj_emb = torch.from_numpy(emb["subj_emb"]).to(args.device)
    item_emb = torch.from_numpy(emb["item_emb"]).to(args.device)
    print(f"  subj_emb {tuple(subj_emb.shape)}  item_emb {tuple(item_emb.shape)}")

    df["s_idx"] = df["subject_content"].map(subj_to_idx)
    df["i_idx"] = df["item_content"].map(item_to_idx)
    df = df.dropna(subset=["s_idx", "i_idx"]).reset_index(drop=True)
    print(f"  rows with both s & i emb: {len(df):,}")

    s_idx = torch.from_numpy(df["s_idx"].to_numpy().astype(np.int64)).to(args.device)
    i_idx = torch.from_numpy(df["i_idx"].to_numpy().astype(np.int64)).to(args.device)
    y = torch.from_numpy(df["label"].to_numpy()).to(args.device)

    rng = np.random.default_rng(args.seed)
    val_items = set(rng.choice(np.unique(df["i_idx"].to_numpy()),
                                size=int(df["i_idx"].nunique() * args.val_frac),
                                replace=False).tolist())
    val_mask = df["i_idx"].isin(val_items).to_numpy()
    val_idx = torch.from_numpy(np.where(val_mask)[0]).to(args.device)
    tr_idx = torch.from_numpy(np.where(~val_mask)[0]).to(args.device)
    print(f"  train rows: {len(tr_idx):,}  val rows: {len(val_idx):,}  "
          f"val items: {len(val_items):,}")

    model = MIRTHeads(emb_dim=768, K=args.K,
                      n_subjects=len(emb["subjects"]),
                      n_items=len(emb["items"]),
                      use_intercepts=args.use_intercepts).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    @torch.no_grad()
    def evaluate(idx):
        model.eval()
        chunks = []
        ys = []
        for chunk in idx.split(args.batch * 4):
            u = subj_emb[s_idx[chunk]]
            v = item_emb[i_idx[chunk]]
            si = s_idx[chunk] if args.use_intercepts else None
            ii = i_idx[chunk] if args.use_intercepts else None
            logit = model(u, v, si, ii)
            chunks.append(torch.sigmoid(logit).cpu())
            ys.append(y[chunk].cpu())
        p = torch.cat(chunks).numpy()
        yy = torch.cat(ys).numpy()
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        return float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())

    n = len(tr_idx)
    best_val = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        perm = tr_idx[torch.randperm(n, device=args.device)]
        total = 0.0; nb = 0
        for s in range(0, n, args.batch):
            batch_idx = perm[s:s + args.batch]
            u = subj_emb[s_idx[batch_idx]]
            v = item_emb[i_idx[batch_idx]]
            si = s_idx[batch_idx] if args.use_intercepts else None
            ii = i_idx[batch_idx] if args.use_intercepts else None
            logit = model(u, v, si, ii)
            loss = F.binary_cross_entropy_with_logits(logit, y[batch_idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        val_ll = evaluate(val_idx)
        marker = ""
        if val_ll < best_val:
            best_val = val_ll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"
        print(f"  epoch {epoch+1}/{args.epochs}  train_bce={total/nb:.4f}  "
              f"val_ll={val_ll:.4f}{marker}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"best val: {best_val:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "K": args.K,
        "emb_dim": 768,
        "val_log_loss": best_val,
        "subjects": emb["subjects"],
        "items": emb["items"],
    }, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
