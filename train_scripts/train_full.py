from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score

from torch_measure.fitting.mle import mle_fit
from torch_measure.models import TwoPL

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "training_long.parquet"
DEFAULT_SUBMISSION_DIR = ROOT / "submission_full"

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

def load_long_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"{path} not found - running fetch_data first...")
        from fetch_data import build_training_table
        return build_training_table(path)
    return pd.read_parquet(path)


def to_binary(df: pd.DataFrame, likert_threshold: float = None) -> pd.DataFrame:
    if likert_threshold is None:
        keep = df["label"].isin([0.0, 1.0])
        dropped = (~keep).sum()
        if dropped:
            print(f"Dropping {dropped:,} non-binary label rows ({dropped / len(df):.2%})")
        out = df.loc[keep].copy()
        out["label"] = out["label"].astype(np.float32)
        return out


    keep = df["label"].notna() & (df["label"] >= 0) & (df["label"] <= 10)
    out = df.loc[keep].copy()
    binary_mask = out["label"].isin([0.0, 1.0])
    nonbin = (~binary_mask).sum()
    if nonbin:

        bin_labels = out["label"].copy()
        bin_labels.loc[~binary_mask] = (out.loc[~binary_mask, "label"] >= likert_threshold).astype(np.float32)
        out["label"] = bin_labels.astype(np.float32)
        print(f"Likert binarization @ {likert_threshold}: converted {nonbin:,} non-binary rows ({nonbin / len(df):.2%})")
    else:
        out["label"] = out["label"].astype(np.float32)
    return out


class NCFHead(nn.Module):
    def __init__(self, d: int, hidden: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [nn.Linear(2 * d, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([u, v], dim=-1)).squeeze(-1)


class ItemParamHead(nn.Module):
    def __init__(self, emb_dim: int, n_benchmarks: int, n_conditions: int,
                 hidden: int = 256, dropout: float = 0.0,
                 out_dim: int = 2) -> None:
        super().__init__()
        in_dim = emb_dim + n_benchmarks + n_conditions
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def encode_unique(encoder, texts: list[str], batch_size: int) -> np.ndarray:
    return encoder.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)

def cold_start_item_mask(items: pd.Series, val_frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique = items.unique()
    n_val = int(round(len(unique) * val_frac))
    val_items = set(rng.choice(unique, size=n_val, replace=False).tolist())
    return items.isin(val_items).to_numpy()


@torch.no_grad()
def ncf_probs(model: NCFHead, U: torch.Tensor, V: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, U.shape[0], batch_size):
        logits = model(U[s:s + batch_size], V[s:s + batch_size])
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)

def train_ncf(
    U_t: torch.Tensor, V_t: torch.Tensor, y_t: torch.Tensor,
    val_mask: np.ndarray, device: str, epochs: int, lr: float,
    weight_decay: float, batch_size: int, seed: int,
    dropout: float = 0.0, hidden: int = 256,
) -> tuple[NCFHead, dict]:

    import copy

    torch.manual_seed(seed)
    d = U_t.shape[1]
    train_idx = torch.from_numpy(np.where(~val_mask)[0]).to(device)
    val_idx = torch.from_numpy(np.where(val_mask)[0]).to(device)
    print(f"[NCF] train rows: {train_idx.numel():,} | "
          f"val rows (cold-start items): {val_idx.numel():,}")

    model = NCFHead(d, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = train_idx.numel()

    best_ll = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_auc = float("nan")

    last_ll, last_auc = float("nan"), float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(n_train, device=device)]
        running = 0.0
        nb = 0
        for s in range(0, n_train, batch_size):
            idx = perm[s:s + batch_size]
            logits = model(U_t[idx], V_t[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        p_val = np.clip(
            ncf_probs(model, U_t[val_idx], V_t[val_idx], 4096),
            1e-6, 1 - 1e-6,
        )
        y_val = y_t[val_idx].cpu().numpy()
        last_ll = float(log_loss(y_val, p_val))
        try:
            last_auc = float(roc_auc_score(y_val, p_val))
        except ValueError:
            last_auc = float("nan")
        marker = ""
        if last_ll < best_ll:
            best_ll = last_ll
            best_auc = last_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            marker = " *"
        print(f"[NCF] epoch {epoch:2d}/{epochs}  train_bce={running / nb:.4f}  "
              f"val_log_loss={last_ll:.4f}  val_auc={last_auc:.4f}{marker}")

    print(f"[NCF] restoring best checkpoint from epoch {best_epoch} "
          f"(val_log_loss={best_ll:.4f}, val_auc={best_auc:.4f})")
    model.load_state_dict(best_state)
    return model, {
        "ncf_val_log_loss": best_ll, "ncf_val_auc": best_auc,
        "ncf_best_epoch": best_epoch, "ncf_last_val_log_loss": last_ll,
    }

def fit_twopl(
    df: pd.DataFrame, device: str, max_epochs: int, lr: float, seed: int,
) -> tuple[TwoPL, dict[str, int], dict[tuple[str, str, str], int], dict]:
    subject_keys = df["subject_name"].tolist()
    item_keys = list(zip(df["item_id"].tolist(),
                         df["benchmark"].tolist(),
                         df["condition"].tolist()))

    subject_to_idx: dict[str, int] = {}
    for k in subject_keys:
        subject_to_idx.setdefault(k, len(subject_to_idx))
    item_to_idx: dict[tuple[str, str, str], int] = {}
    for k in item_keys:
        item_to_idx.setdefault(k, len(item_to_idx))

    s = torch.tensor([subject_to_idx[k] for k in subject_keys],
                     dtype=torch.long, device=device)
    i = torch.tensor([item_to_idx[k] for k in item_keys],
                     dtype=torch.long, device=device)
    y = torch.tensor(df["label"].to_numpy(), dtype=torch.float32, device=device)

    model = TwoPL(n_subjects=len(subject_to_idx),
                  n_items=len(item_to_idx),
                  device=device)
    history = mle_fit(model, s, i, y, max_epochs=max_epochs, lr=lr, verbose=True)
    print(f"[IRT] final NLL: {history['losses'][-1]:.4f}")
    return model, subject_to_idx, item_to_idx, {
        "irt_train_nll": float(history["losses"][-1]),
    }

def fit_item_head(
    df: pd.DataFrame,
    item_to_idx: dict[tuple[str, str, str], int],
    item_text_emb: np.ndarray,
    item_text_lookup: dict[str, int],
    targets: np.ndarray,
    device: str, epochs: int, lr: float, weight_decay: float,
    batch_size: int, seed: int, val_frac: float,
    dropout: float = 0.0, patience: int = 0,
) -> tuple[ItemParamHead, dict[str, int], dict[str, int], dict]:
    import copy

    df = df[df.apply(
        lambda r: (r["item_id"], r["benchmark"], r["condition"]) in item_to_idx,
        axis=1,
    )]
    rep = df.drop_duplicates(subset=["item_id", "benchmark", "condition"]).copy()
    rep["item_idx"] = [
        item_to_idx[(r.item_id, r.benchmark, r.condition)]
        for r in rep.itertuples(index=False)
    ]
    rep = rep.sort_values("item_idx").reset_index(drop=True)

    benchmarks = sorted(df["benchmark"].unique().tolist())
    conditions = sorted(df["condition"].unique().tolist())
    benchmark_to_idx = {b: i for i, b in enumerate(benchmarks)}
    condition_to_idx = {c: i for i, c in enumerate(conditions)}

    embeddings = item_text_emb[
        rep["item_content"].fillna("").astype(str).map(item_text_lookup).to_numpy()
    ]
    b_oh = np.eye(len(benchmarks), dtype=np.float32)[
        rep["benchmark"].map(benchmark_to_idx).to_numpy()
    ]
    c_oh = np.eye(len(conditions), dtype=np.float32)[
        rep["condition"].map(condition_to_idx).to_numpy()
    ]
    X = np.concatenate([embeddings, b_oh, c_oh], axis=1).astype(np.float32)

    torch.manual_seed(seed)
    n = X.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(round(n * val_frac))
    val_pos, train_pos = perm[:n_val], perm[n_val:]

    X_train = torch.from_numpy(X[train_pos]).to(device)
    y_train = torch.from_numpy(targets[train_pos].astype(np.float32)).to(device)
    X_val = torch.from_numpy(X[val_pos]).to(device)
    y_val = torch.from_numpy(targets[val_pos].astype(np.float32)).to(device)

    head = ItemParamHead(embeddings.shape[1],
                         len(benchmarks), len(conditions),
                         dropout=dropout).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = X_train.shape[0]

    best_mse = float("inf")
    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0
    last_mse = float("nan")
    no_improve_epochs = 0
    for epoch in range(1, epochs + 1):
        head.train()
        idx = torch.randperm(n_train, device=device)
        running = 0.0
        nb = 0
        for s in range(0, n_train, batch_size):
            b_idx = idx[s:s + batch_size]
            pred = head(X_train[b_idx])
            loss = F.mse_loss(pred, y_train[b_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        head.eval()
        with torch.no_grad():
            last_mse = float(F.mse_loss(head(X_val), y_val).item())
        marker = ""
        if last_mse < best_mse:
            best_mse = last_mse
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            marker = " *"
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            print(f"[ItemHead] epoch {epoch:2d}/{epochs}  "
                  f"train_mse={running / nb:.4f}  val_mse={last_mse:.4f}{marker}")
        if patience > 0 and no_improve_epochs >= patience:
            print(f"[ItemHead] early stop at epoch {epoch} "
                  f"(no improvement for {no_improve_epochs} epochs)")
            break

    print(f"[ItemHead] restoring best checkpoint from epoch {best_epoch} "
          f"(val_mse={best_mse:.4f})")
    head.load_state_dict(best_state)
    return head, benchmark_to_idx, condition_to_idx, {
        "item_head_val_mse": best_mse,
        "item_head_best_epoch": best_epoch,
        "item_head_last_val_mse": last_mse,
    }

def fit_item_head_bce(
    df_train: pd.DataFrame,
    val_mask: np.ndarray,
    df_all: pd.DataFrame,
    item_text_emb: np.ndarray,
    item_text_lookup: dict[str, int],
    ability_by_name: dict[str, float],
    device: str, epochs: int, lr: float, weight_decay: float,
    batch_size: int, seed: int,
    dropout: float = 0.0, patience: int = 0,
    bench_cond_dropout: float = 0.0,
    use_3pl: bool = False,
    hierarchical: bool = False,
    soft_labels: dict[int, float] | None = None,
    aux_df: pd.DataFrame | None = None,
    aux_weight: float = 0.3,
) -> tuple[ItemParamHead, dict[str, int], dict[str, int], dict]:

    import copy

    benchmarks = sorted(df_train["benchmark"].unique().tolist())
    conditions = sorted(df_train["condition"].unique().tolist())
    benchmark_to_idx = {b: i for i, b in enumerate(benchmarks)}
    condition_to_idx = {c: i for i, c in enumerate(conditions)}
    n_bench = len(benchmarks)
    n_cond = len(conditions)

    def _build_xy(df: pd.DataFrame):
        prior_theta = float(np.mean(list(ability_by_name.values())))\
            if ability_by_name else 0.0
        item_idx = df["item_content"].fillna("").astype(str).map(item_text_lookup).to_numpy()

        keep = item_idx != None
        if not np.all(keep):
            df = df.loc[keep].reset_index(drop=True)
            item_idx = df["item_content"].fillna("").astype(str).map(item_text_lookup).to_numpy()
        item_idx = item_idx.astype(np.int64)
        b_idx = df["benchmark"].map(lambda b: benchmark_to_idx.get(b, 0)).to_numpy().astype(np.int64)
        c_idx = df["condition"].map(lambda c: condition_to_idx.get(c, 0)).to_numpy().astype(np.int64)
        theta = df["subject_name"].map(lambda n: ability_by_name.get(n, prior_theta)).to_numpy().astype(np.float32)
        y = df["label"].to_numpy().astype(np.float32)
        return item_idx, b_idx, c_idx, theta, y

    print(f"[ItemHeadBCE] vocab: {n_bench} benchmarks, {n_cond} conditions")
    it_train, bi_train, ci_train, th_train, y_train = _build_xy(df_train)
    val_df = df_all.loc[val_mask].reset_index(drop=True)
    val_df = val_df[
        val_df["benchmark"].isin(benchmark_to_idx) &
        val_df["condition"].isin(condition_to_idx)
    ].reset_index(drop=True)
    it_val, bi_val, ci_val, th_val, y_val = _build_xy(val_df)

    print(f"[ItemHeadBCE] train rows: {len(y_train):,}  val rows: {len(y_val):,}")

    V_all = torch.from_numpy(item_text_emb).to(device)
    d_emb = V_all.shape[1]
    B_eye = torch.eye(n_bench, device=device)
    C_eye = torch.eye(n_cond, device=device)

    it_train_t = torch.from_numpy(it_train).to(device)
    bi_train_t = torch.from_numpy(bi_train).to(device)
    ci_train_t = torch.from_numpy(ci_train).to(device)
    th_train_t = torch.from_numpy(th_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)

    it_val_t = torch.from_numpy(it_val).to(device)
    bi_val_t = torch.from_numpy(bi_val).to(device)
    ci_val_t = torch.from_numpy(ci_val).to(device)
    th_val_t = torch.from_numpy(th_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    torch.manual_seed(seed)
    out_dim = 3 if use_3pl else 2
    head = ItemParamHead(d_emb, n_bench, n_cond,
                         dropout=dropout, out_dim=out_dim).to(device)
    params = list(head.parameters())
    mu_b = mu_a = mu_c_logit = None
    if hierarchical:
        mu_b = nn.Parameter(torch.zeros(n_bench, device=device))
        mu_a = nn.Parameter(torch.zeros(n_bench, device=device))
        if use_3pl:
            mu_c_logit = nn.Parameter(torch.full((n_bench,), -2.9, device=device))
        params = list(head.parameters())
        opt = torch.optim.Adam([
            {"params": params, "weight_decay": weight_decay},
            {"params": [p for p in [mu_b, mu_a, mu_c_logit] if p is not None],
             "weight_decay": 0.0},
        ], lr=lr)
    else:
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    n_train = it_train_t.numel()

    def _features(it, bi, ci, training: bool = False):
        v = V_all[it]
        b = B_eye[bi]
        c = C_eye[ci]
        if training and bench_cond_dropout > 0:
            mask = (torch.rand(it.shape[0], 1, device=it.device) > bench_cond_dropout).float()
            b = b * mask
            c = c * mask
        return torch.cat([v, b, c], dim=-1)

    def _forward(it, bi, ci, theta, training: bool = False):

        feats = _features(it, bi, ci, training=training)
        out = head(feats)
        b_pred = out[:, 0]
        log_a_pred = out[:, 1]
        if hierarchical:
            b_pred = b_pred + mu_b[bi]
            log_a_pred = log_a_pred + mu_a[bi]
        a_pred = torch.exp(log_a_pred.clamp(-3.0, 3.0))
        z = a_pred * (theta - b_pred)
        if use_3pl:
            c_logit = out[:, 2]
            if hierarchical:
                c_logit = c_logit + mu_c_logit[bi]
            c = torch.sigmoid(c_logit)
            p = c + (1.0 - c) * torch.sigmoid(z)
            p = p.clamp(1e-6, 1.0 - 1e-6)
            logit = torch.log(p / (1.0 - p))
            return logit, p
        else:
            return z, torch.sigmoid(z)

    def _val_logloss():
        head.eval()
        with torch.no_grad():
            out_chunks = []
            for s in range(0, it_val_t.numel(), 16384):
                e = s + 16384
                _, p = _forward(it_val_t[s:e], bi_val_t[s:e],
                                ci_val_t[s:e], th_val_t[s:e])
                out_chunks.append(p.cpu().numpy())
            p_val = np.clip(np.concatenate(out_chunks), 1e-6, 1 - 1e-6)
        return float(log_loss(y_val, p_val))

    aux_tensors = None
    if aux_df is not None and len(aux_df) > 0:
        aux_df_kept = aux_df[
            aux_df["benchmark"].isin(benchmark_to_idx) &
            aux_df["condition"].isin(condition_to_idx)
        ].reset_index(drop=True)
        ai, ab, ac, ath, ay = _build_xy(aux_df_kept)
        aux_tensors = (
            torch.from_numpy(ai).to(device),
            torch.from_numpy(ab).to(device),
            torch.from_numpy(ac).to(device),
            torch.from_numpy(ath).to(device),
            torch.from_numpy(ay).to(device),
        )
        print(f"[ItemHeadBCE] aux rows: {len(ay):,} (weight {aux_weight})")

    best_ll = float("inf")
    best_state = copy.deepcopy(head.state_dict())
    best_aux_state = None
    best_epoch = 0
    no_improve = 0
    last_ll = float("nan")
    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n_train, device=device)
        running = 0.0
        nb = 0
        for s in range(0, n_train, batch_size):
            idx = perm[s:s + batch_size]
            logit, _ = _forward(it_train_t[idx], bi_train_t[idx],
                                ci_train_t[idx], th_train_t[idx], training=True)
            loss = F.binary_cross_entropy_with_logits(logit, y_train_t[idx])
            if aux_tensors is not None:
                a_idx = torch.randint(0, aux_tensors[0].numel(),
                                      (idx.numel(),), device=device)
                ai_t, ab_t, ac_t, ath_t, ay_t = aux_tensors
                logit_aux, _ = _forward(ai_t[a_idx], ab_t[a_idx],
                                        ac_t[a_idx], ath_t[a_idx], training=True)
                loss_aux = F.binary_cross_entropy_with_logits(logit_aux, ay_t[a_idx])
                loss = loss + aux_weight * loss_aux
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        last_ll = _val_logloss()
        marker = ""
        if last_ll < best_ll:
            best_ll = last_ll
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            if hierarchical:
                best_aux_state = (
                    mu_b.detach().clone(),
                    mu_a.detach().clone(),
                    mu_c_logit.detach().clone() if mu_c_logit is not None else None,
                )
            marker = " *"
            no_improve = 0
        else:
            no_improve += 1
        print(f"[ItemHeadBCE] epoch {epoch:2d}/{epochs}  "
              f"train_bce={running/nb:.4f}  val_log_loss={last_ll:.4f}{marker}")
        if patience > 0 and no_improve >= patience:
            print(f"[ItemHeadBCE] early stop at epoch {epoch} (no improvement {no_improve})")
            break

    print(f"[ItemHeadBCE] restoring best checkpoint from epoch {best_epoch} "
          f"(val_log_loss={best_ll:.4f})")
    head.load_state_dict(best_state)
    if hierarchical and best_aux_state is not None:
        mu_b.data.copy_(best_aux_state[0])
        mu_a.data.copy_(best_aux_state[1])
        if best_aux_state[2] is not None:
            mu_c_logit.data.copy_(best_aux_state[2])
    extras = {
        "item_head_val_log_loss": best_ll,
        "item_head_best_epoch": best_epoch,
        "item_head_last_val_log_loss": last_ll,
        "item_head_mode": "joint_bce_3pl" if use_3pl else "joint_bce",
        "use_3pl": use_3pl,
        "hierarchical": hierarchical,
    }
    if hierarchical:
        extras["mu_b"] = mu_b.detach().cpu().numpy().tolist()
        extras["mu_a"] = mu_a.detach().cpu().numpy().tolist()
        if mu_c_logit is not None:
            extras["mu_c_logit"] = mu_c_logit.detach().cpu().numpy().tolist()
    return head, benchmark_to_idx, condition_to_idx, extras

@torch.no_grad()
def cold_start_compare(
    df: pd.DataFrame, val_mask: np.ndarray,
    ncf: NCFHead, U_t: torch.Tensor, V_t: torch.Tensor,
    head: ItemParamHead, item_text_emb: np.ndarray,
    item_text_lookup: dict[str, int],
    benchmark_to_idx: dict[str, int],
    condition_to_idx: dict[str, int],
    ability_by_name: dict[str, float],
    base_rate: float,
    device: str,
    ncf_weight: float = 0.3,
    sb_means: dict[tuple[str, str], float] | None = None,
    use_3pl: bool = False,
    hier_mu_b: np.ndarray | None = None,
    hier_mu_a: np.ndarray | None = None,
    hier_mu_c_logit: np.ndarray | None = None,
) -> dict:
    val_df = df.loc[val_mask].copy()
    if val_df.empty:
        return {}
    y = val_df["label"].to_numpy().astype(np.float32)


    val_idx_t = torch.from_numpy(np.where(val_mask)[0]).to(device)
    ncf.eval()
    p_ncf = []
    for s in range(0, val_idx_t.numel(), 4096):
        chunk = val_idx_t[s:s + 4096]
        p_ncf.append(torch.sigmoid(ncf(U_t[chunk], V_t[chunk])).cpu().numpy())
    p_ncf = np.clip(np.concatenate(p_ncf), 1e-6, 1 - 1e-6)

    item_emb = item_text_emb[
        val_df["item_content"].fillna("").astype(str).map(item_text_lookup).to_numpy()
    ]
    b_oh = np.eye(len(benchmark_to_idx), dtype=np.float32)[
        val_df["benchmark"].map(lambda b: benchmark_to_idx.get(b, 0)).to_numpy()
    ]
    c_oh = np.eye(len(condition_to_idx), dtype=np.float32)[
        val_df["condition"].map(lambda c: condition_to_idx.get(c, 0)).to_numpy()
    ]
    X = np.concatenate([item_emb, b_oh, c_oh], axis=1).astype(np.float32)
    head.eval()
    with torch.no_grad():
        out = head(torch.from_numpy(X).to(device)).cpu().numpy()
    pred_b = out[:, 0]
    log_a = out[:, 1]
    b_idx = val_df["benchmark"].map(lambda b: benchmark_to_idx.get(b, 0)).to_numpy()
    if hier_mu_b is not None:
        pred_b = pred_b + hier_mu_b[b_idx]
    if hier_mu_a is not None:
        log_a = log_a + hier_mu_a[b_idx]
    pred_a = np.exp(np.clip(log_a, -3.0, 3.0))
    base_theta = float(np.mean(list(ability_by_name.values())) if ability_by_name else 0.0)
    theta = val_df["subject_name"].map(
        lambda n: ability_by_name.get(n, base_theta)
    ).to_numpy()
    z = pred_a * (theta - pred_b)
    if use_3pl:
        c_logit = out[:, 2]
        if hier_mu_c_logit is not None:
            c_logit = c_logit + hier_mu_c_logit[b_idx]
        c = 1.0 / (1.0 + np.exp(-c_logit))
        p_irt = c + (1.0 - c) * (1.0 / (1.0 + np.exp(-z)))
    else:
        p_irt = 1.0 / (1.0 + np.exp(-z))
    p_irt = np.clip(p_irt, 1e-6, 1 - 1e-6)

    logit_ncf = np.log(p_ncf / (1 - p_ncf))
    logit_irt = np.log(p_irt / (1 - p_irt))
    w = float(ncf_weight)
    p_ens = 1.0 / (1.0 + np.exp(-(w * logit_ncf + (1.0 - w) * logit_irt)))
    p_ens = np.clip(p_ens, 1e-6, 1 - 1e-6)

    base_ll = -(base_rate * math.log(base_rate)
                + (1 - base_rate) * math.log(1 - base_rate))

    metrics = {"baseline_log_loss": base_ll}
    for name, p in [("ncf", p_ncf), ("irt", p_irt), ("ensemble", p_ens)]:
        ll = float(log_loss(y, p))
        try:
            auc = float(roc_auc_score(y, p))
        except ValueError:
            auc = float("nan")
        print(f"[ColdStart] {name:>8s}  log_loss={ll:.4f}  auc={auc:.4f}")
        metrics[f"{name}_log_loss"] = ll
        metrics[f"{name}_auc"] = auc

    p_sb = None
    if sb_means:
        sb_keys = list(zip(val_df["subject_name"].astype(str).tolist(),
                           val_df["benchmark"].astype(str).tolist()))
        sb_p = np.array(
            [sb_means.get(k, base_rate) for k in sb_keys], dtype=np.float32
        )
        sb_p = np.clip(sb_p, 1e-3, 1 - 1e-3)
        p_sb = sb_p
        ll_sb = float(log_loss(y, p_sb))
        try:
            auc_sb = float(roc_auc_score(y, p_sb))
        except ValueError:
            auc_sb = float("nan")
        print(f"[ColdStart] {'sb_mean':>8s}  log_loss={ll_sb:.4f}  auc={auc_sb:.4f}")
        metrics["sb_log_loss"] = ll_sb
        metrics["sb_auc"] = auc_sb

        logit_sb = np.log(p_sb / (1 - p_sb))
        best3 = (None, None, float("inf"))
        for w_n in np.linspace(0, 1, 11):
            for w_s in np.linspace(0, 1, 11):
                if w_n + w_s > 1.0: continue
                w_i = 1.0 - w_n - w_s
                logit3 = w_n * logit_ncf + w_i * logit_irt + w_s * logit_sb
                p3 = np.clip(1 / (1 + np.exp(-logit3)), 1e-6, 1 - 1e-6)
                ll3 = float(log_loss(y, p3))
                if ll3 < best3[2]:
                    best3 = (float(w_n), float(w_s), ll3)
        print(f"[ColdStart] {'ens3 best':>16s}  w_ncf={best3[0]:.2f} w_sb={best3[1]:.2f} "
              f"w_irt={1-best3[0]-best3[1]:.2f}  log_loss={best3[2]:.4f}")
        metrics["ens3_best_w_ncf"] = best3[0]
        metrics["ens3_best_w_sb"] = best3[1]
        metrics["ens3_best_log_loss"] = best3[2]

    sweep = []
    for w_try in np.linspace(0.0, 1.0, 21):
        p_t = 1.0 / (1.0 + np.exp(-(w_try * logit_ncf + (1.0 - w_try) * logit_irt)))
        p_t = np.clip(p_t, 1e-6, 1 - 1e-6)
        ll_t = float(log_loss(y, p_t))
        sweep.append((float(w_try), ll_t))
    best_w, best_ll = min(sweep, key=lambda t: t[1])
    print(f"[ColdStart] ensemble sweep -> best w_ncf={best_w:.2f}  log_loss={best_ll:.4f}")
    metrics["ensemble_sweep"] = sweep
    metrics["ensemble_best_w"] = best_w
    metrics["ensemble_best_log_loss"] = best_ll

    try:
        from sklearn.linear_model import LogisticRegression
        best_logit_ens = best_w * logit_ncf + (1 - best_w) * logit_irt
        calib = {}
        val_benchmarks = val_df["benchmark"].to_numpy()
        for b in np.unique(val_benchmarks):
            mask = val_benchmarks == b
            if mask.sum() < 50:
                continue
            yb = y[mask]
            if yb.min() == yb.max():
                continue
            X = best_logit_ens[mask].reshape(-1, 1).astype(np.float64)
            clf = LogisticRegression(C=1.0, max_iter=200)
            clf.fit(X, yb)
            calib[str(b)] = {"a": float(clf.coef_[0, 0]), "b": float(clf.intercept_[0]),
                             "n": int(mask.sum())}

        p_calib = np.zeros_like(best_logit_ens)
        for i in range(len(p_calib)):
            b = str(val_benchmarks[i])
            if b in calib:
                lp = calib[b]["a"] * best_logit_ens[i] + calib[b]["b"]
            else:
                lp = best_logit_ens[i]
            p_calib[i] = 1.0 / (1.0 + np.exp(-lp))
        p_calib = np.clip(p_calib, 1e-6, 1 - 1e-6)
        ll_calib = float(log_loss(y, p_calib))
        try:
            auc_calib = float(roc_auc_score(y, p_calib))
        except ValueError:
            auc_calib = float("nan")
        print(f"[ColdStart] {'platt(per-bench)':>16s}  log_loss={ll_calib:.4f}  auc={auc_calib:.4f}")
        metrics["platt_log_loss"] = ll_calib
        metrics["platt_auc"] = auc_calib
        metrics["platt_calib"] = calib


        try:
            X_all = best_logit_ens.reshape(-1, 1).astype(np.float64)
            clf_g = LogisticRegression(C=1.0, max_iter=200)
            clf_g.fit(X_all, y)
            calib["__global__"] = {"a": float(clf_g.coef_[0, 0]),
                                   "b": float(clf_g.intercept_[0]),
                                   "n": int(len(y))}
            print(f"[ColdStart] {'platt_global':>16s} a={clf_g.coef_[0,0]:.3f} b={clf_g.intercept_[0]:.3f}")
        except Exception:
            pass

        try:
            from sklearn.isotonic import IsotonicRegression
            iso_calib = {}
            p_iso = np.zeros_like(best_logit_ens)
            for b in np.unique(val_benchmarks):
                mask = val_benchmarks == b
                if mask.sum() < 100:
                    continue
                yb = y[mask]
                if yb.min() == yb.max():
                    continue
                p_b = 1.0 / (1.0 + np.exp(-best_logit_ens[mask]))
                iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3)
                iso.fit(p_b, yb)

                grid = np.linspace(1e-3, 1 - 1e-3, 65)
                cal_grid = iso.predict(grid).astype(float)
                iso_calib[str(b)] = {"grid": grid.tolist(),
                                     "cal_grid": cal_grid.tolist(),
                                     "n": int(mask.sum())}
            for i in range(len(p_iso)):
                b = str(val_benchmarks[i])
                p_b = 1.0 / (1.0 + np.exp(-best_logit_ens[i]))
                if b in iso_calib:
                    grid = np.array(iso_calib[b]["grid"])
                    cg = np.array(iso_calib[b]["cal_grid"])
                    p_iso[i] = float(np.interp(p_b, grid, cg))
                else:
                    p_iso[i] = p_b
            p_iso = np.clip(p_iso, 1e-6, 1 - 1e-6)
            ll_iso = float(log_loss(y, p_iso))
            print(f"[ColdStart] {'isotonic(per-b)':>16s}  log_loss={ll_iso:.4f}")
            metrics["iso_log_loss"] = ll_iso
            metrics["iso_calib"] = iso_calib
        except Exception as e:
            print(f"[ColdStart] Isotonic calibration failed: {e}")
            metrics["iso_calib"] = {}
    except Exception as e:
        print(f"[ColdStart] Platt calibration failed: {e}")
        metrics["platt_log_loss"] = None
        metrics["platt_calib"] = {}

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION_DIR)
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER)
    parser.add_argument("--ncf-epochs", type=int, default=5)
    parser.add_argument("--ncf-lr", type=float, default=1e-3)
    parser.add_argument("--ncf-batch", type=int, default=1024)
    parser.add_argument("--irt-epochs", type=int, default=300)
    parser.add_argument("--irt-lr", type=float, default=0.05)
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--head-batch", type=int, default=256)
    parser.add_argument("--encode-batch", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 weight decay for NCF and item-head Adam "
                             "(Lecture 3 §sec-l2-reg: L2 == MAP w/ Gaussian prior).")
    parser.add_argument("--ncf-weight", type=float, default=0.3,
                        help="Weight on the NCF logit in the ensemble. "
                             "Prior run had IRT path beating NCF on cold-start, "
                             "so default favors IRT (0.7 IRT + 0.3 NCF).")
    parser.add_argument("--head-dropout", type=float, default=0.0,
                        help="Dropout in the item-head MLP (combats overfit).")
    parser.add_argument("--head-weight-decay", type=float, default=None,
                        help="Weight decay for item-head (defaults to --weight-decay).")
    parser.add_argument("--head-patience", type=int, default=0,
                        help="Early stop item-head after N epochs of no val_mse "
                             "improvement (0=disabled).")
    parser.add_argument("--likert-threshold", type=float, default=None,
                        help="If set, include non-binary rows binarized as "
                             "(label >= threshold). E.g., 4.0 turns ultrafeedback "
                             "5-pt Likert into binary with 'good=4-5, bad=1-3'.")
    parser.add_argument("--head-tune-weight", action="store_true",
                        help="After training, grid-search ncf_weight on val "
                             "and save the argmin. Otherwise use --ncf-weight as-is.")
    parser.add_argument("--joint-bce-head", action="store_true",
                        help="Train item-head with BCE on raw labels using "
                             "fixed theta from Stage B, instead of MSE on the "
                             "noisy (b, log_a) targets.")
    parser.add_argument("--joint-head-batch", type=int, default=8192,
                        help="Batch size for joint BCE head training (one batch "
                             "is many rows, not many items).")
    parser.add_argument("--ncf-dropout", type=float, default=0.0,
                        help="Dropout in the NCF MLP (combats overfit; NCF "
                             "currently bottoms out at epoch 1).")
    parser.add_argument("--ncf-hidden", type=int, default=256,
                        help="Hidden width for NCF MLP.")
    parser.add_argument("--use-ens3", action="store_true",
                        help="Use 3-way ensemble (NCF + IRT + sb_mean) even if "
                             "the val-best ens3 is not strictly better than ens2.")
    parser.add_argument("--bench-cond-dropout", type=float, default=0.0,
                        help="During joint-bce-head training, randomly zero "
                             "bench_oh + cond_oh per example to force item-text "
                             "generalization (helps if test has unseen benchmarks).")
    parser.add_argument("--use-3pl", action="store_true",
                        help="3PL extension: item-head emits (b, log_a, c_logit). "
                             "p = c + (1-c)*sigmoid(a*(theta-b)). Helps MCQ.")
    parser.add_argument("--hierarchical", action="store_true",
                        help="Hierarchical IRT: add learnable per-benchmark "
                             "(mu_b, mu_a) priors; item-head is residual from them.")
    parser.add_argument("--soft-likert", action="store_true",
                        help="Include non-binary rows as soft labels in the joint "
                             "BCE head training (aux data, weight 0.3). "
                             "ultrafeedback: label/5; livecodebench: label as-is; "
                             "etc. Doesn't feed IRT Stage B (still binary-only).")
    parser.add_argument("--aux-weight", type=float, default=0.3,
                        help="Weight on the auxiliary BCE loss term.")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction of UNIQUE items held out (cold-start).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Init / training seed.")
    parser.add_argument("--val-seed", type=int, default=None,
                        help="Seed for the cold-start val split (defaults to "
                             "--seed). Set this independently so multiple "
                             "seed runs share the same val items.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-package", action="store_true",
                        help="Skip post-training zip + smoke test + check_submission_zip.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = to_binary(load_long_table(args.data), likert_threshold=args.likert_threshold)
    base_rate = float(df["label"].mean())
    print(f"Rows: {len(df):,}  base_rate={base_rate:.4f}")

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(args.encoder, device=args.device)
    encoder.eval()

    subjects = df["subject_content"].fillna("").astype(str)
    items = df["item_content"].fillna("").astype(str)
    unique_subjects = subjects.unique().tolist()
    unique_items = items.unique().tolist()
    print(f"Unique subject strings: {len(unique_subjects):,}  "
          f"unique item strings: {len(unique_items):,}")

    print("Encoding unique subjects...")
    U_unique = encode_unique(encoder, unique_subjects, args.encode_batch)
    print("Encoding unique items...")
    V_unique = encode_unique(encoder, unique_items, args.encode_batch)
    emb_dim = U_unique.shape[1]

    s_lookup = {s: i for i, s in enumerate(unique_subjects)}
    i_lookup = {it: i for i, it in enumerate(unique_items)}
    U = U_unique[subjects.map(s_lookup).to_numpy()]
    V = V_unique[items.map(i_lookup).to_numpy()]
    y = df["label"].to_numpy(dtype=np.float32)

    U_t = torch.from_numpy(U).to(args.device)
    V_t = torch.from_numpy(V).to(args.device)
    y_t = torch.from_numpy(y).to(args.device)

    val_seed = args.val_seed if args.val_seed is not None else args.seed
    val_mask = cold_start_item_mask(items, args.val_frac, val_seed)

    ncf, ncf_metrics = train_ncf(
        U_t, V_t, y_t, val_mask,
        device=args.device, epochs=args.ncf_epochs,
        lr=args.ncf_lr, weight_decay=args.weight_decay,
        batch_size=args.ncf_batch, seed=args.seed,
        dropout=args.ncf_dropout, hidden=args.ncf_hidden,
    )

    df_train = df.loc[~val_mask].reset_index(drop=True)
    twopl, subj_to_idx, item_to_idx, irt_metrics = fit_twopl(
        df_train, args.device, args.irt_epochs, args.irt_lr, args.seed,
    )
    ability = twopl.ability.detach().cpu().numpy().astype(np.float32)
    difficulty = twopl.difficulty.detach().cpu().numpy().astype(np.float32)
    discrimination = twopl.discrimination.detach().cpu().numpy().astype(np.float32)
    ability_by_name: dict[str, float] = {
        name: float(ability[idx]) for name, idx in subj_to_idx.items()
    }

    targets = np.stack(
        [difficulty, np.log(np.clip(discrimination, 1e-3, None))], axis=1
    )
    head_wd = args.head_weight_decay if args.head_weight_decay is not None else args.weight_decay
    if args.joint_bce_head:
        aux_df = None
        if args.soft_likert:
            raw = load_long_table(args.data)
            non_bin = raw.loc[(~raw["label"].isin([0.0, 1.0])) &
                              raw["label"].notna() &
                              (raw["label"] >= 0) & (raw["label"] <= 10)].copy()
            if len(non_bin) > 0:
                def _scale(row):
                    b = row["benchmark"]
                    if b == "ultrafeedback":
                        return float(row["label"]) / 5.0
                    if b == "mtbench":
                        return float(row["label"]) / 10.0
                    return float(min(max(row["label"], 0.0), 1.0))
                non_bin["label"] = non_bin.apply(_scale, axis=1).astype(np.float32)
                aux_df = non_bin
                print(f"[soft-likert] {len(aux_df):,} aux rows added "
                      f"(benchmarks: {sorted(aux_df['benchmark'].unique().tolist())})")

        head, benchmark_to_idx, condition_to_idx, head_metrics = fit_item_head_bce(
            df_train, val_mask, df, V_unique, i_lookup, ability_by_name,
            device=args.device, epochs=args.head_epochs, lr=args.head_lr,
            weight_decay=head_wd,
            batch_size=args.joint_head_batch, seed=args.seed,
            dropout=args.head_dropout, patience=args.head_patience,
            bench_cond_dropout=args.bench_cond_dropout,
            use_3pl=args.use_3pl,
            hierarchical=args.hierarchical,
            aux_df=aux_df,
            aux_weight=args.aux_weight,
        )
    else:
        head, benchmark_to_idx, condition_to_idx, head_metrics = fit_item_head(
            df_train, item_to_idx, V_unique, i_lookup, targets,
            device=args.device, epochs=args.head_epochs, lr=args.head_lr,
            weight_decay=head_wd,
            batch_size=args.head_batch, seed=args.seed, val_frac=0.1,
            dropout=args.head_dropout, patience=args.head_patience,
        )

    sb_means: dict[tuple[str, str], float] = {}
    for (s, b), grp in df_train.groupby(["subject_name", "benchmark"]):
        if len(grp) >= 3:
            sb_means[(str(s), str(b))] = float(grp["label"].mean())
    print(f"[SB] computed {len(sb_means):,} (subject, benchmark) means "
          f"from training rows")

    hier_mu_b = np.array(head_metrics["mu_b"], dtype=np.float32)\
        if "mu_b" in head_metrics else None
    hier_mu_a = np.array(head_metrics["mu_a"], dtype=np.float32)\
        if "mu_a" in head_metrics else None
    hier_mu_c = np.array(head_metrics["mu_c_logit"], dtype=np.float32)\
        if "mu_c_logit" in head_metrics else None
    eval_metrics = cold_start_compare(
        df, val_mask, ncf, U_t, V_t, head, V_unique, i_lookup,
        benchmark_to_idx, condition_to_idx, ability_by_name,
        base_rate, args.device, ncf_weight=args.ncf_weight,
        sb_means=sb_means,
        use_3pl=bool(head_metrics.get("use_3pl", False)),
        hier_mu_b=hier_mu_b, hier_mu_a=hier_mu_a, hier_mu_c_logit=hier_mu_c,
    )

    args.submission_dir.mkdir(parents=True, exist_ok=True)

    torch.save(ncf.state_dict(), args.submission_dir / "ncf_head.pt")
    chosen_w = float(eval_metrics.get("ensemble_best_w", args.ncf_weight))\
        if args.head_tune_weight else float(args.ncf_weight)
    print(f"Saving ncf_meta with ncf_weight={chosen_w:.3f} "
          f"({'tuned' if args.head_tune_weight else 'user'})")
    with open(args.submission_dir / "ncf_meta.json", "w") as f:
        json.dump({
            "encoder": args.encoder,
            "embedding_dim": int(emb_dim),
            "hidden": int(args.ncf_hidden),
            "base_rate": base_rate,
            "ncf_weight": chosen_w,
            "ncf_dropout": float(args.ncf_dropout),
        }, f, indent=2)

    np.savez(
        args.submission_dir / "irt_params.npz",
        ability=ability,
        difficulty=difficulty,
        discrimination=discrimination,
        base_rate=np.float32(base_rate),
    )
    with open(args.submission_dir / "ability_by_name.pkl", "wb") as f:
        pickle.dump(ability_by_name, f)

    torch.save(head.state_dict(), args.submission_dir / "item_head.pt")
    with open(args.submission_dir / "vocab.pkl", "wb") as f:
        pickle.dump({
            "benchmark_to_idx": benchmark_to_idx,
            "condition_to_idx": condition_to_idx,
        }, f)

    extras_payload = {
        "use_3pl": bool(head_metrics.get("use_3pl", False)),
        "hierarchical": bool(head_metrics.get("hierarchical", False)),
    }
    if "mu_b" in head_metrics:
        extras_payload["mu_b"] = head_metrics["mu_b"]
        extras_payload["mu_a"] = head_metrics["mu_a"]
        if "mu_c_logit" in head_metrics:
            extras_payload["mu_c_logit"] = head_metrics["mu_c_logit"]
    with open(args.submission_dir / "head_extras.json", "w") as f:
        json.dump(extras_payload, f, indent=2)

    platt = eval_metrics.get("platt_calib", {}) or {}
    with open(args.submission_dir / "platt_calib.pkl", "wb") as f:
        pickle.dump(platt, f)
    if platt:
        print(f"Saved Platt calibrators for {len(platt)} benchmarks "
              f"(platt_log_loss={eval_metrics.get('platt_log_loss', 'NA')})")

    iso = eval_metrics.get("iso_calib", {}) or {}
    with open(args.submission_dir / "iso_calib.pkl", "wb") as f:
        pickle.dump(iso, f)
    iso_ll = eval_metrics.get("iso_log_loss")
    platt_ll = eval_metrics.get("platt_log_loss")
    if iso and iso_ll is not None and platt_ll is not None:
        prefer = "iso" if iso_ll < platt_ll else "platt"
        with open(args.submission_dir / "calib_choice.json", "w") as f:
            json.dump({"prefer": prefer, "iso_log_loss": iso_ll,
                       "platt_log_loss": platt_ll}, f, indent=2)
        print(f"Calib choice: {prefer} (iso={iso_ll:.4f}, platt={platt_ll:.4f})")

    with open(args.submission_dir / "sb_means.pkl", "wb") as f:
        pickle.dump(sb_means, f)

    ens3_meta = {
        "use_ens3": False,
        "w_ncf": chosen_w,
        "w_sb": 0.0,
        "w_irt": 1.0 - chosen_w,
    }
    ens3_w_ncf = eval_metrics.get("ens3_best_w_ncf")
    ens3_w_sb = eval_metrics.get("ens3_best_w_sb")
    ens3_ll = eval_metrics.get("ens3_best_log_loss")
    ens2_ll = eval_metrics.get("ensemble_best_log_loss")
    if (args.use_ens3 or
        (ens3_ll is not None and ens2_ll is not None and ens3_ll < ens2_ll - 0.0005)):
        ens3_meta["use_ens3"] = True
        ens3_meta["w_ncf"] = float(ens3_w_ncf)
        ens3_meta["w_sb"] = float(ens3_w_sb)
        ens3_meta["w_irt"] = 1.0 - float(ens3_w_ncf) - float(ens3_w_sb)
        print(f"Saving ens3_meta: w_ncf={ens3_meta['w_ncf']:.2f} "
              f"w_sb={ens3_meta['w_sb']:.2f} w_irt={ens3_meta['w_irt']:.2f}")
    with open(args.submission_dir / "ens3_meta.json", "w") as f:
        json.dump(ens3_meta, f, indent=2)

    full_meta = {
        "encoder": args.encoder,
        "embedding_dim": int(emb_dim),
        "ncf_hidden": 256,
        "head_hidden": 256,
        "n_subjects": len(subj_to_idx),
        "n_items": len(item_to_idx),
        "n_benchmarks": len(benchmark_to_idx),
        "n_conditions": len(condition_to_idx),
        "base_rate": base_rate,
        **ncf_metrics,
        **irt_metrics,
        **head_metrics,
        **eval_metrics,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }
    with open(args.submission_dir / "full_meta.json", "w") as f:
        json.dump(full_meta, f, indent=2)

    print(f"\nWrote artifacts to {args.submission_dir}")

    if not args.no_package:
        _package_submission(args.submission_dir)


def _package_submission(submission_dir: Path) -> None:
    import subprocess
    import sys
    script = ROOT / "scripts" / "package.py"
    if not script.exists():
        print(f"[train_full] WARNING: {script} missing; skipping packaging.")
        return
    out_zip = submission_dir.parent / f"{submission_dir.name}.zip"
    proc = subprocess.run(
        [sys.executable, str(script),
         "--submission-dir", str(submission_dir),
         "--out", str(out_zip)],
        cwd=ROOT,
    )
    if proc.returncode != 0:
        raise SystemExit(f"[train_full] packaging failed (exit {proc.returncode})")


if __name__ == "__main__":
    main()
