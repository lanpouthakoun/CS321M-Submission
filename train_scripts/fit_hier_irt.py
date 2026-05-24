from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def hier_mirt_with_ncf(subject_idx, item_idx, cond_idx,
                       N_subjects, N_items, N_conds, K,
                       U_text_prior, V_text_prior,
                       b_subj_prior, b_item_prior,
                       ncf_logit, theta_prior_mean,
                       use_3pl=False, c_prior_alpha=1.0, c_prior_beta=4.0,
                       use_3pl_item=False,
                       y=None):
    sigma_U = numpyro.sample("sigma_U", dist.HalfNormal(0.5))
    sigma_V = numpyro.sample("sigma_V", dist.HalfNormal(0.5))
    sigma_bsubj = numpyro.sample("sigma_bsubj", dist.HalfNormal(0.5))
    sigma_bitem = numpyro.sample("sigma_bitem", dist.HalfNormal(0.5))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(0.5))
    beta = numpyro.sample("beta", dist.Normal(0.0, sigma_beta))

    if use_3pl:


        if use_3pl_item:
            alpha_c = numpyro.sample("alpha_c", dist.LogNormal(0.0, 0.5))
            beta_c = numpyro.sample("beta_c", dist.LogNormal(1.0, 0.5))

            with numpyro.plate("item_c", N_items):
                c_item = numpyro.sample("c_item", dist.Beta(alpha_c, beta_c))

            c_bench_mean = numpyro.deterministic("c_bench_mean",
                                                  alpha_c / (alpha_c + beta_c))
        else:
            c_bench = numpyro.sample("c_bench",
                                     dist.Beta(c_prior_alpha, c_prior_beta))


    with numpyro.plate("subj_k", K, dim=-1):
        with numpyro.plate("subj", N_subjects, dim=-2):
            U_raw = numpyro.sample("U_raw", dist.Normal(0.0, 1.0))
    U_unanchored = U_text_prior + sigma_U * U_raw
    anchor = jnp.where(U_unanchored[0, 0] >= 0.0, 1.0, -1.0)
    U = numpyro.deterministic("U", anchor * U_unanchored)


    with numpyro.plate("item_k", K, dim=-1):
        with numpyro.plate("item", N_items, dim=-2):
            V_raw = numpyro.sample("V_raw", dist.Normal(0.0, 1.0))
    V = numpyro.deterministic("V", anchor * (V_text_prior + sigma_V * V_raw))


    with numpyro.plate("subj_b", N_subjects):
        bs_raw = numpyro.sample("bs_raw", dist.Normal(0.0, 1.0))
        b_subj = numpyro.deterministic("b_subj", b_subj_prior + sigma_bsubj * bs_raw)
    with numpyro.plate("item_b", N_items):
        bi_raw = numpyro.sample("bi_raw", dist.Normal(0.0, 1.0))
        b_item = numpyro.deterministic("b_item", b_item_prior + sigma_bitem * bi_raw)

    with numpyro.plate("cond", N_conds):
        tau = numpyro.sample("tau", dist.Normal(0.0, 1.0))


    U_row = U[subject_idx]
    V_row = V[item_idx]
    dot = (U_row * V_row).sum(axis=-1)
    logit = dot + b_subj[subject_idx] + b_item[item_idx] + tau[cond_idx] + beta * ncf_logit
    if use_3pl:
        p_irt = jax.nn.sigmoid(logit)
        if use_3pl_item:
            c_row = c_item[item_idx]
        else:
            c_row = c_bench
        p = c_row + (1.0 - c_row) * p_irt
        p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
        numpyro.sample("y", dist.Bernoulli(probs=p), obs=y)
    else:
        numpyro.sample("y", dist.Bernoulli(logits=logit), obs=y)


def hier_2pl_with_ncf(subject_idx, item_idx, cond_idx,
                      N_subjects, N_items, N_conds,
                      b_text_prior, loga_text_prior,
                      ncf_logit,
                      theta_prior_mean,
                      use_3pl=False,
                      c_prior_alpha=1.0, c_prior_beta=4.0,
                      y=None):
    sigma_theta = numpyro.sample("sigma_theta", dist.HalfNormal(1.0))
    sigma_b = numpyro.sample("sigma_b", dist.HalfNormal(1.0))
    sigma_a = numpyro.sample("sigma_a", dist.HalfNormal(0.5))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(0.5))
    beta = numpyro.sample("beta", dist.Normal(0.0, sigma_beta))

    if use_3pl:


        c_bench = numpyro.sample("c_bench",
                                 dist.Beta(c_prior_alpha, c_prior_beta))

    with numpyro.plate("subj", N_subjects):
        theta_raw = numpyro.sample("theta_raw", dist.Normal(0.0, 1.0))
        theta = numpyro.deterministic(
            "theta", theta_prior_mean + sigma_theta * theta_raw)

    with numpyro.plate("cond", N_conds):
        tau = numpyro.sample("tau", dist.Normal(0.0, 1.0))

    with numpyro.plate("item", N_items):
        b_raw = numpyro.sample("b_raw", dist.Normal(0.0, 1.0))
        loga_raw = numpyro.sample("loga_raw", dist.Normal(0.0, 1.0))
        b = b_text_prior + sigma_b * b_raw
        log_a = loga_text_prior + sigma_a * loga_raw

    a = jnp.exp(jnp.clip(log_a, -3.0, 3.0))
    logit = (a[item_idx] * (theta[subject_idx] - b[item_idx])
             + tau[cond_idx]
             + beta * ncf_logit)
    if use_3pl:

        p_2pl = jax.nn.sigmoid(logit)
        p = c_bench + (1.0 - c_bench) * p_2pl
        p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
        numpyro.sample("y", dist.Bernoulli(probs=p), obs=y)
    else:
        numpyro.sample("y", dist.Bernoulli(logits=logit), obs=y)


def _make_heads():
    import torch.nn as nn
    class _ItemParamHead(nn.Module):
        def __init__(self, emb_dim, n_benchmarks, n_conditions,
                     hidden=256, dropout=0.0):
            super().__init__()
            in_dim = emb_dim + n_benchmarks + n_conditions
            layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
            if dropout > 0: layers.append(nn.Dropout(dropout))
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
            if dropout > 0: layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden, 2))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    class _NCFHead(nn.Module):
        def __init__(self, d, hidden=256, dropout=0.0):
            import torch
            super().__init__()
            layers = [nn.Linear(2 * d, hidden), nn.ReLU()]
            if dropout > 0: layers.append(nn.Dropout(dropout))
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
            if dropout > 0: layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden, 1))
            self.net = nn.Sequential(*layers)
        def forward(self, u, v):
            import torch
            return self.net(torch.cat([u, v], dim=-1)).squeeze(-1)
    return _ItemParamHead, _NCFHead


def compute_text_priors(items, bench, vocab, item_head_state, emb_dim,
                        encoder, head_dropout, device="cpu", emb_cache=None):
    import torch
    _ItemParamHead, _ = _make_heads()

    bench_to_idx = vocab["benchmark_to_idx"]
    cond_to_idx = vocab["condition_to_idx"]
    head = _ItemParamHead(emb_dim, len(bench_to_idx), len(cond_to_idx),
                          dropout=head_dropout).to(device)
    head.load_state_dict(item_head_state)
    head.eval()

    bench_idx = bench_to_idx[bench]
    b_oh = torch.zeros(1, len(bench_to_idx), device=device)
    b_oh[0, bench_idx] = 1.0

    if emb_cache is not None and "item_to_emb" in emb_cache:
        cached = [emb_cache["item_to_emb"].get(it) for it in items]
        missing_idx = [i for i, c in enumerate(cached) if c is None]
        if missing_idx:
            missing_items = [items[i] for i in missing_idx]
            print(f"  encoding {len(missing_items)} missing items "
                  f"(of {len(items)}, {len(items) - len(missing_items)} cached)...",
                  flush=True)
            missing_v = encoder.encode(missing_items, convert_to_numpy=True,
                                       normalize_embeddings=True,
                                       show_progress_bar=False,
                                       batch_size=128).astype(np.float32)
            for j, it in zip(missing_idx, range(len(missing_v))):
                cached[j] = missing_v[it]
        else:
            print(f"  reusing cached item embeddings for {len(items)} items",
                  flush=True)
        v = np.stack(cached, axis=0).astype(np.float32)
    else:
        print(f"  encoding {len(items)} items...", flush=True)
        v = encoder.encode(items, convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False, batch_size=128).astype(np.float32)
    v_t = torch.from_numpy(v).to(device)

    bs, logas = [], []
    n_cond = len(cond_to_idx)
    with torch.no_grad():
        for cond_idx in range(n_cond):
            c_oh = torch.zeros(v_t.shape[0], n_cond, device=device)
            c_oh[:, cond_idx] = 1.0
            x = torch.cat([v_t, b_oh.expand(v_t.shape[0], -1), c_oh], dim=-1)
            out = head(x).cpu().numpy()
            bs.append(out[:, 0]); logas.append(out[:, 1])
    return np.mean(bs, axis=0).astype(np.float32), np.mean(logas, axis=0).astype(np.float32), v


def compute_ncf_logits(df, subj_emb_cache, item_v, item_to_idx, ncf_state,
                       ncf_meta, device="cpu"):
    import torch
    _, _NCFHead = _make_heads()

    ncf_dropout = ncf_meta.get("ncf_dropout", 0.0)
    head = _NCFHead(d=int(ncf_meta["embedding_dim"]),
                    hidden=int(ncf_meta["hidden"]),
                    dropout=ncf_dropout).to(device)
    head.load_state_dict(ncf_state)
    head.eval()


    print(f"  ncf logits for {len(df):,} rows...", flush=True)
    u_arr = np.stack([subj_emb_cache[s] for s in df.subject_content], axis=0)
    v_arr = np.stack([item_v[item_to_idx[it]] for it in df.item_content], axis=0)

    with torch.no_grad():
        u_t = torch.from_numpy(u_arr).to(device)
        v_t = torch.from_numpy(v_arr).to(device)
        logits = head(u_t, v_t).cpu().numpy().ravel()
    return logits.astype(np.float32)


def fit_benchmark(bench, df_all, vocab, ability_by_name,
                  item_head_state, head_dropout,
                  ncf_state, ncf_meta,
                  encoder, args, out_dir,
                  emb_cache=None):
    print(f"\n========== {bench} ==========", flush=True)
    df = df_all.loc[df_all.benchmark == bench].reset_index(drop=True)
    print(f"  rows={len(df):,}  items={df.item_content.nunique()}  "
          f"subjects={df.subject_content.nunique()}  conds={df.condition.nunique()}",
          flush=True)


    filter_map = getattr(args, "_filter_map", None)
    if filter_map is not None and bench in filter_map:
        keep_items = set(filter_map[bench])
        n_before = len(df)
        df = df.loc[df.item_content.isin(keep_items)].reset_index(drop=True)
        print(f"  iterative-filter: kept {df.item_content.nunique()} items, "
              f"rows {n_before:,} -> {len(df):,}", flush=True)

    if len(df) > args.max_rows:
        print(f"  subsampling to {args.max_rows:,} rows for MCMC speed", flush=True)
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)


    subjects = sorted(df.subject_content.unique().tolist())
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    items = sorted(df.item_content.unique().tolist())
    item_to_idx = {it: i for i, it in enumerate(items)}
    conds = sorted(df.condition.unique().tolist())
    cond_to_idx = {c: i for i, c in enumerate(conds)}


    subj_names = [_parse_subject_name(s) for s in subjects]
    if ability_by_name:
        pop_mean = float(np.mean(list(ability_by_name.values())))
    else:
        pop_mean = 0.0
    theta_prior_mean_arr = np.array(
        [ability_by_name.get(n, pop_mean) for n in subj_names],
        dtype=np.float32)
    n_with_mle = sum(1 for n in subj_names if n in ability_by_name)
    print(f"  theta prior: {n_with_mle}/{len(subjects)} subjects have MLE θ  "
          f"(fallback={pop_mean:.3f})", flush=True)


    if emb_cache is not None and "subj_to_emb" in emb_cache:
        subj_emb_cache = {s: emb_cache["subj_to_emb"][s]
                          for s in subjects if s in emb_cache["subj_to_emb"]}
        if len(subj_emb_cache) < len(subjects):
            missing = [s for s in subjects if s not in emb_cache["subj_to_emb"]]
            print(f"  encoding {len(missing)} missing subjects...", flush=True)
            missing_arr = encoder.encode(missing, convert_to_numpy=True,
                                         normalize_embeddings=True,
                                         show_progress_bar=False,
                                         batch_size=128).astype(np.float32)
            for s, v in zip(missing, missing_arr):
                subj_emb_cache[s] = v
        else:
            print(f"  reusing cached subj embeddings for {len(subjects)} subjects",
                  flush=True)
    else:
        print(f"  encoding {len(subjects)} subjects...", flush=True)
        subj_arr = encoder.encode(subjects, convert_to_numpy=True,
                                  normalize_embeddings=True,
                                  show_progress_bar=False, batch_size=128).astype(np.float32)
        subj_emb_cache = {s: subj_arr[i] for i, s in enumerate(subjects)}


    b_text, loga_text, item_v = compute_text_priors(
        items, bench, vocab, item_head_state,
        emb_dim=int(ncf_meta["embedding_dim"]),
        encoder=encoder, head_dropout=head_dropout,
        emb_cache=emb_cache)
    print(f"  text prior: b mean={b_text.mean():.3f} std={b_text.std():.3f}",
          flush=True)


    if args.use_ncf:
        ncf_logits = compute_ncf_logits(df, subj_emb_cache, item_v, item_to_idx,
                                        ncf_state, ncf_meta)
    else:
        ncf_logits = np.zeros(len(df), dtype=np.float32)

    tr_subj = df.subject_content.map(subj_to_idx).to_numpy()
    tr_item = df.item_content.map(item_to_idx).to_numpy()
    tr_cond = df.condition.map(cond_to_idx).to_numpy()
    tr_y = df.label.to_numpy().astype(np.int32)


    mirt_state = getattr(args, "_mirt_state", None)
    if args.use_mirt and mirt_state is not None:
        K = int(mirt_state["K"])
        Wsubj = mirt_state["state_dict"]["subj_proj.weight"].cpu().numpy()
        Witem = mirt_state["state_dict"]["item_proj.weight"].cpu().numpy()

        subj_arr = np.stack([subj_emb_cache[s] for s in subjects], axis=0)
        U_text_prior = (subj_arr @ Wsubj.T).astype(np.float32)

        V_text_prior = (item_v @ Witem.T).astype(np.float32)

        b_subj_prior = theta_prior_mean_arr.copy()
        b_item_prior = -b_text
        print(f"  MIRT priors: K={K}  U_text {U_text_prior.shape}  V_text {V_text_prior.shape}",
              flush=True)

    print(f"  running NUTS ({args.num_warmup} warmup + {args.num_samples} samples)...",
          flush=True)
    t0 = time.time()
    if args.use_mirt and mirt_state is not None:
        kernel = NUTS(hier_mirt_with_ncf, target_accept_prob=0.85, max_tree_depth=8)
    else:
        kernel = NUTS(hier_2pl_with_ncf, target_accept_prob=0.85, max_tree_depth=8)
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples,
                num_chains=int(args.num_chains),
                chain_method="vectorized" if args.num_chains > 1 else "sequential",
                progress_bar=False)
    if args.use_mirt and mirt_state is not None:
        mcmc.run(jax.random.PRNGKey(args.seed),
                 subject_idx=jnp.array(tr_subj),
                 item_idx=jnp.array(tr_item),
                 cond_idx=jnp.array(tr_cond),
                 N_subjects=len(subjects), N_items=len(items), N_conds=len(conds),
                 K=int(mirt_state["K"]),
                 U_text_prior=jnp.array(U_text_prior),
                 V_text_prior=jnp.array(V_text_prior),
                 b_subj_prior=jnp.array(b_subj_prior),
                 b_item_prior=jnp.array(b_item_prior),
                 ncf_logit=jnp.array(ncf_logits),
                 theta_prior_mean=jnp.array(theta_prior_mean_arr),
                 use_3pl=bool(args.use_3pl),
                 y=jnp.array(tr_y))
    else:
      mcmc.run(jax.random.PRNGKey(args.seed),
             subject_idx=jnp.array(tr_subj),
             item_idx=jnp.array(tr_item),
             cond_idx=jnp.array(tr_cond),
             N_subjects=len(subjects), N_items=len(items), N_conds=len(conds),
             b_text_prior=jnp.array(b_text),
             loga_text_prior=jnp.array(loga_text),
             ncf_logit=jnp.array(ncf_logits),
             theta_prior_mean=jnp.array(theta_prior_mean_arr),
             use_3pl=bool(args.use_3pl),
             use_3pl_item=bool(getattr(args, "use_3pl_item", False)),
             y=jnp.array(tr_y))
    elapsed = time.time() - t0
    samples = mcmc.get_samples()
    extra = ""
    if "c_bench" in samples:
        extra += f"  c_bench={float(samples['c_bench'].mean()):.3f}"
    if "c_bench_mean" in samples:
        extra += f"  c_bench_mean={float(samples['c_bench_mean'].mean()):.3f}"
    if "sigma_U" in samples:
        extra += f"  sigma_U={float(samples['sigma_U'].mean()):.3f}  sigma_V={float(samples['sigma_V'].mean()):.3f}"
        print(f"  NUTS done in {elapsed:.1f}s"
              f"  beta={float(samples['beta'].mean()):.3f}"
              f"  sigma_bsubj={float(samples['sigma_bsubj'].mean()):.3f}"
              f"  sigma_bitem={float(samples['sigma_bitem'].mean()):.3f}"
              f"{extra}",
              flush=True)
    else:
        print(f"  NUTS done in {elapsed:.1f}s  "
              f"sigma_b={float(samples['sigma_b'].mean()):.3f}  "
              f"sigma_a={float(samples['sigma_a'].mean()):.3f}  "
              f"beta={float(samples['beta'].mean()):.3f}  "
              f"sigma_beta={float(samples['sigma_beta'].mean()):.3f}"
              f"{extra}",
              flush=True)


    parse_name = _parse_subject_name
    subj_names = [parse_name(s) for s in subjects]

    is_mirt = "sigma_U" in samples
    out = {
        "benchmark": bench,
        "subj_names": subj_names,
        "subj_contents": subjects,
        "conds": conds,
        "tau_samples": np.asarray(samples["tau"]),
        "beta_samples": np.asarray(samples["beta"]),
        "sigma_beta_samples": np.asarray(samples["sigma_beta"]),
        "use_ncf": bool(args.use_ncf),
        "use_3pl": bool(args.use_3pl),
        "use_mirt": bool(is_mirt),
        "n_train_rows": int(len(df)),
        "n_items": int(len(items)),
        "fit_time_seconds": float(elapsed),
    }
    if is_mirt:

        out["K"] = int(samples["U_raw"].shape[-1])


        out["U_post_mean"] = np.asarray(samples["U"].mean(axis=0))
        out["U_post_std"] = np.asarray(samples["U"].std(axis=0))
        out["b_subj_post_mean"] = np.asarray(samples["b_subj"].mean(axis=0))
        out["b_subj_post_std"] = np.asarray(samples["b_subj"].std(axis=0))
        out["sigma_U_samples"] = np.asarray(samples["sigma_U"])
        out["sigma_V_samples"] = np.asarray(samples["sigma_V"])
        out["sigma_bitem_samples"] = np.asarray(samples["sigma_bitem"])
        out["sigma_bsubj_samples"] = np.asarray(samples["sigma_bsubj"])
    else:
        out["theta_samples"] = np.asarray(samples["theta"])
        out["sigma_b_samples"] = np.asarray(samples["sigma_b"])
        out["sigma_a_samples"] = np.asarray(samples["sigma_a"])
        out["sigma_theta_samples"] = np.asarray(samples["sigma_theta"])
    if "c_bench" in samples:
        out["c_bench_samples"] = np.asarray(samples["c_bench"])
    if "c_bench_mean" in samples:


        out["c_bench_samples"] = np.asarray(samples["c_bench_mean"])
        out["alpha_c_samples"] = np.asarray(samples["alpha_c"])
        out["beta_c_samples"] = np.asarray(samples["beta_c"])
        out["use_3pl_item"] = True

    out_path = out_dir / f"posterior_{bench}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    size_kb = out_path.stat().st_size / 1024
    print(f"  saved {out_path.name} ({size_kb:.0f} KB)", flush=True)
    return out


def _parse_subject_name(subject_content: str) -> str:
    if not subject_content:
        return ""
    for line in subject_content.splitlines():
        if line.startswith("Name:"):
            return line[len("Name:"):].strip()
    for line in subject_content.splitlines():
        if line.strip():
            return line.strip()
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data", default=str(ROOT / "data" / "training_long.parquet"))
    parser.add_argument("--submission-dir", default=str(ROOT / "submission"))
    parser.add_argument("--benchmarks", nargs="*", default=None,
                        help="Subset of benchmarks (default: all).")
    parser.add_argument("--max-rows", type=int, default=30000,
                        help="Subsample benchmarks larger than this for fit speed.")
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-ncf", action="store_true",
                        help="Include NCF residual term in the model.")
    parser.add_argument("--use-3pl", action="store_true",
                        help="Add benchmark-level guessing parameter c_b ~ Beta(1,4).")
    parser.add_argument("--use-3pl-item", action="store_true",
                        help="True 3PL: per-item c_i with benchmark-level Beta hyperprior. "
                             "Implies --use-3pl.")
    parser.add_argument("--use-mirt", action="store_true",
                        help="Use K-dim logistic factor model instead of scalar 2PL.")
    parser.add_argument("--mirt-heads", type=Path, default=None,
                        help="mirt_heads.pt with W_subj, W_item projections.")
    parser.add_argument("--emb-cache", type=Path, default=None,
                        help="Pickle with pre-encoded {subjects, items, subj_emb, item_emb}.")
    parser.add_argument("--filter-items", type=Path, default=None,
                        help="Pickle from training/iterative_filter.py: per-benchmark surviving item_content lists.")
    parser.add_argument("--encoder",
                        default="sentence-transformers/all-mpnet-base-v2")
    args = parser.parse_args()
    if args.num_chains > 1:
        numpyro.set_host_device_count(int(args.num_chains))

    args.out.mkdir(parents=True, exist_ok=True)

    print("loading data...", flush=True)
    df = pd.read_parquet(args.data)
    df = df.loc[df.label.isin([0.0, 1.0])].copy()
    df["label"] = df["label"].astype(np.float32)
    print(f"  rows={len(df):,}  benchmarks={df.benchmark.nunique()}", flush=True)

    if args.benchmarks:
        df = df.loc[df.benchmark.isin(args.benchmarks)].reset_index(drop=True)

    sub_dir = Path(args.submission_dir)
    with open(sub_dir / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(sub_dir / "ability_by_name.pkl", "rb") as f:
        ability_by_name = pickle.load(f)
    with open(sub_dir / "ncf_meta.json") as f:
        ncf_meta = json.load(f)

    import torch
    item_head_state = torch.load(sub_dir / "item_head.pt", map_location="cpu")
    head_dropout = 0.3 if "net.3.weight" in item_head_state else 0.0
    ncf_state = torch.load(sub_dir / "ncf_head.pt", map_location="cpu")

    print(f"loading encoder {args.encoder}...", flush=True)
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(args.encoder)

    args._mirt_state = None
    if args.use_mirt:
        if args.mirt_heads is None:
            args.mirt_heads = Path(args.submission_dir) / "mirt_heads.pt"
        if not args.mirt_heads.exists():
            raise SystemExit(f"--use-mirt requires mirt_heads.pt; missing at {args.mirt_heads}")
        print(f"loading MIRT heads {args.mirt_heads}...", flush=True)
        args._mirt_state = torch.load(args.mirt_heads, map_location="cpu", weights_only=False)
        print(f"  K={args._mirt_state['K']}  val_log_loss={args._mirt_state.get('val_log_loss', 'na')}",
              flush=True)

    args._filter_map = None
    if args.filter_items and args.filter_items.exists():
        with open(args.filter_items, "rb") as f:
            filt = pickle.load(f)
        args._filter_map = filt["filtered"]
        print(f"loaded iterative filter: {len(args._filter_map)} benchmarks", flush=True)

    emb_cache = None
    if args.emb_cache and args.emb_cache.exists():
        print(f"loading embedding cache {args.emb_cache}...", flush=True)
        with open(args.emb_cache, "rb") as f:
            raw = pickle.load(f)
        emb_cache = {
            "subj_to_emb": {s: e for s, e in zip(raw["subjects"], raw["subj_emb"])},
            "item_to_emb": {s: e for s, e in zip(raw["items"], raw["item_emb"])},
        }
        print(f"  cached: {len(emb_cache['subj_to_emb']):,} subjects, "
              f"{len(emb_cache['item_to_emb']):,} items", flush=True)

    benchmarks = sorted(df.benchmark.unique().tolist())
    print(f"will fit {len(benchmarks)} benchmarks: {benchmarks}", flush=True)

    results = {}
    t_start = time.time()
    for bench in benchmarks:
        try:
            results[bench] = fit_benchmark(
                bench, df, vocab, ability_by_name,
                item_head_state, head_dropout,
                ncf_state, ncf_meta,
                encoder, args, args.out,
                emb_cache=emb_cache)
        except Exception as e:
            print(f"  FAILED on {bench}: {e}", flush=True)
            import traceback
            traceback.print_exc()
    elapsed = time.time() - t_start
    print(f"\n========== ALL DONE in {elapsed:.1f}s ==========", flush=True)
    print(f"Saved posteriors for {len(results)}/{len(benchmarks)} benchmarks "
          f"to {args.out}", flush=True)


if __name__ == "__main__":
    main()
