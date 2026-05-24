"""MCMC submission: hierarchical Bayesian 2PL IRT with text-informed priors.

For each (subject, item, benchmark, condition) we marginalize over the
benchmark-specific posterior of (theta_s, sigma_b, sigma_a, tau_c, beta)
drawn by NUTS at training time. Item parameters (b, log_a) are *not*
fit per item -- at predict time they are sampled from the prior
    b_i ~ N(b_hat(text_i), sigma_b)
    log a_i ~ N(log_a_hat(text_i), sigma_a)
and the per-sample probabilities are averaged. This is the
posterior-predictive distribution for a new (cold-start) item.

The NCF head is retained as a residual term:
    logit = a * (theta_s - b) + tau_c + beta * f_NCF(u_s, v_i)

with beta ~ N(0, sigma_beta) so the MCMC shrinks NCF when it adds noise.

No lookup tables. No per-benchmark alpha blend. No Platt calibration.
The posterior-predictive average naturally produces conservative,
well-calibrated probabilities on held-out items.

Files loaded at module init (all at the ZIP root):
    Required:                       Optional (for adaptive labeling):
        ncf_head.pt                     irt_params.npz   (not used here)
        ncf_meta.json
        item_head.pt
        vocab.pkl
        posteriors/posterior_<bench>.pkl   (one per benchmark)
        (encoder/ dir if shipping a fine-tuned encoder)
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent

with open(HERE / "ncf_meta.json") as _f:
    _NCF_META = json.load(_f)

_ENCODER_REPO: str = _NCF_META["encoder"]
_EMB_DIM: int = int(_NCF_META["embedding_dim"])
_NCF_HIDDEN: int = int(_NCF_META.get("hidden", 256))
_BASE_RATE: float = float(_NCF_META.get("base_rate", 0.65))

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Local encoder dir takes precedence (for shipping a fine-tuned encoder).
_LOCAL_ENCODER = HERE / "encoder"
if _LOCAL_ENCODER.exists() and (_LOCAL_ENCODER / "modules.json").exists():
    _ENCODER_PATH = str(_LOCAL_ENCODER)
else:
    _ENCODER_PATH = _ENCODER_REPO
_ENCODER = SentenceTransformer(_ENCODER_PATH, device=_DEVICE)
_ENCODER.eval()


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class _NCFHead(nn.Module):
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


class _ItemParamHead(nn.Module):
    def __init__(self, emb_dim: int, n_benchmarks: int, n_conditions: int,
                 hidden: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        in_dim = emb_dim + n_benchmarks + n_conditions
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Load heads
# ---------------------------------------------------------------------------

_ncf_state = torch.load(HERE / "ncf_head.pt", map_location=_DEVICE)
_ncf_dropout = 0.3 if "net.3.weight" in _ncf_state else 0.0
_NCF = _NCFHead(d=_EMB_DIM, hidden=_NCF_HIDDEN, dropout=_ncf_dropout).to(_DEVICE)
_NCF.load_state_dict(_ncf_state)
_NCF.eval()

with open(HERE / "vocab.pkl", "rb") as _f:
    _vocab = pickle.load(_f)
_BENCHMARK_TO_IDX = _vocab["benchmark_to_idx"]
_CONDITION_TO_IDX = _vocab["condition_to_idx"]

_item_state = torch.load(HERE / "item_head.pt", map_location=_DEVICE)
_head_dropout = 0.3 if "net.3.weight" in _item_state else 0.0
_ITEM_HEAD = _ItemParamHead(
    _EMB_DIM, len(_BENCHMARK_TO_IDX), len(_CONDITION_TO_IDX),
    dropout=_head_dropout,
).to(_DEVICE)
_ITEM_HEAD.load_state_dict(_item_state)
_ITEM_HEAD.eval()


# ---------------------------------------------------------------------------
# Load per-benchmark posteriors (from the MCMC fits)
# ---------------------------------------------------------------------------

_POSTERIORS_DIR = HERE / "posteriors"
_POSTERIORS: dict[str, dict] = {}
if _POSTERIORS_DIR.exists():
    for p in _POSTERIORS_DIR.glob("posterior_*.pkl"):
        bench = p.stem[len("posterior_"):]
        with open(p, "rb") as f:
            _POSTERIORS[bench] = pickle.load(f)
print(f"[model_mcmc] loaded posteriors for {len(_POSTERIORS)} benchmarks: "
      f"{sorted(_POSTERIORS.keys())}")

# Precompute per-benchmark subject_name -> theta_samples row index for fast lookup.
_SUBJ_LOOKUP: dict[str, dict[str, int]] = {}
for bench, post in _POSTERIORS.items():
    _SUBJ_LOOKUP[bench] = {name: i for i, name in enumerate(post["subj_names"])}
# RESIDUAL_V2_PATCH_MARKER
_RES_V2_ROUND_CACHE: dict = {}
_RES_V2_N0 = 10.0


def _residual_v2_get_delta(benchmark: str, condition: str, labeled: list) -> float:
    if labeled is None or len(labeled) == 0:
        return 0.0
    key = id(labeled)
    cache = _RES_V2_ROUND_CACHE.get(key)
    if cache is None:
        # Compute (bench, cond) and (bench) shifts in one pass
        per_bc: dict = {}
        per_b: dict = {}
        for ex in labeled:
            try:
                b = ex.get("benchmark", "")
                y = ex.get("label")
                if y is None or b == "":
                    continue
                sub_content = ex.get("subject_content", "") or ""
                it_content = ex.get("item_content", "") or ""
                cond = ex.get("condition", "") or ""
            except Exception:
                continue
            try:
                p = _predict_inner({
                    "subject_content": sub_content,
                    "item_content": it_content,
                    "benchmark": b,
                    "condition": cond,
                })
                r = float(y) - p
            except Exception:
                continue
            per_bc.setdefault((b, cond), []).append(r)
            per_b.setdefault(b, []).append(r)

        cache = {"bc": {}, "b": {}}
        for (b, c), rs in per_bc.items():
            n = len(rs); m = sum(rs) / n
            cache["bc"][(b, c)] = m * n / (n + _RES_V2_N0)
        for b, rs in per_b.items():
            n = len(rs); m = sum(rs) / n
            cache["b"][b] = m * n / (n + _RES_V2_N0)
        _RES_V2_ROUND_CACHE[key] = cache

    if (benchmark, condition) in cache["bc"]:
        return cache["bc"][(benchmark, condition)]
    return cache["b"].get(benchmark, 0.0)

# CALIB_PATCH_MARKER
import json as _json
_TEMP_CALIB: dict = {}
_ISO_CALIB: dict = {}
_CALIB_PREFER = "temp"
if (HERE / "temp_calib.pkl").exists():
    try:
        with open(HERE / "temp_calib.pkl", "rb") as _f:
            _TEMP_CALIB = pickle.load(_f) or {}
        print(f"[model_mcmc] temp calibrators: {len(_TEMP_CALIB)} entries")
    except Exception as _e:
        print(f"[model_mcmc] WARN: temp_calib.pkl load: {_e}")
if (HERE / "iso_calib.pkl").exists():
    try:
        with open(HERE / "iso_calib.pkl", "rb") as _f:
            _ISO_CALIB = pickle.load(_f) or {}
        print(f"[model_mcmc] iso calibrators: {len(_ISO_CALIB)} entries")
    except Exception as _e:
        print(f"[model_mcmc] WARN: iso_calib.pkl load: {_e}")
if (HERE / "calib_choice.json").exists():
    try:
        with open(HERE / "calib_choice.json") as _f:
            _CALIB_PREFER = _json.load(_f).get("prefer", "temp")
        print(f"[model_mcmc] calibration prefer: {_CALIB_PREFER}")
    except Exception:
        pass


def _apply_calib(p: float, bench: str) -> float:
    if _CALIB_PREFER == "iso" and _ISO_CALIB:
        cal = _ISO_CALIB.get(bench) or _ISO_CALIB.get("__global__")
        if cal is not None:
            grid = cal["grid"]; cg = cal["cal_grid"]
            return float(np.interp(p, grid, cg))
    if _CALIB_PREFER == "temp" and _TEMP_CALIB:
        cal = _TEMP_CALIB.get(bench) or _TEMP_CALIB.get("__global__")
        if cal is not None:
            tau = float(cal["tau"])
            pe = min(max(p, 1e-6), 1 - 1e-6)
            z = math.log(pe / (1 - pe))
            z2 = tau * z
            if z2 >= 0:
                return 1.0 / (1.0 + math.exp(-z2))
            e = math.exp(z2)
            return e / (1.0 + e)
    return p



# --- kNN density index for per-item prior-width modulation -----------------
# Loads pre-encoded training-item embeddings (all 70k items, ~210 MB) so we
# can ask, for each test item: "how close is this to the training items the
# text head actually saw?" Cold-start items far from anything in training
# get a wider per-item sigma_b, pulling predictions toward the bench/cond mean.
_KNN_ITEMS_EMB: np.ndarray | None = None
_KNN_K = 10
_KNN_LAMBDA = 2.0      # per-item sigma_b multiplier: 1 + lambda * mean_cos_dist
if (HERE / "embeddings.pkl").exists():
    try:
        with open(HERE / "embeddings.pkl", "rb") as _f:
            _emb_blob = pickle.load(_f)
        _KNN_ITEMS_EMB = np.asarray(_emb_blob["item_emb"], dtype=np.float32)
        print(f"[model_mcmc] kNN index: {_KNN_ITEMS_EMB.shape[0]:,} items")
    except Exception as _e:
        print(f"[model_mcmc] WARN: failed to load embeddings.pkl: {_e}")
else:
    print("[model_mcmc] no embeddings.pkl found; kNN density disabled")


def _knn_density_modifier(v: np.ndarray) -> float:
    """Return (1 + lambda * mean_top_k_cosine_distance) for the test item.

    v is already L2-normalized; training-item embeddings are also normalized,
    so v @ V.T gives cosine similarities. We take the top-K largest sims
    (closest neighbors) and use 1 - mean_top_K as distance.
    """
    if _KNN_ITEMS_EMB is None:
        return 1.0
    sims = _KNN_ITEMS_EMB @ v  # [N_train]
    # Top-K closest neighbors (largest cosine similarities)
    if sims.shape[0] > _KNN_K:
        topk = np.partition(sims, -_KNN_K)[-_KNN_K:]
    else:
        topk = sims
    mean_sim = float(topk.mean())
    dist = max(0.0, 1.0 - mean_sim)
    return 1.0 + _KNN_LAMBDA * dist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


_EMB_CACHE: dict[str, np.ndarray] = {}


def _encode(text: str) -> np.ndarray:
    cached = _EMB_CACHE.get(text)
    if cached is not None:
        return cached
    vec = _ENCODER.encode(text, convert_to_numpy=True,
                          normalize_embeddings=True,
                          show_progress_bar=False).astype(np.float32)
    _EMB_CACHE[text] = vec
    return vec


def _item_text_prior(v: np.ndarray, bench: str, cond: str) -> tuple[float, float]:
    """Run item_head to get (b_hat, log_a_hat) for one item.

    If `bench` or `cond` are unknown, fall back to uniform one-hots.
    """
    b_oh = np.zeros(len(_BENCHMARK_TO_IDX), dtype=np.float32)
    b_idx = _BENCHMARK_TO_IDX.get(bench)
    if b_idx is not None:
        b_oh[b_idx] = 1.0
    c_oh = np.zeros(len(_CONDITION_TO_IDX), dtype=np.float32)
    c_idx = _CONDITION_TO_IDX.get(cond)
    if c_idx is not None:
        c_oh[c_idx] = 1.0
    x = np.concatenate([v, b_oh, c_oh], axis=0).astype(np.float32)
    with torch.no_grad():
        out = _ITEM_HEAD(torch.from_numpy(x).unsqueeze(0).to(_DEVICE)).cpu().numpy()
    return float(out[0, 0]), float(out[0, 1])


def _ncf_logit(u: np.ndarray, v: np.ndarray) -> float:
    with torch.no_grad():
        out = _NCF(torch.from_numpy(u).unsqueeze(0).to(_DEVICE),
                   torch.from_numpy(v).unsqueeze(0).to(_DEVICE)).cpu().numpy()
    return float(out[0])


_RNG = np.random.default_rng(42)


# Load MIRT projection heads if present (for v3 K-dim factor model).
_MIRT_HEADS = None
_MIRT_K = None
_W_SUBJ = None  # [K, 768] numpy
_W_ITEM = None  # [K, 768] numpy
if (HERE / "mirt_heads.pt").exists():
    try:
        _MIRT_HEADS = torch.load(HERE / "mirt_heads.pt", map_location=_DEVICE, weights_only=False)
        _MIRT_K = int(_MIRT_HEADS["K"])
        _W_SUBJ = _MIRT_HEADS["state_dict"]["subj_proj.weight"].cpu().numpy().astype(np.float32)
        _W_ITEM = _MIRT_HEADS["state_dict"]["item_proj.weight"].cpu().numpy().astype(np.float32)
        print(f"[model_mcmc] MIRT heads: K={_MIRT_K}  W_subj {_W_SUBJ.shape}  W_item {_W_ITEM.shape}")
    except Exception as _e:
        print(f"[model_mcmc] WARN: failed to load mirt_heads.pt: {_e}")


def _posterior_predictive_mirt(post: dict, subj_idx: int, cond_idx: int | None,
                                u: np.ndarray, v: np.ndarray, ncf_lg: float,
                                density_mod: float = 1.0) -> float:
    """MIRT (v3) posterior-predictive: K-dim factor model.

    Subject side uses stored per-sample U_s posterior + b_subj_post.
    Item side is cold-start -- draw fresh V_i ~ N(V_text(item), sigma_V).
    """
    K = int(post["K"])
    # Per-sample U_s for this subject (we stored mean+std, not full samples to
    # save space) -- reconstruct draws around the posterior mean.
    U_mean = post["U_post_mean"][subj_idx]   # [K]
    U_std = post.get("U_post_std")
    if U_std is not None:
        U_std = U_std[subj_idx]              # [K]
    b_subj_mean = float(post["b_subj_post_mean"][subj_idx])
    b_subj_std = float(post.get("b_subj_post_std", np.zeros_like(post["b_subj_post_mean"]))[subj_idx])

    # Per-sample hyperparams
    sigma_V = post["sigma_V_samples"] * density_mod   # [S]
    sigma_bitem = post["sigma_bitem_samples"]
    beta = post["beta_samples"]
    if cond_idx is not None and cond_idx < post["tau_samples"].shape[1]:
        tau = post["tau_samples"][:, cond_idx]
    else:
        tau = post["tau_samples"].mean(axis=1)
    S = sigma_V.shape[0]

    # V_text from item embedding
    V_text = _W_ITEM @ v                              # [K]
    # b_item prior: -b_hat (1-D text head gives "ease", IRT uses "difficulty")
    # We pass b_hat into the posterior pred via the caller. Here use 0; we'll
    # let the b_subj_post absorb the offset.
    b_item_mean = 0.0  # neutral; per-item ~0 prior since text head not used here

    # Draw fresh samples for cold-start item factors
    eps_V = _RNG.standard_normal((S, K)).astype(np.float32)
    eps_b = _RNG.standard_normal(S).astype(np.float32)
    V = V_text[None, :] + sigma_V[:, None] * eps_V                # [S, K]
    b_item = b_item_mean + sigma_bitem * eps_b                    # [S]

    # Subject side: per-sample U_s -- approximate as Normal(U_mean, U_std)
    if U_std is not None:
        eps_U = _RNG.standard_normal((S, K)).astype(np.float32)
        U_s = U_mean[None, :] + U_std[None, :] * eps_U            # [S, K]
    else:
        U_s = np.broadcast_to(U_mean[None, :], (S, K))
    eps_bs = _RNG.standard_normal(S).astype(np.float32)
    b_subj = b_subj_mean + b_subj_std * eps_bs                    # [S]

    dot = (U_s * V).sum(axis=-1)                                  # [S]
    logit = dot + b_subj + b_item + tau + beta * ncf_lg
    logit_clipped = np.clip(logit, -30.0, 30.0)
    p_irt = 1.0 / (1.0 + np.exp(-logit_clipped))
    c_samples = post.get("c_bench_samples")
    if c_samples is not None:
        p = c_samples + (1.0 - c_samples) * p_irt
    else:
        p = p_irt
    return float(p.mean())


def _posterior_predictive(post: dict, theta_idx: int, cond_idx: int | None,
                          b_hat: float, log_a_hat: float, ncf_lg: float,
                          density_mod: float = 1.0) -> float:
    """1-D IRT posterior-predictive (used when MIRT samples not present)."""
    theta = post["theta_samples"][:, theta_idx]     # [S]
    if cond_idx is not None and cond_idx < post["tau_samples"].shape[1]:
        tau = post["tau_samples"][:, cond_idx]      # [S]
    else:
        tau = post["tau_samples"].mean(axis=1)
    sigma_b = post["sigma_b_samples"] * density_mod
    sigma_a = post["sigma_a_samples"] * density_mod
    beta = post["beta_samples"]
    S = theta.shape[0]

    eps_b = _RNG.standard_normal(S).astype(np.float32)
    eps_a = _RNG.standard_normal(S).astype(np.float32)
    b = b_hat + sigma_b * eps_b
    log_a = log_a_hat + sigma_a * eps_a
    log_a = np.clip(log_a, -3.0, 3.0)
    a = np.exp(log_a)

    logit = a * (theta - b) + tau + beta * ncf_lg
    logit_clipped = np.clip(logit, -30.0, 30.0)
    p_irt = 1.0 / (1.0 + np.exp(-logit_clipped))
    c_samples = post.get("c_bench_samples")
    if c_samples is not None:
        p = c_samples + (1.0 - c_samples) * p_irt
    else:
        p = p_irt
    return float(p.mean())


def _global_fallback(b_hat: float, log_a_hat: float, ncf_lg: float,
                     theta_prior: float = 0.0) -> float:
    """If no posterior for this benchmark, fall back to a plug-in estimate."""
    a = math.exp(min(max(log_a_hat, -3.0), 3.0))
    logit = a * (theta_prior - b_hat) + 0.5 * ncf_lg
    return 1.0 / (1.0 + math.exp(-logit))


def _clamp(p: float, lo: float = 0.10, hi: float = 0.90) -> float:
    """Tighter clamp than [0.05, 0.95] -- caps per-item log-loss at ~2.3 nats.
    3PL provides a per-bench guess floor, so the clamp is a safety net only."""
    return max(lo, min(hi, p))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _predict_inner(input: dict, labeled: list[dict] | None = None) -> float:
    # `labeled` is accepted for interface compatibility but currently unused:
    # the MCMC posterior already includes subject uncertainty, and at K=5 the
    # marginal gain from a per-round Bayesian update was too small to bother.
    subject_content = input.get("subject_content", "") or ""
    item_content = input.get("item_content", "") or ""
    benchmark = input.get("benchmark", "") or ""
    condition = input.get("condition", "") or ""

    u = _encode(subject_content)
    v = _encode(item_content)
    b_hat, log_a_hat = _item_text_prior(v, benchmark, condition)
    ncf_lg = _ncf_logit(u, v)

    post = _POSTERIORS.get(benchmark)
    if post is None:
        # No posterior for this benchmark -- use plug-in estimate.
        return _clamp(_apply_calib(_global_fallback(b_hat, log_a_hat, ncf_lg), benchmark))

    subj_name = _parse_subject_name(subject_content)
    subj_idx = _SUBJ_LOOKUP.get(benchmark, {}).get(subj_name)
    is_mirt = bool(post.get("use_mirt", False)) and _W_SUBJ is not None
    cond_idx = post["conds"].index(condition) if condition in post["conds"] else None
    density_mod = _knn_density_modifier(v) if _KNN_ITEMS_EMB is not None else 1.0

    if is_mirt:
        if subj_idx is None:
            # Subject not in this benchmark's MIRT posterior -- average over all.
            # Approximate by mean-subject U_post (over the subj plate).
            U_mean = post["U_post_mean"].mean(axis=0)
            U_std = post.get("U_post_std")
            U_std = U_std.mean(axis=0) if U_std is not None else None
            b_subj_mean = float(post["b_subj_post_mean"].mean())
            b_subj_std = float(post.get("b_subj_post_std",
                np.zeros_like(post["b_subj_post_mean"])).mean())
            # Inject into post dict-like wrapper
            post_wrap = dict(post)
            post_wrap["U_post_mean"] = U_mean[None, :]
            post_wrap["U_post_std"] = U_std[None, :] if U_std is not None else None
            post_wrap["b_subj_post_mean"] = np.array([b_subj_mean])
            post_wrap["b_subj_post_std"] = np.array([b_subj_std])
            p = _posterior_predictive_mirt(post_wrap, 0, cond_idx, u, v, ncf_lg,
                                            density_mod=density_mod)
        else:
            p = _posterior_predictive_mirt(post, subj_idx, cond_idx, u, v, ncf_lg,
                                            density_mod=density_mod)
        return _clamp(_apply_calib(p, benchmark))

    # ---- 1-D IRT fallback (iter 1/2 posteriors) ----
    if subj_idx is None:
        theta_avg = post["theta_samples"].mean(axis=1)
        sigma_b = post["sigma_b_samples"] * density_mod
        sigma_a = post["sigma_a_samples"] * density_mod
        beta = post["beta_samples"]
        S = theta_avg.shape[0]
        eps_b = _RNG.standard_normal(S).astype(np.float32)
        eps_a = _RNG.standard_normal(S).astype(np.float32)
        b = b_hat + sigma_b * eps_b
        log_a = np.clip(log_a_hat + sigma_a * eps_a, -3.0, 3.0)
        a = np.exp(log_a)
        if cond_idx is not None:
            tau = post["tau_samples"][:, cond_idx]
        else:
            tau = post["tau_samples"].mean(axis=1)
        logit = a * (theta_avg - b) + tau + beta * ncf_lg
        logit_clipped = np.clip(logit, -30.0, 30.0)
        p_irt = 1.0 / (1.0 + np.exp(-logit_clipped))
        c_samples = post.get("c_bench_samples")
        if c_samples is not None:
            p = c_samples + (1.0 - c_samples) * p_irt
        else:
            p = p_irt
        return _clamp(_apply_calib(float(p.mean()), benchmark))

    p = _posterior_predictive(post, subj_idx, cond_idx, b_hat, log_a_hat, ncf_lg,
                              density_mod=density_mod)
    return _clamp(_apply_calib(p, benchmark))




def predict(input: dict, labeled: list[dict] | None = None) -> float:
    p = _predict_inner(input, labeled=labeled)
    benchmark = (input.get("benchmark", "") or "")
    condition = (input.get("condition", "") or "")
    delta = _residual_v2_get_delta(benchmark, condition, labeled)
    p_corrected = max(0.001, min(0.999, p + delta))
    return _clamp(p_corrected)


def predict_batch(inputs, labeled=None):
    return [predict(x, labeled=labeled) for x in inputs]
