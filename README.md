# PEC Submission

Final artifact: **`submission.zip`** (17 MB, at the root of this folder). Test score: **−0.61**.

Upload `submission.zip` directly to Codabench. Nothing else needs to run for the submission itself.


## Reproducing submission.zip

```bash
pip install -r requirements.txt
# Optional GPU MCMC:
pip install -U "jax[cuda12]"

mkdir -p work

# 0. Download data
python train_scripts/fetch_data.py

# 1. Train NCF + IRT + item_head (~3 hr CPU / 30 min GPU)
python train_scripts/train_full.py --submission-dir work/heads \
    --ncf-epochs 20 --ncf-lr 3e-4 --head-epochs 30 \
    --joint-bce-head --head-dropout 0.3 --head-weight-decay 1e-3 \
    --head-patience 5 --head-tune-weight --ncf-dropout 0.3 --no-package

# 2. Pre-encode subjects + items with mpnet (~5 min GPU)
python train_scripts/encode_all_items.py --out work/embeddings.pkl --device cuda

# 3. Train K=1 MIRT projection heads (~3 min)
python train_scripts/train_mirt_heads.py --K 1 --epochs 5 --batch 8192 --lr 1e-3 \
    --embeddings work/embeddings.pkl --out work/mirt_heads_k1.pt

# 4. Per-benchmark MCMC. Two options:
# --- (a) reproduce the -0.61 submission exactly via pinned outputs (default) ---
cp pinned/mirt_heads_k1.pt work/mirt_heads_k1.pt
mkdir -p work/posteriors && cp pinned/posteriors/*.pkl work/posteriors/
# --- (b) retrain instead (gives ~-0.62 due to MCMC chain variance across hardware) ---
# python train_scripts/fit_hier_irt.py --out work/posteriors \
#     --max-rows 30000 --num-warmup 1000 --num-samples 1000 \
#     --num-chains 4 --seed 42 \
#     --emb-cache work/embeddings.pkl --submission-dir work/heads \
#     --use-ncf --use-3pl --use-mirt --mirt-heads work/mirt_heads_k1.pt

# 5. Assemble submission base
mkdir -p work/submission/posteriors
cp submission_template/* work/submission/
cp work/heads/{ncf_head.pt,item_head.pt,vocab.pkl,ncf_meta.json} work/submission/
cp work/mirt_heads_k1.pt work/submission/mirt_heads.pt
cp work/posteriors/*.pkl work/submission/posteriors/

# 6. Fit per-bench isotonic calibration
python train_scripts/fit_calibration.py --sub-dir work/submission --max-per-bench 400
echo '{"prefer": "iso"}' > work/submission/calib_choice.json

# 7. Package
python train_scripts/package.py --submission-dir work/submission --out work/submission.zip
```
