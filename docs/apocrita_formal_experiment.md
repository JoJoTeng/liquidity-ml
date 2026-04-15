# Apocrita Run: Formal Experiment (liquidity-ml, formalanalysis branch)

> **Project:** LiquidityML v3 — 2x2 formal experiment (Sections 7–9 of the paper).
> **Branch:** `formalanalysis`
> **Goal:** Train all 12 model × weight specifications on HPC and generate the full set of tables and figures for the paper.

---

## What This Run Produces

| Dimension | Values |
|---|---|
| Models | Elastic-Net, XGBoost, Neural Network |
| Weight families | `dolvol` (Eq. 21, AUM-independent), `tc` (Eq. 23, AUM-dependent) |
| AUM scenarios | $100M, $500M, $1B (TC only) |
| Total specs | **12 jobs** = 3 models × (1 dolvol + 3 TC AUM levels) |

Each job trains:
- **M_std** — standard (unweighted) model, shared across weight families within a model
- **M_w** — weighted model for that specific weight spec

**M_std sharing:** The first job to run for a given model creates `outputs/formalanalysis/experiment/{model}/standard/predictions.parquet`. Subsequent jobs for the same model check for this file and skip M_std training. So effectively you train 3 M_std's + 12 M_w's = 15 model fits across rolling windows.

---

## Differences vs. Motivation Run (Apocrita_optuna.md)

| Aspect | Motivation run | Formal experiment |
|---|---|---|
| Script | `07_step3_ml_diagnostics.py` + `10_quintile...` + `12_progressive...` | `02_run_experiment.py` (unified) |
| Models | XGBoost only (M_std) | Elastic-Net + XGBoost + NN (M_std + M_w) |
| Weighting | None (standard) | dolvol + tc weights |
| Hyperparameter tuning | Grid search (81 combos) | Grid search (81 combos XGB, 5 EN, 4 NN) |
| Dependencies | xgboost, sklearn, pandas | + tensorflow, shap |
| Output dir | `outputs/motivation/step3/dvol/` | `outputs/formalanalysis/experiment/{model}/{weight}/` |

---

## Part A: First-Time Setup on Apocrita

### A1. Login

```bash
ssh tew775@login.hpc.qmul.ac.uk
```

### A2. Clone project from GitHub (branch `formalanalysis`)

On **Apocrita**:

```bash
cd /data/home/tew775
git clone -b formalanalysis https://github.com/JoJoTeng/liquidity-ml.git liquidity-ml-formal
cd liquidity-ml-formal
git log --oneline -3
```

> **Note:** We use a separate directory `liquidity-ml-formal` so the formal experiment does not interfere with any existing `liquidity-ml` directory on Apocrita.

### A3. Upload data files from Mac (not in git)

The `data/` directory is not tracked in git (too large). Upload only the files the pipeline needs. From your **local Mac**:

```bash
# Panel data + feature list
scp /Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/data/signed_predictors_all_wide.parquet \
  tew775@login.hpc.qmul.ac.uk:/data/home/tew775/liquidity-ml-formal/data/

scp /Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/data/feature_list.json \
  tew775@login.hpc.qmul.ac.uk:/data/home/tew775/liquidity-ml-formal/data/

# Fama-French factors CSV (for factor alpha regressions)
scp /Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/data/FFResearch_Data_Factors.csv \
  tew775@login.hpc.qmul.ac.uk:/data/home/tew775/liquidity-ml-formal/data/

# SignalDoc (used by motivation.py helpers; safe to upload even if unused here)
scp /Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/data/SignalDoc.csv \
  tew775@login.hpc.qmul.ac.uk:/data/home/tew775/liquidity-ml-formal/data/ 2>/dev/null || true
```

You also need the Step 2 + Step 3 motivation outputs if you want `03_analyze_results.py` to produce Prediction 2 (ΔĪ_j regression + group shares) and Prediction 3 (restriction curve overlay). Upload them:

```bash
scp -r /Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/outputs/motivation/step2 \
  tew775@login.hpc.qmul.ac.uk:/data/home/tew775/liquidity-ml-formal/outputs/motivation/

scp -r /Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/outputs/motivation/step3_restriction_rerank \
  tew775@login.hpc.qmul.ac.uk:/data/home/tew775/liquidity-ml-formal/outputs/motivation/
```

If you only want to run training (and do analysis locally), skip this step — training does not need those files.

### A4. Create virtualenv & install dependencies

Back on **Apocrita**:

```bash
cd /data/home/tew775/liquidity-ml-formal
module load python
python -m venv ~/formal_env
source ~/formal_env/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn xgboost statsmodels matplotlib seaborn \
  pyyaml pyarrow arch shap tensorflow
```

> **Note on TensorFlow:** If you only need Elastic-Net + XGBoost (skipping NN for now), you can omit `tensorflow` and save ~500MB. Add it later if you want NN results.

### A5. Verify

```bash
python -c "import xgboost; print('XGBoost', xgboost.__version__)"
python -c "import sklearn; print('sklearn', sklearn.__version__)"
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
python -c "import shap; print('SHAP', shap.__version__)"
python -c "from src.weighting import compute_weights; print('weighting OK')"
python -c "from src.models import create_model; m = create_model('elastic_net'); print('elastic_net OK')"
python -c "from src.models import create_model; m = create_model('xgboost'); print('xgboost OK')"
python -c "from src.models import create_model; m = create_model('neural_network'); print('neural_network OK')"
```

### A6. Verify data

```bash
cd /data/home/tew775/liquidity-ml-formal
ls -lh data/signed_predictors_all_wide.parquet data/feature_list.json
```

Both must exist before submitting jobs.

### A7. Verify branch

```bash
cd /data/home/tew775/liquidity-ml-formal
git status
git log --oneline -3
```

Should show you are on `formalanalysis` branch with the latest commits (`Fix formal analysis audit findings` + `LiquidityML v3 formal experiment pipeline`).

---

## Part B: Returning to Existing Setup

```bash
ssh tew775@login.hpc.qmul.ac.uk
cd /data/home/tew775/liquidity-ml-formal
source ~/formal_env/bin/activate
```

### Update code from GitHub

The standard workflow: commit + push from Mac, then pull on Apocrita.

```bash
# On Mac
cd /Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml
git add -A
git commit -m "your change"
git push origin formalanalysis

# On Apocrita
cd /data/home/tew775/liquidity-ml-formal
git fetch origin
git pull origin formalanalysis
git log --oneline -3  # confirm latest commit matches Mac
```

If you have uncommitted work on Apocrita that you need to keep, stash first:

```bash
git stash
git pull origin formalanalysis
git stash pop
```

Clean up old jobs:

```bash
rm -f jobs/*.sh logs/*
```

---

## Part C: Create & Submit Jobs (12 total)

### Overview

| Job type | Count | Runtime (est.) | Notes |
|---|---|---|---|
| Elastic-Net × 4 weight specs | 4 | ~4h each | Fast (linear model) |
| XGBoost × 4 weight specs | 4 | ~24h each | Medium (81-combo grid × many windows) |
| Neural Network × 4 weight specs | 4 | ~48h each | Slow (TF + ensemble optional) |

**Total wall-clock if run sequentially:** ~300h (~12 days). **In parallel:** ~48h (limited by slowest NN job).

### M_std sharing

The first job to run per model trains M_std and saves it. Subsequent jobs for the same model reuse it. To maximize parallelism, submit **one job per model first** (to train M_std), then submit the other 3 jobs per model with a dependency. Or submit all 12 at once and accept that M_std gets trained once per model by whichever job wins the race.

**Recommended: submit all 12 independently.** M_std check-and-skip logic is based on file existence; if two jobs start M_std at the same time they will both train it (wasting ~24h × 2 for XGBoost). To avoid this, use a dependency chain (see C2).

### C1. Option 1 — Use the provided generator script

The repo already includes `scripts/generate_hpc_jobs.sh`. First, update the paths inside it to match Apocrita:

```bash
cd /data/home/tew775/liquidity-ml-formal

# Edit the two paths in generate_hpc_jobs.sh
sed -i 's|PROJECT_DIR=.*|PROJECT_DIR="/data/home/tew775/liquidity-ml-formal"|' \
  scripts/generate_hpc_jobs.sh
sed -i 's|VENV=.*|VENV="source ~/formal_env/bin/activate"|' \
  scripts/generate_hpc_jobs.sh

# Generate jobs
bash scripts/generate_hpc_jobs.sh
```

This creates `jobs/{elastic_net,xgboost,neural_network}_{dolvol,tc_100m,tc_500m,tc_1000m}.sh` — 12 job scripts total.

### C2. Option 2 — Dependency chain (M_std shared cleanly)

If you want to avoid duplicate M_std training, submit the DolVol job first per model (which trains M_std), then submit the TC jobs with `--dependency=afterok` on the DolVol job:

```bash
cd /data/home/tew775/liquidity-ml-formal
mkdir -p jobs logs

# Generate job scripts first
bash scripts/generate_hpc_jobs.sh

# Submit with dependency chain
for MODEL in elastic_net xgboost neural_network; do
    DOLVOL_ID=$(sbatch --parsable jobs/${MODEL}_dolvol.sh)
    echo "${MODEL} dolvol (M_std trained here): $DOLVOL_ID"
    for AUM in 100 500 1000; do
        sbatch --dependency=afterok:$DOLVOL_ID jobs/${MODEL}_tc_${AUM}m.sh
    done
done
echo "Submitted 12 jobs total (3 DolVol + 9 TC with dependency)"
```

### C3. Option 3 — Submit all 12 independently (fastest wall clock, may duplicate M_std)

```bash
cd /data/home/tew775/liquidity-ml-formal
bash scripts/generate_hpc_jobs.sh --submit
```

This submits all 12 jobs at once. Each model's M_std may be trained by the first TC/dolvol job to start.

### C4. Monitor

```bash
squeue --me                            # all jobs
squeue --me -t RUNNING                 # only running
sacct --format=JobID,JobName,State,Elapsed,MaxRSS -j <jobid>  # after completion

# Training logs go to .err (Python's logging writes to stderr by default)
cat logs/xgboost_dolvol_*.err          # check XGBoost dolvol progress
tail -f logs/xgboost_dolvol_*.err      # live stream
```

> **Why `.err` not `.out`:** Python's `logging` module writes all INFO/WARNING/ERROR to stderr. The `.out` file only contains `module load python` stdout (nothing useful for progress tracking).

Expected `squeue` output after dependency submission:
- `elastic_net_dolvol` — `R` (running, trains EN M_std)
- `xgboost_dolvol` — `R` (running, trains XGB M_std)
- `neural_network_dolvol` — `R` (running, trains NN M_std)
- `*_tc_*` — `PD (Dependency)` until corresponding dolvol finishes

---

## Part D: After All Jobs Complete

### D1. Verify outputs on Apocrita

```bash
ssh tew775@login.hpc.qmul.ac.uk
cd /data/home/tew775/liquidity-ml-formal

for MODEL in elastic_net xgboost neural_network; do
    echo "=== $MODEL ==="
    ls -lh outputs/formalanalysis/experiment/$MODEL/standard/predictions.parquet 2>/dev/null
    for WT in dolvol tc_100m tc_500m tc_1000m; do
        echo "  $WT:"
        ls -lh outputs/formalanalysis/experiment/$MODEL/$WT/predictions.parquet 2>/dev/null
    done
done
```

Each row should show a non-zero-size `predictions.parquet`.

### D2. Run analysis on Apocrita (optional — can do locally instead)

```bash
cd /data/home/tew775/liquidity-ml-formal
source ~/formal_env/bin/activate

# Full analysis: Predictions 1-4, Tables 11/12, hypothesis tests
python scripts/03_analyze_results.py

# Or per-model for debugging
python scripts/03_analyze_results.py --model xgboost
python scripts/03_analyze_results.py --model elastic_net
python scripts/03_analyze_results.py --model neural_network
```

### D3. Download results to Mac

From your **local Mac**:

```bash
cd /Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml

# Experiment outputs (predictions + importance files)
scp -r "tew775@login.hpc.qmul.ac.uk:/data/home/tew775/liquidity-ml-formal/outputs/formalanalysis/experiment" \
  outputs/formalanalysis/

# Analysis outputs (tables + figures) — only if you ran D2 on Apocrita
scp -r "tew775@login.hpc.qmul.ac.uk:/data/home/tew775/liquidity-ml-formal/outputs/formalanalysis/analysis" \
  outputs/formalanalysis/
```

### D4. Run analysis locally

```bash
cd /Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml
python scripts/03_analyze_results.py
```

This generates everything in `outputs/formalanalysis/analysis/`.

---

## Output Locations

```
outputs/formalanalysis/
├── experiment/                             # From 02_run_experiment.py
│   ├── elastic_net/
│   │   ├── standard/                       # M_std (shared across weight families)
│   │   │   ├── predictions.parquet
│   │   │   ├── importance_shap.csv
│   │   │   ├── importance_native.csv
│   │   │   └── tuned_params.csv
│   │   ├── dolvol/                         # M_w (weighted by dolvol)
│   │   │   ├── predictions.parquet
│   │   │   ├── importance_shap.csv
│   │   │   ├── importance_native.csv
│   │   │   ├── tuned_params.csv
│   │   │   └── meta.json
│   │   ├── tc_100m/                        # M_w (weighted by TC at $100M AUM)
│   │   ├── tc_500m/
│   │   └── tc_1000m/
│   ├── xgboost/                            # (same subfolder structure)
│   └── neural_network/                     # (same subfolder structure)
│
└── analysis/                               # From 03_analyze_results.py
    ├── tables/
    │   ├── prediction_1_r2_{spec}.csv              # R² by quintile
    │   ├── prediction_1_utility_r2_{spec}.csv      # Utility-weighted R²
    │   ├── prediction_2_{spec}.csv                 # SHAP shift per feature
    │   ├── prediction_2_regression_{spec}.json     # ΔĪ_j ~ γ̄_j regression
    │   ├── prediction_2_group_shares_{spec}.csv    # Q1/Q5/both importance shares
    │   ├── prediction_3_{spec}.csv                 # Weighted vs restriction curve
    │   ├── prediction_4_{spec}.csv                 # Cumulative SE differential
    │   ├── table_11_{spec}.csv                     # Within-quintile SR
    │   ├── table_12_{spec}_100M.csv                # 2x2 decomposition at $100M
    │   ├── table_12_{spec}_500M.csv                # $500M (primary)
    │   ├── table_12_{spec}_1B.csv                  # $1B
    │   └── hypothesis_tests.json                   # Consolidated H1–H4 tests
    └── figures/
        ├── importance_shift_{spec}.png             # Prediction 2 bar chart
        ├── restriction_curve_{spec}.png            # Prediction 3 overlay
        └── se_diff_{spec}.png                      # Prediction 4 time series
```

where `{spec}` = e.g. `xgboost_dolvol`, `xgboost_tc_500m`, `elastic_net_dolvol`, etc. (12 specs total).

---

## Resource Requests

| Resource | Value | Rationale |
|---|---|---|
| `-t` | 240:0:0 (10 days max) | Conservative upper bound; most jobs finish in ≤48h |
| `--mem-per-cpu` | 8G (32G total with 4 cores) | Panel is ~8GB; XGBoost/NN add overhead |
| `-n` | 4 cores | XGBoost parallel tree building; TF ops parallelisation |

NN jobs may need more memory if TF allocates a lot. If you hit OOM:

```bash
# Increase mem per CPU to 16G
sed -i 's|mem-per-cpu=8G|mem-per-cpu=16G|' jobs/neural_network_*.sh
```

---

## Troubleshooting

### Check if a job failed

```bash
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS
cat logs/<jobname>_<jobid>.err
cat logs/<jobname>_<jobid>.out
```

### Common issues

**`ModuleNotFoundError: No module named 'tensorflow'`**
```bash
source ~/formal_env/bin/activate
pip install tensorflow
```

**`ModuleNotFoundError: No module named 'shap'`**
```bash
source ~/formal_env/bin/activate
pip install shap
```

**`feature_list.json not found`** — Not in git; upload from Mac:
```bash
# From Mac
scp /Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/data/feature_list.json \
  tew775@login.hpc.qmul.ac.uk:/data/home/tew775/liquidity-ml-formal/data/
```

**`signed_predictors_all_wide.parquet not found`** — Same — upload from Mac:
```bash
scp /Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/data/signed_predictors_all_wide.parquet \
  tew775@login.hpc.qmul.ac.uk:/data/home/tew775/liquidity-ml-formal/data/
```

**`git pull` shows conflicts** — Someone (or you) committed on a different branch. Reset hard to `origin/formalanalysis`:
```bash
cd /data/home/tew775/liquidity-ml-formal
git fetch origin
git reset --hard origin/formalanalysis
```

**M_std skipped but weighted job crashed mid-run** — Delete the incomplete weighted directory and resubmit:
```bash
rm -rf outputs/formalanalysis/experiment/xgboost/dolvol
sbatch jobs/xgboost_dolvol.sh
```

**M_std needs to be re-trained (e.g. after code change)** — Delete the standard directory:
```bash
rm -rf outputs/formalanalysis/experiment/xgboost/standard
sbatch jobs/xgboost_dolvol.sh  # will re-train M_std + M_w (dolvol)
```

### Cancelling jobs

```bash
scancel <jobid>                # single job
scancel -u tew775              # all your jobs
scancel --name=xgboost_dolvol  # by name
```

### Manual re-run of a single job

```bash
cd /data/home/tew775/liquidity-ml-formal
source ~/formal_env/bin/activate

# Quick test (2020-2024, ~60 OOS months)
python scripts/02_run_experiment.py --model xgboost --weights dolvol --quick

# Full run (2000-2024)
python scripts/02_run_experiment.py --model xgboost --weights dolvol
```

---

## Quick Sanity Check Before Full Submission

> **Important:** Apocrita's login node does NOT allow long-running processes (>10 min wall time). You cannot just `python scripts/02_run_experiment.py` on the login node — the cluster will kill it. Always go through the scheduler (`sbatch`) even for quick tests.

Submit a short test job that uses `--quick` (OOS 2020-2024, ~30 min runtime on 4 cores):

```bash
ssh tew775@login.hpc.qmul.ac.uk
cd /data/home/tew775/liquidity-ml-formal
mkdir -p jobs logs

cat > jobs/quick_test.sh << 'JOBEOF'
#!/bin/bash
#SBATCH -J quick_test
#SBATCH -n 4
#SBATCH --mem-per-cpu=8G
#SBATCH -t 4:0:0
#SBATCH -o logs/quick_test_%j.out
#SBATCH -e logs/quick_test_%j.err
module load python
source ~/formal_env/bin/activate
export OMP_NUM_THREADS=${SLURM_NTASKS}
export PYTHONHASHSEED=0
cd /data/home/tew775/liquidity-ml-formal
python scripts/02_run_experiment.py --model xgboost --weights dolvol --quick
JOBEOF

sbatch jobs/quick_test.sh
```

Monitor:

```bash
squeue --me                         # should show quick_test in PD or R state
tail -f logs/quick_test_*.err       # see note below about .err vs .out
```

> **Where the training logs go:** Python's `logging` module writes to **stderr** by default, so all the `[INFO]` training messages end up in `logs/quick_test_*.err`, NOT in `logs/quick_test_*.out`. The `.out` file only contains module-load output (`Loading python/3.11.7...`). This is normal — watch the `.err` file for training progress.

Expected lines in `.err` during a successful run:

```
[INFO] 02_experiment: QUICK MODE: OOS 2020-01 to 2024-12
[INFO] src.data.loader: Panel ready: ...
[INFO] 02_experiment: === Training M_std (standard) ===
[INFO] 02_experiment: Rolling training [std]: model=xgboost, months=59, ...
[INFO]   [std] Tuning at month 202001 (1/59)
[INFO] src.models.xgboost_model: XGBoost standard training: 1000 trees, ...
...
[INFO] 02_experiment: M_std complete: XXXX predictions in X.X min
[INFO] 02_experiment: === Training M_w (weighted: dolvol, aum=N/A) ===
...
[INFO] 02_experiment: M_w complete: XXXX predictions in X.X min
[INFO] 02_experiment: All done. Outputs in: ...
```

Check the output files after it finishes:

```bash
ls -lh outputs/formalanalysis/experiment/xgboost/standard/
ls -lh outputs/formalanalysis/experiment/xgboost/dolvol/
```

If everything looks good, delete the quick-test outputs (they use OOS=2020 instead of 2000) and submit the full jobs:

```bash
rm -rf outputs/formalanalysis/experiment/xgboost
bash scripts/generate_hpc_jobs.sh --submit
```

---

## Checklist

### Before submitting jobs
- [ ] `liquidity-ml-formal` cloned from GitHub on `formalanalysis` branch
- [ ] Latest commit on Apocrita matches latest commit on Mac (`git log --oneline -3`)
- [ ] `~/formal_env` created with all dependencies installed (`xgboost`, `tensorflow`, `shap`, `scikit-learn`, `statsmodels`, `pandas`, `pyarrow`, etc.)
- [ ] `data/signed_predictors_all_wide.parquet` uploaded via scp
- [ ] `data/feature_list.json` uploaded via scp
- [ ] `data/FFResearch_Data_Factors.csv` uploaded via scp
- [ ] (Optional) `outputs/motivation/step2/` and `outputs/motivation/step3_restriction_rerank/` uploaded if running Predictions 2 & 3 on Apocrita
- [ ] Import smoke tests pass (A5)
- [ ] Quick test with `--quick` flag succeeds (~30 min, optional but recommended)
- [ ] 12 job scripts generated in `jobs/`
- [ ] Paths in `jobs/*.sh` point to `/data/home/tew775/liquidity-ml-formal` and `~/formal_env`

### After jobs complete
- [ ] All 12 jobs finished (`squeue --me` shows empty)
- [ ] `predictions.parquet` exists for all 3 models × 4 weight specs (12 files) plus 3 standard/ (total 15 files)
- [ ] Optional: run `03_analyze_results.py` on Apocrita for a fast sanity check
- [ ] Download `outputs/formalanalysis/experiment/` to Mac
- [ ] Run `03_analyze_results.py` locally to generate tables + figures
- [ ] Inspect `outputs/formalanalysis/analysis/tables/hypothesis_tests.json` for H1/H3 p-values

---

## Estimated Timeline

Assuming good parallelism and no queue delays:

| Phase | Duration |
|---|---|
| Setup (A1–A6) | ~1h |
| Quick test (optional) | ~30 min |
| Full job wall-clock (parallel) | ~48h (limited by slowest NN job) |
| Download + analysis | ~30 min |
| **Total** | **~2.5 days** |

If queue is busy, add 6–24h waiting time per job launch.

---

## Related Documents

- `scripts/generate_hpc_jobs.sh` — the job generator script (baseline for job templates)
- `CLAUDE.md` — project architecture and build order
- `LiquidityML_v3.pdf` — Sections 7–9 for the theory behind the 2x2 framework
