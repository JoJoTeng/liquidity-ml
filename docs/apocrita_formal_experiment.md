# Apocrita Run: Full LiquidityML Pipeline

> **Project:** LiquidityML v3
> **Branch:** `formalanalysis`
> **HPC:** Apocrita / SLURM
> **Project path:** `/data/home/tew775/liquidity-ml`
> **Virtualenv:** `~/liquidml_env`
> **Rule:** do not run data/training Python directly on the login node. Generate and submit SLURM jobs.

This guide matches the current codebase and the current
`scripts/generate_hpc_jobs.sh` pipeline. The recommended flow is to run
`00_fetch_data.py` and `01_process_data.py` locally, upload the processed data
files, and let Apocrita start from Motivation Step 1 / script `02`.

---

## What Runs

The generator supports two modes:

```bash
bash scripts/generate_hpc_jobs.sh --from-processed --submit
```

Recommended. Skips `00` and `01`; assumes `processed_panel.parquet` and
`feature_list.json` were created locally and uploaded.

```bash
bash scripts/generate_hpc_jobs.sh --submit
```

Full HPC mode. Runs `00` and `01` on Apocrita too. This is less convenient
because `00` needs non-interactive WRDS authentication in a batch job.

The recommended `--from-processed` mode submits this dependency chain:

| Stage | Script | Purpose |
|---|---|---|
| 02 | `scripts/02_motivation_step1_divergence.py --liquidity dvol --vw` | Motivation Step 1 |
| 03 | `scripts/03_motivation_step2_heterogeneity.py --liquidity dvol --full` | Motivation Step 2, including full-feature outputs |
| 04 | `scripts/04_motivation_step3_ml_diagnostics.py --model {model} --liquidity dvol` | Standard model diagnostics and shared M_std cache |
| 05 | `scripts/05_motivation_step3d_progressive_restriction.py` | Progressive restriction, split by minimum quintile |
| 06 | `scripts/06_motivation_step3e_quintile_specific_models.py` | Quintile-specific models, split by quintile |
| 07 | `scripts/07_motivation_regime_analysis.py` | Regime data download + regime diagnostics |
| 20 | `scripts/20_formal_run_experiment.py` | Formal weighted model training |
| 21 | `scripts/21_formal_analyze_results.py` | Final formal tables and figures |

In `--from-processed` mode, the generator creates **62 SLURM job scripts**:

- 4 shared prerequisite jobs: `02`, `03`, `07_download`, `07_regime`
- 57 model-specific jobs: 19 jobs for each of `elastic_net`, `xgboost`, `neural_network`
- 1 final analysis job: `21_formal_analyze_results`

Full HPC mode creates **64 SLURM job scripts** because it also adds `00` and
`01`.

The formal experiment still has **21 model x weight specifications**:

| Dimension | Values |
|---|---|
| Models | `elastic_net`, `xgboost`, `neural_network` |
| Weight families | `dolvol`, `softmax_rank`, `tc` |
| Softmax lambdas | `2`, `3` |
| TC AUM scenarios | `$10M`, `$100M`, `$500M`, `$1B` |
| Formal specs | `3 models x (1 dolvol + 2 softmax + 4 TC) = 21` |

Formal outputs are saved under:

```text
outputs/formalanalysis/experiment/{model}/standard/
outputs/formalanalysis/experiment/{model}/dolvol/
outputs/formalanalysis/experiment/{model}/softmax_rank_lam2/
outputs/formalanalysis/experiment/{model}/softmax_rank_lam3/
outputs/formalanalysis/experiment/{model}/tc_10m/
outputs/formalanalysis/experiment/{model}/tc_100m/
outputs/formalanalysis/experiment/{model}/tc_500m/
outputs/formalanalysis/experiment/{model}/tc_1000m/
```

Motivation Step 3 outputs are saved under:

```text
outputs/motivation/step3/{model}/dvol/
outputs/motivation/step3_restriction/{model}/dvol/global/baseline/
outputs/motivation/step3_quintile/{model}/dvol/global/baseline/
outputs/motivation/step1_regime/xgboost/dvol/
```

`07` uses the `xgboost` namespace only because the regime diagnostics are
model-independent in the current code.

---

## Current Weight Definitions

`dolvol`:

```text
w_it = DolVol_it / mean_i(DolVol_it)
```

This is mean-normalized within each month, so average sample weight is 1.

`softmax_rank`:

```text
rank_it = within-month percentile rank of DolVol_it
w_it = exp(lambda * rank_it) / mean_i(exp(lambda * rank_it))
```

The formal grid is `lambda = 2` and `lambda = 3`. The current
`20_formal_run_experiment.py` requires `--softmax-lambda`; there is no ambiguous
default softmax output folder anymore.

`tc`:

```text
TC_it = Spread_it / 2 + lambda_market_impact * sigma_it * sqrt(Q_it / ADV_it)
w_it = exp(-alpha_t * TC_it)
```

Then `w_it` is mean-normalized within month. The sigma input is
`excess_sigma_12m_daily`, produced by `01_process_data.py` through
`load_panel()`.

---

## M_std Sharing

`04_motivation_step3_ml_diagnostics.py` trains the standard model for each
model family and writes the shared formal cache:

```text
outputs/formalanalysis/experiment/{model}/standard/
```

The formal weighted jobs in `20_formal_run_experiment.py` depend on the
corresponding Step 04 job. That means:

- M_std is trained once per model.
- Each formal `20` job trains only its weighted model if the standard cache is complete.
- The only intended difference between standard and weighted formal training is `sample_weight`.

This is why the generator does **not** submit all formal jobs independently.

---

## Before Resetting HPC

The current local code has many renamed and newly created files. Before deleting
and recloning on Apocrita, make sure the current branch has been committed and
pushed, or the HPC clone will get stale code.

Locally:

```bash
cd /Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml
git status
git branch --show-current
git remote -v
```

Expected branch:

```text
formalanalysis
```

After committing/pushing the current code, continue on HPC.

---

## Local Data Preparation

Run `00` and `01` locally first:

```bash
cd /Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml
python scripts/00_fetch_data.py
python scripts/01_process_data.py
```

After this, local `data/` should contain:

```text
data/signed_predictors_all_wide.csv
data/signed_predictors_all_wide.parquet
data/processed_panel.parquet
data/feature_list.json
```

The key files needed by the HPC jobs are `processed_panel.parquet` and
`feature_list.json`; `signed_predictors_all_wide.*` are useful to keep as
reproducibility backups, but Step `02` onward does not read them.

---

## Fresh HPC Setup

Login:

```bash
ssh tew775@login.hpc.qmul.ac.uk
cd /data/home/tew775
```

Safer reset:

```bash
mv liquidity-ml liquidity-ml_backup_$(date +%Y%m%d_%H%M%S)
git clone -b formalanalysis https://github.com/JoJoTeng/liquidity-ml.git liquidity-ml
cd liquidity-ml
```

If you really want to delete instead of backup:

```bash
rm -rf /data/home/tew775/liquidity-ml
git clone -b formalanalysis https://github.com/JoJoTeng/liquidity-ml.git liquidity-ml
cd liquidity-ml
```

Do **not** delete `~/liquidml_env` unless you plan to rebuild the Python
environment.

---

## Required Non-Git Data Files

The repo clone is not enough. For the recommended `--from-processed` HPC flow,
upload these files:

```text
data/processed_panel.parquet
data/feature_list.json
data/SignalDoc.csv
data/FFResearch_Data_Factors.csv
data/ff5_factors.csv
data/momentum_factor.csv
```

From your Mac:

```bash
cd /Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml
ssh tew775@login.hpc.qmul.ac.uk "mkdir -p /data/home/tew775/liquidity-ml/data"

scp data/processed_panel.parquet \
    data/feature_list.json \
    data/SignalDoc.csv \
    data/FFResearch_Data_Factors.csv \
    data/ff5_factors.csv \
    data/momentum_factor.csv \
  tew775@login.hpc.qmul.ac.uk:/data/home/tew775/liquidity-ml/data/
```

Optional backups to upload:

```bash
scp data/signed_predictors_all_wide.csv \
    data/signed_predictors_all_wide.parquet \
  tew775@login.hpc.qmul.ac.uk:/data/home/tew775/liquidity-ml/data/
```

Only if you choose full HPC mode, also upload:

```text
data/temp/signed_predictors_dl_wide.zip
```

Full HPC mode WRDS note: `00_fetch_data.py` calls `wrds.Connection(...)`. If
Apocrita cannot authenticate to WRDS non-interactively inside a batch job,
configure WRDS credentials before submitting. Otherwise job `00_fetch_data` may
fail or hang. This is why the local `00`/`01` flow is cleaner.

---

## Login Node Rule

Do not run these directly on the login node:

```bash
python scripts/00_fetch_data.py
python scripts/01_process_data.py
python scripts/04_motivation_step3_ml_diagnostics.py
python scripts/20_formal_run_experiment.py
python scripts/21_formal_analyze_results.py
```

Allowed on the login node:

```bash
bash -n scripts/generate_hpc_jobs.sh
bash scripts/generate_hpc_jobs.sh
squeue --me
sacct ...
tail -f logs/<job>_<jobid>.err
```

Python import checks are usually fine, but anything that reads the 2.4 GB panel
or trains a model should go through `sbatch`.

---

## Generate and Submit the Pipeline

On HPC:

```bash
cd /data/home/tew775/liquidity-ml
source ~/liquidml_env/bin/activate

bash -n scripts/generate_hpc_jobs.sh
bash scripts/generate_hpc_jobs.sh --from-processed
ls jobs | wc -l
```

Expected:

```text
62
```

Submit the full dependency chain:

```bash
bash scripts/generate_hpc_jobs.sh --from-processed --submit
```

The generator submits dependencies in this order:

```text
02 -> 03
07_download -> 07_regime
04_{model}
04_{model} -> 05 shards -> 05 collect
04_{model} -> 06 shards -> 06 collect
04_{model} -> 20 formal weighted jobs
03 + 07 + 05 collects + 06 collects + 20 jobs -> 21
```

Monitor:

```bash
squeue --me
tail -f logs/02_motivation_step1_dvol_*.err
```

Training logs usually appear in `.err` because Python logging writes to stderr.
The `.out` files mostly contain module-load output.

---

## Output Checks

After all jobs complete:

```bash
cd /data/home/tew775/liquidity-ml
squeue --me
```

Check formal predictions:

```bash
for MODEL in elastic_net xgboost neural_network; do
    echo "=== ${MODEL} ==="
    ls -lh outputs/formalanalysis/experiment/${MODEL}/standard/predictions.parquet
    for WT in dolvol softmax_rank_lam2 softmax_rank_lam3 tc_10m tc_100m tc_500m tc_1000m; do
        ls -lh outputs/formalanalysis/experiment/${MODEL}/${WT}/predictions.parquet
    done
done
```

Expected: 24 prediction files.

Check motivation outputs:

```bash
for MODEL in elastic_net xgboost neural_network; do
    ls outputs/motivation/step3/${MODEL}/dvol/predictions.parquet
    ls outputs/motivation/step3_restriction/${MODEL}/dvol/global/baseline/restriction_comparison.csv
    ls outputs/motivation/step3_quintile/${MODEL}/dvol/global/baseline/r2_comparison.csv
done
```

Check final analysis:

```bash
ls outputs/formalanalysis/analysis/tables
ls outputs/formalanalysis/analysis/figures
```

Look for real failures:

```bash
grep -iE "traceback|error|failed|killed|oom" logs/*.err | \
  grep -vE "cuda|Could not find cuda|date_parser|FutureWarning|DeprecationWarning" | \
  head -50
```

---

## Download Results

From your Mac:

```bash
cd /Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml
mkdir -p outputs/formalanalysis outputs/motivation

scp -r tew775@login.hpc.qmul.ac.uk:/data/home/tew775/liquidity-ml/outputs/formalanalysis \
  outputs/

scp -r tew775@login.hpc.qmul.ac.uk:/data/home/tew775/liquidity-ml/outputs/motivation \
  outputs/
```

If you already let job `21` run on HPC, you do not need to rerun `21` locally
unless you changed analysis-only code.

---

## Rerunning Parts Safely

Rerun all HPC jobs from uploaded processed data:

```bash
rm -rf jobs logs outputs data/regime_indicators.csv
mkdir -p jobs logs outputs
bash scripts/generate_hpc_jobs.sh --from-processed --submit
```

Rerun formal training for one model after changing model/training code:

```bash
rm -rf outputs/formalanalysis/experiment/xgboost
bash scripts/generate_hpc_jobs.sh --from-processed
sbatch jobs/04_step3_xgboost_dvol.sh
```

Then submit the dependent jobs manually or rerun the full generator with
dependencies after cleaning the affected outputs.

Rerun analysis only:

```bash
sbatch jobs/21_formal_analyze_results.sh
```

This reads existing outputs and does not retrain models.

---

## Troubleshooting

Check a job:

```bash
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
cat logs/<jobname>_<jobid>.err
```

Cancel jobs:

```bash
scancel <jobid>
scancel -u tew775
```

Common issues:

| Symptom | Likely cause | Fix |
|---|---|---|
| `signed_predictors_dl_wide.zip not found` | CZ ZIP not uploaded | Upload to `data/temp/` |
| `SignalDoc.csv not found` | ignored data file missing | Upload to `data/` |
| `processed_panel.parquet not found` | processed file was not uploaded | Run local `00/01`, upload `processed_panel.parquet` |
| FF5+Mom analysis falls back | factor CSVs missing | Upload `ff5_factors.csv` and `momentum_factor.csv` |
| Step 03 exits with OOM | full 113-feature heterogeneity run needs more memory | Use the current generator; Step 03 requests `4 x 24G` |
| Step 05/06 says `tuned_params.csv not found` | Step 04 failed or did not finish | Inspect `04_step3_*` logs |
| `21` misses Prediction 3 | Step 05 collect output missing | Inspect `05_restrict_*_collect` logs |
| `git pull` blocked by outputs | stale untracked outputs | move/delete `outputs/` before pull |

---

## Checklist

Before submit:

- [ ] Current local code committed and pushed to `formalanalysis`
- [ ] Local `00_fetch_data.py` and `01_process_data.py` completed successfully
- [ ] HPC repo freshly cloned or pulled
- [ ] `~/liquidml_env` exists
- [ ] Processed panel, feature list, SignalDoc, and factor CSVs uploaded
- [ ] `bash -n scripts/generate_hpc_jobs.sh` passes
- [ ] `bash scripts/generate_hpc_jobs.sh --from-processed` creates 62 jobs
- [ ] `bash scripts/generate_hpc_jobs.sh --from-processed --submit` submitted the dependency chain

After completion:

- [ ] `squeue --me` is empty
- [ ] 24 formal prediction files exist
- [ ] Motivation Step 3, 3d, 3e outputs exist for all three models
- [ ] `outputs/formalanalysis/analysis/tables/` and `figures/` exist
- [ ] Logs do not contain real tracebacks/OOM/failed jobs

---

## Related Files

- `scripts/generate_hpc_jobs.sh`
- `scripts/20_formal_run_experiment.py`
- `scripts/21_formal_analyze_results.py`
- `docs/run_order_cheat_sheet.md`
- `docs/weighting_schemes.md`
