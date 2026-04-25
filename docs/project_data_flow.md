# LiquidityML Data Flow

This document is a visual map of how data moves through the repository.

Use it together with [project_code_walkthrough.md](/Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/docs/project_code_walkthrough.md):

- this file answers: "where does data go next?"
- the walkthrough answers: "which code should I read, and in what order?"
- the motivation-only function map is in [motivation_module_flow.md](/Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/docs/motivation_module_flow.md)
- the exact run commands are in [run_order_cheat_sheet.md](/Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/docs/run_order_cheat_sheet.md)

## 1. End-to-End Flow

```mermaid
flowchart TD
    cfg["config/config.yaml"]
    cz["data/temp/signed_predictors_dl_wide.zip"]
    ff["data/FFResearch_Data_Factors.csv"]
    sig["data/SignalDoc.csv"]
    wrds["WRDS / CRSP monthly + daily"]

    fetch["scripts/00_fetch_data.py"]
    raw["data/signed_predictors_all_wide.csv"]
    loader["src/data/loader.py::load_panel()"]
    process["scripts/01_process_data.py"]
    processed["data/processed_panel.parquet"]
    featlist["data/feature_list.json"]
    catmap["config/feature_categories.json"]

    step1["scripts/05_step1_divergence.py"]
    step2["scripts/06_step2_heterogeneity.py"]
    step3["scripts/07_step3_ml_diagnostics.py"]
    step3en["scripts/08_step3_elasticnet.py"]
    restrict["scripts/12_progressive_restriction.py"]
    quint["scripts/10_quintile_specific_models.py"]
    regime["scripts/11_regime_analysis.py"]

    mot1["outputs/motivation/step1/..."]
    mot2["outputs/motivation/step2/..."]
    mot3["outputs/motivation/step3/..."]
    motext["outputs/motivation/step3_restriction...<br/>outputs/motivation/step3_quintile...<br/>outputs/motivation/step1_regime..."]

    weights["src/weighting/schemes.py"]
    models["src/models/*"]
    portfolio["src/portfolio/construction.py"]
    stats["src/evaluation/statistics.py"]
    exp["scripts/02_run_experiment.py"]
    expout["outputs/formalanalysis/experiment/..."]
    analyze["scripts/03_analyze_results.py"]
    formalout["outputs/formalanalysis/analysis/..."]

    paper["paper/TablesNew/ + paper/FiguresNew/"]

    cz --> fetch
    ff --> fetch
    wrds --> fetch
    fetch --> raw

    cfg --> loader
    raw --> loader
    ff --> loader
    loader --> process
    sig --> process
    process --> processed
    process --> featlist
    process --> catmap

    processed --> step1
    featlist --> step1
    catmap --> step1
    cfg --> step1
    step1 --> mot1

    processed --> step2
    featlist --> step2
    mot1 --> step2
    cfg --> step2
    step2 --> mot2

    raw --> loader
    featlist --> step3
    cfg --> step3
    loader --> step3
    step3 --> mot3

    featlist --> step3en
    cfg --> step3en
    loader --> step3en
    step3en --> mot3

    processed --> restrict
    featlist --> restrict
    mot3 --> restrict
    cfg --> restrict
    restrict --> motext

    processed --> quint
    featlist --> quint
    mot3 --> quint
    cfg --> quint
    quint --> motext

    processed --> regime
    cfg --> regime
    regime --> motext

    loader --> exp
    featlist --> exp
    cfg --> exp
    weights --> exp
    models --> exp
    exp --> expout

    expout --> analyze
    cfg --> analyze
    loader --> analyze
    portfolio --> analyze
    stats --> analyze
    mot2 --> analyze
    motext --> analyze
    analyze --> formalout

    mot1 --> paper
    mot2 --> paper
    mot3 --> paper
    motext --> paper
    formalout --> paper
```

## 2. Motivation Pipeline Flow

```mermaid
flowchart TD
    proc["scripts/01_process_data.py"]
    processed["data/processed_panel.parquet"]
    featlist["data/feature_list.json"]
    catmap["config/feature_categories.json"]
    cfg["config/config.yaml"]
    loader["src/data/loader.py::load_panel()"]

    step1["Step 1<br/>scripts/05_step1_divergence.py"]
    step2["Step 2<br/>scripts/06_step2_heterogeneity.py"]
    step3["Step 3 baseline<br/>scripts/07_step3_ml_diagnostics.py"]
    step3en["Step 3 ElasticNet<br/>scripts/08_step3_elasticnet.py"]
    restrict["Step 3d restriction curve<br/>scripts/12_progressive_restriction.py"]
    quint["Step 3e quintile models<br/>scripts/10_quintile_specific_models.py"]
    regime["Regime extension<br/>scripts/11_regime_analysis.py"]

    out1["outputs/motivation/step1/{liq}/"]
    out2["outputs/motivation/step2/{liq}/"]
    out3["outputs/motivation/step3/{liq}/"]
    out3en["outputs/motivation/step3_elasticnet/{liq}/"]
    outrest["outputs/motivation/step3_restriction_rerank/{liq}/{mode}/"]
    outquint["outputs/motivation/step3_quintile_rerank/{liq}/{mode}/"]
    outreg["outputs/motivation/step1_regime/"]

    proc --> processed
    proc --> featlist
    proc --> catmap

    processed --> step1
    featlist --> step1
    catmap --> step1
    cfg --> step1
    step1 --> out1

    processed --> step2
    featlist --> step2
    out1 --> step2
    cfg --> step2
    step2 --> out2

    loader --> step3
    featlist --> step3
    cfg --> step3
    step3 --> out3

    loader --> step3en
    featlist --> step3en
    cfg --> step3en
    step3en --> out3en

    processed --> restrict
    featlist --> restrict
    out3 --> restrict
    cfg --> restrict
    restrict --> outrest

    processed --> quint
    featlist --> quint
    out3 --> quint
    cfg --> quint
    quint --> outquint

    processed --> regime
    cfg --> regime
    regime --> outreg
```

## 3. Formal Experiment Flow

```mermaid
flowchart TD
    cfg["config/config.yaml"]
    raw["data/signed_predictors_all_wide.csv"]
    featlist["data/feature_list.json"]

    loader["src/data/loader.py::load_panel()"]
    panel["In-memory raw panel<br/>with forward excess returns + liq_* columns"]

    weights["src/weighting/schemes.py"]
    dolvol["dolvol weights<br/>DolVol / month mean DolVol"]
    softmax["softmax_rank weights<br/>exp(lambda * DolVol percentile rank)<br/>lambda=2 and lambda=3"]
    tcweights["tc weights<br/>AUM-dependent TC penalty"]
    models["src/models/base.py + registry + model classes"]

    exp["scripts/02_run_experiment.py"]
    std["M_std artifacts<br/>standard/"]
    wt["M_w artifacts<br/>dolvol/ or tc_{aum}m/"]
    expout["outputs/formalanalysis/experiment/{model}/..."]

    portfolio["src/portfolio/construction.py"]
    stats["src/evaluation/statistics.py"]
    analyze["scripts/03_analyze_results.py"]
    analysis["outputs/formalanalysis/analysis/"]

    cfg --> loader
    raw --> loader
    loader --> panel
    featlist --> exp
    panel --> exp
    cfg --> exp
    weights --> exp
    weights --> dolvol
    weights --> softmax
    weights --> tcweights
    dolvol --> exp
    softmax --> exp
    tcweights --> exp
    models --> exp

    exp --> std
    exp --> wt
    std --> expout
    wt --> expout

    expout --> analyze
    panel --> analyze
    cfg --> analyze
    portfolio --> analyze
    stats --> analyze
    analyze --> analysis
```

Weighting note:
- `dolvol` is computed as `DolVol_it / mean_i(DolVol_it)` within each `yyyymm`.
- The mean denominator keeps the average sample weight equal to 1 inside every month.
- `softmax_rank` is computed from the within-month percentile rank of `DolVol`, then mean-normalized. The formal grid now runs lambda 2 and lambda 3 as separate output folders.
- `tc` weights are separate from `dolvol`; they use transaction-cost inputs and require an AUM scenario.
- Current `tc` weights use `alpha_t = 3.0 / median_i(TC_it)`, then normalize weights to mean 1 within month.

## 4. What Each Stage Produces

### Stage A. Raw data assembly

- `scripts/00_fetch_data.py`
- Produces:
  - `data/signed_predictors_all_wide.csv`

This is the raw master panel.

### Stage B. Processed motivation panel

- `scripts/01_process_data.py`
- Produces:
  - `data/processed_panel.parquet`
  - `data/feature_list.json`
  - `config/feature_categories.json`

This is the main entry point for motivation Steps 1 and 2.

### Stage C. Motivation evidence

- `scripts/05_step1_divergence.py`
  - distribution mismatch outputs

- `scripts/06_step2_heterogeneity.py`
  - heterogeneous-predictability outputs
  - with `--full`, also writes the Step 2 files used later by formal Prediction 2

- `scripts/07_step3_ml_diagnostics.py`
  - baseline XGBoost diagnostics

- `scripts/08_step3_elasticnet.py`
  - linear benchmark version of Step 3

- `scripts/12_progressive_restriction.py`
  - restriction-curve comparison
  - its baseline output is reused later by formal Prediction 3

- `scripts/10_quintile_specific_models.py`
  - quintile-specific model comparison

- `scripts/11_regime_analysis.py`
  - regime-specific distribution plots

### Stage D. Formal experiment

- `scripts/02_run_experiment.py`
  - trains standard and weighted models
  - writes predictions and importance files

- `scripts/03_analyze_results.py`
  - consumes those experiment files
  - also consumes Step 2 full outputs and Step 3d baseline restriction outputs
  - writes formal tables and figures

## 5. How To Follow One Observation Through the System

Think of one stock-month observation as moving through these transformations:

1. It is created in the raw merged panel by `scripts/00_fetch_data.py`.
2. `src/data/loader.py` loads it and shifts its return forward so month `t` characteristics predict month `t+1` return.
3. In the motivation branch:
   - `scripts/01_process_data.py` rank-transforms its features into the processed panel.
   - Step 1 compares how much this kind of observation matters under equal-weighted versus implementability-weighted distributions.
   - Step 2 studies whether its feature-return relationship differs by liquidity group.
   - Step 3 tests how well baseline ML predicts it.
4. In the formal branch:
   - `scripts/02_run_experiment.py` may assign it a training weight.
   - one of the model classes uses it in rolling estimation
   - the resulting prediction is written to experiment output
   - `scripts/03_analyze_results.py` may then use that prediction in quintile R2 analysis, Prediction 3 restriction-curve comparison, and portfolio formation

## 6. Best Way To Use This Diagram

If you are reading the project for the first time, I recommend this sequence:

1. Open [project_code_walkthrough.md](/Users/tengjiao/Desktop/PhD-Y3/Liquidity/liquidity_ml/docs/project_code_walkthrough.md).
2. Keep this flowchart open beside it.
3. As you read each script, check:
   - what files it reads
   - what files it writes
   - whether it belongs to the motivation branch or the formal branch

If you do that, the codebase becomes much easier to navigate because each file stops feeling isolated.
