#!/bin/bash
# Generate and optionally submit the full Apocrita SLURM pipeline.
#
# Usage:
#   bash scripts/generate_hpc_jobs.sh
#   bash scripts/generate_hpc_jobs.sh --submit
#   bash scripts/generate_hpc_jobs.sh --from-processed
#   bash scripts/generate_hpc_jobs.sh --from-processed --submit
#
# Login-node rule: this script only creates/submits SLURM jobs. All Python
# commands run inside jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-/data/home/tew775/liquidity-ml}"
VENV="${VENV:-source ~/liquidml_env/bin/activate}"
SUBMIT=false
FROM_PROCESSED=false

for arg in "$@"; do
    case "${arg}" in
        --submit)
            SUBMIT=true
            ;;
        --from-processed|--skip-data)
            FROM_PROCESSED=true
            ;;
        -h|--help)
            cat <<EOF
Usage:
  bash scripts/generate_hpc_jobs.sh [--from-processed] [--submit]

Options:
  --from-processed  Skip jobs 00/01. Use when processed_panel.parquet and
                    feature_list.json were created locally and uploaded to HPC.
  --submit          Submit the generated SLURM jobs with dependencies.
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: ${arg}" >&2
            exit 1
            ;;
    esac
done

MODELS=(elastic_net xgboost neural_network)
SOFTMAX_LAMBDAS=(2 3)
TC_AUMS=(10 100 500 1000)

cd "${REPO_ROOT}"
mkdir -p jobs logs
rm -f jobs/*.sh

write_job() {
    local jobname="$1"
    local ntasks="$2"
    local mem_per_cpu="$3"
    local walltime="$4"
    local command="$5"

    cat > "jobs/${jobname}.sh" << JOBEOF
#!/bin/bash
#SBATCH -J ${jobname}
#SBATCH -n ${ntasks}
#SBATCH --mem-per-cpu=${mem_per_cpu}
#SBATCH -t ${walltime}
#SBATCH -o logs/${jobname}_%j.out
#SBATCH -e logs/${jobname}_%j.err

set -euo pipefail
module load python
${VENV}
export OMP_NUM_THREADS=\${SLURM_NTASKS:-${ntasks}}
export PYTHONHASHSEED=0
cd ${PROJECT_DIR}

${command}
JOBEOF
    echo "  Created jobs/${jobname}.sh"
}

submit_job() {
    local jobname="$1"
    local dependency="${2:-}"
    local jid
    if [ -n "${dependency}" ]; then
        jid=$(sbatch --parsable --dependency=afterok:${dependency} "jobs/${jobname}.sh")
    else
        jid=$(sbatch --parsable "jobs/${jobname}.sh")
    fi
    echo "${jid}"
}

join_by_colon() {
    local IFS=":"
    echo "$*"
}

if [ "${FROM_PROCESSED}" = true ]; then
    echo "=== Generating HPC pipeline jobs from processed data in jobs/ ==="
else
    echo "=== Generating full pipeline jobs in jobs/ ==="
fi

# Data + motivation prerequisites.
if [ "${FROM_PROCESSED}" = false ]; then
    write_job "00_fetch_data" 2 "12G" "240:0:0" \
        "python scripts/00_fetch_data.py"
    write_job "01_process_data" 4 "16G" "240:0:0" \
        "python scripts/01_process_data.py"
fi
write_job "02_motivation_step1_dvol" 2 "12G" "120:0:0" \
    "python scripts/02_motivation_step1_divergence.py --liquidity dvol --vw"
# Step 2 --full runs both the focal 15-feature and full 113-feature
# interaction regressions. The full run is memory-heavy, so give it a wider
# allocation than Step 1.
write_job "03_motivation_step2_dvol_full" 4 "24G" "120:0:0" \
    "python scripts/03_motivation_step2_heterogeneity.py --liquidity dvol --full"

# Optional regime extension: xgboost namespace only because the plots are
# model-independent and the model argument only namespaces outputs.
write_job "07_regime_download" 1 "4G" "12:0:0" \
    "python scripts/07_motivation_regime_analysis.py --download-regime-data"
write_job "07_regime_xgboost_dvol" 1 "8G" "24:0:0" \
    "python scripts/07_motivation_regime_analysis.py --model xgboost --liquidity dvol"

for MODEL in "${MODELS[@]}"; do
    # Step 3 baseline diagnostics. This also creates the formal M_std cache.
    write_job "04_step3_${MODEL}_dvol" 4 "12G" "240:0:0" \
        "python scripts/04_motivation_step3_ml_diagnostics.py --model ${MODEL} --liquidity dvol"

    # Step 3d restriction models split by minimum quintile, then one collect job.
    for MQ in 2 3 4 5; do
        write_job "05_restrict_${MODEL}_mq${MQ}" 4 "12G" "240:0:0" \
            "python scripts/05_motivation_step3d_progressive_restriction.py --model ${MODEL} --liquidity dvol --normalization global --use-baseline-params --min-quintile ${MQ}"
    done
    write_job "05_restrict_${MODEL}_collect" 1 "8G" "24:0:0" \
        "python scripts/05_motivation_step3d_progressive_restriction.py --model ${MODEL} --liquidity dvol --normalization global --use-baseline-params --use-cache"

    # Step 3e quintile-specific models split by quintile, then one collect job.
    for Q in 1 2 3 4 5; do
        write_job "06_quintile_${MODEL}_q${Q}" 4 "12G" "240:0:0" \
            "python scripts/06_motivation_step3e_quintile_specific_models.py --model ${MODEL} --liquidity dvol --normalization global --use-baseline-params --quintile ${Q}"
    done
    write_job "06_quintile_${MODEL}_collect" 1 "8G" "24:0:0" \
        "python scripts/06_motivation_step3e_quintile_specific_models.py --model ${MODEL} --liquidity dvol --normalization global --use-baseline-params --use-cache"

    # Formal weighted-training specs. M_std is pre-populated by Step 04.
    write_job "20_${MODEL}_dolvol" 4 "12G" "240:0:0" \
        "python scripts/20_formal_run_experiment.py --model ${MODEL} --weights dolvol"
    for LAM in "${SOFTMAX_LAMBDAS[@]}"; do
        LAM_LABEL=${LAM//./p}
        write_job "20_${MODEL}_softmax_rank_lam${LAM_LABEL}" 4 "12G" "240:0:0" \
            "python scripts/20_formal_run_experiment.py --model ${MODEL} --weights softmax_rank --softmax-lambda ${LAM}"
    done
    for AUM in "${TC_AUMS[@]}"; do
        write_job "20_${MODEL}_tc_${AUM}m" 4 "12G" "240:0:0" \
            "python scripts/20_formal_run_experiment.py --model ${MODEL} --weights tc --aum ${AUM}"
    done
done

write_job "21_formal_analyze_results" 2 "16G" "120:0:0" \
    "python scripts/21_formal_analyze_results.py"

echo ""
echo "=== Job generation complete ==="

if [ "${SUBMIT}" = true ]; then
    echo ""
    echo "=== Submitting dependency chain ==="

    data_ready_dep=""
    if [ "${FROM_PROCESSED}" = false ]; then
        id00=$(submit_job "00_fetch_data")
        echo "00_fetch_data -> ${id00}"
        id01=$(submit_job "01_process_data" "${id00}")
        echo "01_process_data -> ${id01}"
        data_ready_dep="${id01}"
    else
        echo "Skipping 00/01; assuming processed panel and feature list are already uploaded."
    fi

    id02=$(submit_job "02_motivation_step1_dvol" "${data_ready_dep}")
    echo "02_motivation_step1_dvol -> ${id02}"
    id03=$(submit_job "03_motivation_step2_dvol_full" "${id02}")
    echo "03_motivation_step2_dvol_full -> ${id03}"
    id07dl=$(submit_job "07_regime_download" "${data_ready_dep}")
    echo "07_regime_download -> ${id07dl}"
    if [ -n "${data_ready_dep}" ]; then
        id07_dep=$(join_by_colon "${data_ready_dep}" "${id07dl}")
    else
        id07_dep="${id07dl}"
    fi
    id07=$(submit_job "07_regime_xgboost_dvol" "${id07_dep}")
    echo "07_regime_xgboost_dvol -> ${id07}"

    final_deps=("${id03}" "${id07}")

    for MODEL in "${MODELS[@]}"; do
        id04=$(submit_job "04_step3_${MODEL}_dvol" "${data_ready_dep}")
        echo "04_step3_${MODEL}_dvol -> ${id04}"

        restrict_ids=()
        for MQ in 2 3 4 5; do
            jid=$(submit_job "05_restrict_${MODEL}_mq${MQ}" "${id04}")
            restrict_ids+=("${jid}")
            echo "05_restrict_${MODEL}_mq${MQ} -> ${jid}"
        done
        id05=$(submit_job "05_restrict_${MODEL}_collect" "$(join_by_colon "${restrict_ids[@]}")")
        echo "05_restrict_${MODEL}_collect -> ${id05}"
        final_deps+=("${id05}")

        quintile_ids=()
        for Q in 1 2 3 4 5; do
            jid=$(submit_job "06_quintile_${MODEL}_q${Q}" "${id04}")
            quintile_ids+=("${jid}")
            echo "06_quintile_${MODEL}_q${Q} -> ${jid}"
        done
        id06=$(submit_job "06_quintile_${MODEL}_collect" "$(join_by_colon "${quintile_ids[@]}")")
        echo "06_quintile_${MODEL}_collect -> ${id06}"
        final_deps+=("${id06}")

        formal_ids=()
        jid=$(submit_job "20_${MODEL}_dolvol" "${id04}")
        formal_ids+=("${jid}")
        echo "20_${MODEL}_dolvol -> ${jid}"
        for LAM in "${SOFTMAX_LAMBDAS[@]}"; do
            LAM_LABEL=${LAM//./p}
            jid=$(submit_job "20_${MODEL}_softmax_rank_lam${LAM_LABEL}" "${id04}")
            formal_ids+=("${jid}")
            echo "20_${MODEL}_softmax_rank_lam${LAM_LABEL} -> ${jid}"
        done
        for AUM in "${TC_AUMS[@]}"; do
            jid=$(submit_job "20_${MODEL}_tc_${AUM}m" "${id04}")
            formal_ids+=("${jid}")
            echo "20_${MODEL}_tc_${AUM}m -> ${jid}"
        done
        final_deps+=("${formal_ids[@]}")
    done

    id21=$(submit_job "21_formal_analyze_results" "$(join_by_colon "${final_deps[@]}")")
    echo "21_formal_analyze_results -> ${id21}"

    echo ""
    echo "=== Submitted full pipeline ==="
    echo "Monitor with: squeue --me"
else
    echo ""
    echo "To submit the full dependency chain:"
    if [ "${FROM_PROCESSED}" = true ]; then
        echo "  bash scripts/generate_hpc_jobs.sh --from-processed --submit"
    else
        echo "  bash scripts/generate_hpc_jobs.sh --submit"
    fi
    echo ""
    echo "To submit manually, start with:"
    if [ "${FROM_PROCESSED}" = true ]; then
        echo "  sbatch jobs/02_motivation_step1_dvol.sh"
    else
        echo "  sbatch jobs/00_fetch_data.sh"
    fi
fi
