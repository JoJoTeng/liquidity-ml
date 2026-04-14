#!/bin/bash
# Generate and optionally submit all 12 training jobs for the formal experiment.
#
# Usage:
#   bash scripts/generate_hpc_jobs.sh          # generate job scripts only
#   bash scripts/generate_hpc_jobs.sh --submit  # generate and submit
#
# Each job trains one model x one weight specification.
# All 12 jobs are independent - no dependency chain needed.
# M_std (standard training) is shared: the first job per model trains it,
# subsequent jobs reuse the saved predictions.

set -e

PROJECT_DIR="/data/home/tew775/liquidity-ml"
VENV="source ~/liquidml_env/bin/activate"
SUBMIT=${1:-""}

mkdir -p jobs logs

MODELS="elastic_net xgboost neural_network"
TC_AUMS="100 500 1000"

echo "=== Generating 12 job scripts ==="

for MODEL in $MODELS; do
    # -- DolVol job (AUM-independent) --
    JOBNAME="${MODEL}_dolvol"
    cat > jobs/${JOBNAME}.sh << JOBEOF
#!/bin/bash
#SBATCH -J ${JOBNAME}
#SBATCH -n 4
#SBATCH --mem-per-cpu=8G
#SBATCH -t 240:0:0
#SBATCH -o logs/${JOBNAME}_%j.out
#SBATCH -e logs/${JOBNAME}_%j.err
module load python
${VENV}
export OMP_NUM_THREADS=\${SLURM_NTASKS}
export PYTHONHASHSEED=0
cd ${PROJECT_DIR}
python scripts/02_run_experiment.py --model ${MODEL} --weights dolvol
JOBEOF
    echo "  Created jobs/${JOBNAME}.sh"

    # -- TC jobs (one per AUM level) --
    for AUM in $TC_AUMS; do
        JOBNAME="${MODEL}_tc_${AUM}m"
        cat > jobs/${JOBNAME}.sh << JOBEOF
#!/bin/bash
#SBATCH -J ${JOBNAME}
#SBATCH -n 4
#SBATCH --mem-per-cpu=8G
#SBATCH -t 240:0:0
#SBATCH -o logs/${JOBNAME}_%j.out
#SBATCH -e logs/${JOBNAME}_%j.err
module load python
${VENV}
export OMP_NUM_THREADS=\${SLURM_NTASKS}
export PYTHONHASHSEED=0
cd ${PROJECT_DIR}
python scripts/02_run_experiment.py --model ${MODEL} --weights tc --aum ${AUM}
JOBEOF
        echo "  Created jobs/${JOBNAME}.sh"
    done
done

echo ""
echo "=== 12 job scripts generated in jobs/ ==="

if [ "$SUBMIT" = "--submit" ]; then
    echo ""
    echo "=== Submitting all 12 jobs ==="
    for MODEL in $MODELS; do
        sbatch jobs/${MODEL}_dolvol.sh
        for AUM in $TC_AUMS; do
            sbatch jobs/${MODEL}_tc_${AUM}m.sh
        done
    done
    echo ""
    echo "=== All 12 jobs submitted ==="
    echo "Monitor with: squeue --me"
else
    echo ""
    echo "To submit: bash scripts/generate_hpc_jobs.sh --submit"
    echo "Or manually: sbatch jobs/xgboost_dolvol.sh"
fi
