#!/usr/bin/env bash
# Multi-GPU material-passport scene processing job.

#SBATCH --job-name=ase_material_lang
#SBATCH --output=logs/ase_material_lang_%j.out
#SBATCH --error=logs/ase_material_lang_%j.err
#SBATCH --time=6-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40s:6

# Submit from the repository root (logs/ must exist: mkdir -p logs).
# Paths and GPU count can be overridden at submit time, e.g.:
#   sbatch --gres=gpu:a40:n \
#     --export=ALL,DATA_ROOT=/path/to/ase/scenes,OUTPUT_DIR=/path/to/output,N_GPUS=n,N_WORKERS=n \
#     sbatch_material_ase_lang.sh

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
DATA_ROOT=${DATA_ROOT:-${PROJECT_DIR}/sample_data}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/material_ase_lang_output}
CONDA_ENV=${CONDA_ENV:-material_passport}
N_GPUS=${N_GPUS:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}}
if ! [[ "${N_GPUS}" =~ ^[0-9]+$ ]]; then
  N_GPUS=1
fi
# Change worker count to 2 per GPU (oversubscribe) to hide CPU preparation latency
N_WORKERS=${N_WORKERS:-$(( N_GPUS * 2 ))}
VLM_MODEL=${VLM_MODEL:-google/gemma-4-E4B-it}
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-64}

if ! [[ "${N_WORKERS}" =~ ^[0-9]+$ ]]; then
  N_WORKERS=$(( N_GPUS * 2 ))
fi

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MP_VLM_PROVIDER=hf
export MP_VLM_MODEL="${VLM_MODEL}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-$$}"
export OMP_NUM_THREADS=$(( CPUS_PER_TASK / N_WORKERS > 0 ? CPUS_PER_TASK / N_WORKERS : 1 ))
mkdir -p "${MPLCONFIGDIR}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Project: ${PROJECT_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Using ${N_GPUS} GPU(s), ${N_WORKERS} worker process(es), VLM=${MP_VLM_MODEL}"
nvidia-smi || true

python -m material_passport.cli \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --material-methods dms vlm cv \
  --condition-methods vlm rule \
  --material-methods-consensus priority \
  --condition-methods-consensus priority \
  --skip-existing \
  --num-workers "${N_WORKERS}" \
  --num-gpus "${N_GPUS}" \
  --vlm-provider hf \
  --vlm-model "${MP_VLM_MODEL}"
