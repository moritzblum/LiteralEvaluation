#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-7:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c10
#SBATCH -o /homes/mblum/outputs/stdout_LiteralEvaluation_base_%A_%a.txt
#SBATCH --array=0-27%8

source /homes/mblum/.bashrc
source /homes/mblum/miniconda3/etc/profile.d/conda.sh
conda activate tucker
cd /homes/mblum/KGA

job_count=0
for run_id in "1" "2" "3"; do
  for dataset in "LitWD48K" "YAGO3-10" "FB15k-237"; do

    for model in "distmult_literal" "complex_literal" "tucker_literal"; do    # "tucker" "distmult" "complex"
      if [ "${job_count}" -ne "${SLURM_ARRAY_TASK_ID}" ]; then
        job_count=$((job_count+1))
        continue
      fi
      python -u run.py --dataset "${dataset}" --model "${model}" --output "out_${run_id}"
      exit 0
    done
  done
done
