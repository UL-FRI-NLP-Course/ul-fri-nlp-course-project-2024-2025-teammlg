#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --output=logs/test-%J.out
#SBATCH --error=logs/test-%J.err
#SBATCH --job-name="Evaluation LLM - Custom"

srun singularity exec --writable-tmpfs --nv ~/containers/nlp-v1.sif python ./llm_evaluation.py $1 $2 $3 $4
