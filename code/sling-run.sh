#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --output=logs/test-%J.out
#SBATCH --error=logs/test-%J.err
#SBATCH --job-name="gg6898 test"

srun singularity exec --nv ./containers/nlp.sif python \
    ul-fri-nlp-course-project-2024-2025-teammlg/code/tests/transformers_rag.py

