#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --output=logs/test-%J.out
#SBATCH --error=logs/test-%J.err
#SBATCH --job-name="Evaluation LLM - Custom"

srun singularity exec --writable-tmpfs --nv ~/containers/nlp-v1.sif python ./llm_evaluation.py qwen_baseline Qwen/Qwen3-14B 1 5
srun singularity exec --writable-tmpfs --nv ~/containers/nlp-v1.sif python ./llm_evaluation.py qwen_baseline Qwen/Qwen3-14B 6 10
srun singularity exec --writable-tmpfs --nv ~/containers/nlp-v1.sif python ./llm_evaluation.py qwen_baseline Qwen/Qwen3-14B 11 15
srun singularity exec --writable-tmpfs --nv ~/containers/nlp-v1.sif python ./llm_evaluation.py qwen_baseline Qwen/Qwen3-14B 16 20
srun singularity exec --writable-tmpfs --nv ~/containers/nlp-v1.sif python ./llm_evaluation.py qwen_baseline Qwen/Qwen3-14B 30 35
