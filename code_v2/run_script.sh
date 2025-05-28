#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --output=logs/test-%J.out
#SBATCH --error=logs/test-%J.err
#SBATCH --job-name="Evaluation LLM"

srun singularity exec --writable-tmpfs --nv ~/containers/nlp-v1.sif python main.py --rag_type $1 --model_type $2 --operation $3 --output_directory $4 --uses_memory
