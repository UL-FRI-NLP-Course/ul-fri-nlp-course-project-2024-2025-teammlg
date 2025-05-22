#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --output=logs/sling-nlp-showcase-%J.out
#SBATCH --error=logs/sling-nlp-showcase-%J.err
#SBATCH --job-name="SLING NLP showcase"

srun singularity exec --writable-tmpfs --nv ../../containers/nlp-v1.sif python ./evaluation.py 
