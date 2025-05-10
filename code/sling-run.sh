#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --output=logs/sling-nlp-showcase-%J.out
#SBATCH --error=logs/sling-nlp-showcase-%J.err
#SBATCH --job-name="SLING NLP showcase"
