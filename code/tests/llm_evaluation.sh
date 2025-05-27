#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --output=logs/test-%J.out
#SBATCH --error=logs/test-%J.err
#SBATCH --job-name="Evaluation LLM - Custom"

#srun singularity exec --writable-tmpfs --nv ~/containers/nlp-v1.sif python ./llm_evaluation.py $1 $2 1 5
srun singularity exec --writable-tmpfs --nv ~/containers/nlp-v1.sif python ./llm_evaluation.py $1 $2 6 10
#srun singularity exec --writable-tmpfs --nv ~/containers/nlp-v1.sif python ./llm_evaluation.py $1 $2 11 15
srun singularity exec --writable-tmpfs --nv ~/containers/nlp-v1.sif python ./llm_evaluation.py $1 $2 16 20
#srun singularity exec --writable-tmpfs --nv ~/containers/nlp-v1.sif python ./llm_evaluation.py $1 $2 21 25
srun singularity exec --writable-tmpfs --nv ~/containers/nlp-v1.sif python ./llm_evaluation.py $1 $2 26 30
#srun singularity exec --writable-tmpfs --nv ~/containers/nlp-v1.sif python ./llm_evaluation.py $1 $2 31 35
srun singularity exec --writable-tmpfs --nv ~/containers/nlp-v1.sif python ./llm_evaluation.py $1 $2 36 40
#srun singularity exec --writable-tmpfs --nv ~/containers/nlp-v1.sif python ./llm_evaluation.py $1 $2 41 45
srun singularity exec --writable-tmpfs --nv ~/containers/nlp-v1.sif python ./llm_evaluation.py $1 $2 46 50

