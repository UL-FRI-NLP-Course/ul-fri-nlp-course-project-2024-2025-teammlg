cd ~
mkdir containers
singularity build ./containers/nlp-v1.sif docker://pytorch/pytorch
singularity exec ./containers/nlp-v1.sif pip install pytextrank numpy scikit-learn transformers==4.51 spacy rouge_score requests beautifulsoup4 evaluate accelerate
mkdir logs
sbatch ~/ul-fri-nlp-course-project-2024-2025-teammlg/run-slurm.sh