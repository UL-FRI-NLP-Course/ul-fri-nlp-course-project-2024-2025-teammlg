cd ~
mkdir containers
singularity build ./containers/nlp-v1.sif docker://pytorch/pytorch
singularity exec ./containers/nlp-v1.sif pip install transformers==4.51
singularity exec ./containers/nlp-v1.sif pip install scipy
singularity exec ./containers/nlp-v1.sif pip install spacy[cuda12x]
singularity exec ./containers/nlp-v1.sif pip install fsspec==2025.3.0
mkdir logs
sbatch ~/ul-fri-nlp-course-project-2024-2025-teammlg/run-slurm.sh