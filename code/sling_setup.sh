cd ~
mkdir containers
singularity build ./containers/nlp-v1.sif docker://pytorch/pytorch
singularity exec ./containers/nlp-v1.sif pip install transformers==4.51
singularity exec ./containers/nlp-v1.sif pip install scipy
singularity exec ./containers/nlp-v1.sif pip install spacy[cuda12x]
singularity exec ./containers/nlp-v1.sif pip install fsspec==2025.3.0
singularity exec ./containers/nlp-v1.sif pip install scikit-learn
singularity exec ./containers/nlp-v1.sif pip install rouge_score
singularity exec ./containers/nlp-v1.sif pip install beautifulsoup4
singularity exec ./containers/nlp-v1.sif pip install wikipedia
mkdir logs
sbatch ~/ul-fri-nlp-course-project-2024-2025-teammlg/sling-run.sh