cd ~
mkdir containers
singularity build ./containers/nlp-v1.sif docker://pytorch/pytorch
singularity exec ./containers/nlp-v1.sif pip install GitPython
singularity exec ./containers/nlp-v1.sif pip install transformers==4.51
singularity exec ./containers/nlp-v1.sif pip install scipy
singularity exec ./containers/nlp-v1.sif pip install pytextrank
singularity exec ./containers/nlp-v1.sif pip install accelerate
singularity exec ./containers/nlp-v1.sif pip install deepeval
singularity exec ./containers/nlp-v1.sif pip install evaluate
singularity exec ./containers/nlp-v1.sif pip install ollama
singularity exec ./containers/nlp-v1.sif python -m spacy download en_core_web_sm
singularity exec ./containers/nlp-v1.sif python -m spacy download en_core_web_trf
singularity exec ./containers/nlp-v1.sif pip install nltk==3.9.1
singularity exec ./containers/nlp-v1.sif pip install spacy[cuda12x]
singularity exec ./containers/nlp-v1.sif pip install fsspec==2025.3.0
singularity exec ./containers/nlp-v1.sif pip install scikit-learn
singularity exec ./containers/nlp-v1.sif pip install rouge_score
singularity exec ./containers/nlp-v1.sif pip install absl-py
singularity exec ./containers/nlp-v1.sif pip install beautifulsoup4
singularity exec ./containers/nlp-v1.sif pip install wikipedia
mkdir logs
sbatch ~/ul-fri-nlp-course-project-2024-2025-teammlg/code/sling-run.sh