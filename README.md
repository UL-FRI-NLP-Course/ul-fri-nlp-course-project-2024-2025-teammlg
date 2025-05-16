# Natural language processing course: Conversational Agent with Retrieval-Augmented Generation

Develop a conversational agent that enhances the quality and accuracy of its responses by dynamically retrieving and integrating relevant external documents from the web. Unlike traditional chatbots that rely solely on pre-trained knowledge, this system will perform real-time information retrieval, ensuring up-to-date answers. Potential applications include customer support, academic research assistance, and general knowledge queries. The project will involve natural language processing (NLP), web scraping, and retrieval-augmented generation (RAG) techniques to optimize answer quality.


## Set-up
On HPC clone the repository and then run the following to create the container:

```bash
./ul-fri-nlp-course-project-2024-2025-teammlg/code/sling_setup.sh
```

Then navigate into <code>code</code> directory and run:

```bash
sbatch ./sling-run.sh
```

to run the evaluation.


## Functionality
Brief description of main functionality of each script:
- <code>evaluation.py</code>: simultaneously evaluate multiple models on a list of questions, either <code>data/evaluation_questions.txt</code> (no ground truth) or <code>data/evaluation_questions.json</code> (ground truth).
- <code>converstaion_evaluation.py</code>: evaluate one model at a time, by having a real-time conversation with it.
- <code>scraper.py</code>: obtain structured data from websites TMDB, Letterboxd, JustWatch.
- <code>POStagger.py</code>: finds titles and names of interest in user's prompt
- <code>rag.py</code>: inserts scraped data (based on POS tagger) into the model's context.
- <code>summarizer.py</code>: summarizes scraped data for advanced RAG.
- <code>memory.py</code>: buffer for recent queries and replies for conversational models.
- <code>metrics.py</code>: calculates evaluation metrics based on model's output and ground truth.