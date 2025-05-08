# Natural language processing course: Conversational Agent with Retrieval-Augmented Generation

Develop a conversational agent that enhances the quality and accuracy of its responses by dynamically retrieving and integrating relevant external documents from the web. Unlike traditional chatbots that rely solely on pre-trained knowledge, this system will perform real-time information retrieval, ensuring up-to-date answers. Potential applications include customer support, academic research assistance, and general knowledge queries. The project will involve natural language processing (NLP), web scraping, and retrieval-augmented generation (RAG) techniques to optimize answer quality.


## Set-up
For setting up the environment, [Conda](https://docs.conda.io/projects/conda/en/latest/index.html) is used. In the `code` directory, `conda_environment.yaml` file defines the environment parameters. Execute (warning: this will also install some models, so it might take some time and requires ~800MB of disk space):

```bash
conda env create -f conda_environment.yml
```

and then:

```bash
conda activate teammlg-project
```

to activate the environment.

Also download and install [Ollama](https://ollama.com/download).

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