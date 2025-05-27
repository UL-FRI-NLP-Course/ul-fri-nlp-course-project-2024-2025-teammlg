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

to run the evaluation (you have to be in code directory for relative imports to work).


## Chatbot usage
Run the following command to create an interactive session:
```bash
srun --job-name "chatbot testing" --cpus-per-task 4 --mem-per-cpu 1500 --time 30:00 --gres=gpu:2 --partition=gpu --pty bash
```

Then run the following (you have to specify which model you want to use, options are: deepseek_baseline, deepseek_naive, deepseek_advanced, qwen_baseline, qwen_naive, qwen_advanced):
```bash
singularity exec --nv ../../containers/nlp-v1.sif python ./conversation_evaluation.py --model qwen_naive
```

Shard loading can take up to 30 minutes. The <code>></code> symbol indicates that the system is waiting for your query. Response generation typically takes around 30 seconds. To terminate the current session type <code>quit</code>.

If you get SSL-related errors, run:

```bash
unset SSL_CERT_FILE
```


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