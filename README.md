# Natural language processing course: Conversational Agent with Retrieval-Augmented Generation

In this report, we developed a set of RAG solutions and performed a thorough evaluation. We used two LLMs, DeepSeek and Qwen3, to compare the behavior of RAG with and without reasoning, respectively. We built two methods for information retrieval and injection, one based on prompt analysis and nondiscriminative prompt injection, and one based on function calling at the LLM's discretion. To perform more robust testing, we evaluated all variations of our RAG systems on a prepared question set, using two different-sized LLMs-as-judge -- GPT-4.1.-mini and 14-billion-parameter Qwen model -- and our own subjective observations. We found that both judges performed similarly on some metrics, with GPT-4.1.-mini performing better on others. The results show that Qwen with function-calling RAG system outperforms all other variants.
 


## Set-up
This repository can be cloned onto the HPC, or it can also be found on the shared folder at `/d/hpc/projects/onj_fri/teammlg/`.

Once you have the repository, run the following to create the container:

```bash
./ul-fri-nlp-course-project-2024-2025-teammlg/code/sling_setup.sh
```

Then navigate into <code>code</code> directory and run:

```bash
sbatch ./sling-run.sh
```

to run the evaluation (you have to be in `code` directory for relative imports to work).


## Chatbot usage
Run the following command to create an interactive session:
```bash
srun --job-name "chatbot testing" --cpus-per-task 4 --mem-per-cpu 1500 --time 30:00 --gres=gpu:2 --partition=gpu --pty bash
```

Then run the following (model options are: deepseek_baseline, deepseek_naive, qwen_baseline, qwen_naive):
```bash
singularity exec --nv ../../containers/nlp-v1.sif python ./conversation_evaluation.py --model qwen_naive
```

For instructions on running the advanced RAG systems and different options, see [Codebase](#codebase).

Shard loading can take up to 30 minutes. The <code>></code> symbol indicates that the system is waiting for your query. Response generation typically takes around 30 seconds. To terminate the current session type <code>quit</code>.

If you get SSL-related errors, run:

```bash
unset SSL_CERT_FILE
```


## Codebase
The codebase diverged slightly during the development; therefore, the RAG systems are split between `code` and `code_v2`.

**`code`** - contains the baseline and naive RAG systems, the evaluation of those systems, and additional code for processing results.

**`code_v2`** - contains the baseline and advanced RAG systems, their partial evaluation, and code for results analysis.

To run interactive conversation on either baseline or naive RAG system, move into `code` and run:

```bash
python ./conversation_evaluation.py --model qwen_naive
```

To run interactive conversation on either baseline or advanced RAG system, move into `code_v2` and run:

```bash
python ./main.py --rag_type {baseline,advanced} --model {qwen,deepseek} --operation converse --output_directory <optional, a string> --uses_memory
```
- (the `{}` signify that you should choose one of the options)

- (the flag `--uses_memory` is optional and should be used if running a conversation with the chatbot)

### Functionality inside `code`
Brief description of the main functionality of each script:
- <code>evaluation.py</code>: simultaneously evaluate multiple models on a list of questions, either <code>data/evaluation_questions.txt</code> (no ground truth) or <code>data/evaluation_questions.json</code> (ground truth).
- <code>converstaion_evaluation.py</code>: evaluate one model at a time by having a real-time conversation with it.
- <code>scraper.py</code>: obtain structured data from websites TMDB, Letterboxd, JustWatch.
- <code>scraper_v2.py</code>: modified scraper, such that Qwen model can choose which functions to call based on user query.
- <code>POStagger.py</code>: finds titles and names of interest in the user's prompt
- <code>rag.py</code>: inserts scraped data (based on POS tagger) into the model's context.
- <code>summarizer.py</code>: summarizes scraped data for advanced RAG.
- <code>memory.py</code>: buffer for recent queries and replies for conversational models.
- <code>metrics.py</code>: calculates evaluation metrics based on the model's output and ground truth.