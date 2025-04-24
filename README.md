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