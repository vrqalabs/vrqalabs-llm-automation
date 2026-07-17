# LLM Automation — Python (FastAPI)

This repository is a **sample LLM automation and benchmarking framework** built
with **Python** and **FastAPI** for **VRQA Labs**, an AI and automation
consultancy. It showcases how we structure API integrations, evaluation
pipelines, testing, and reporting, while demonstrating a realistic workflow for
comparing multiple Large Language Models (LLMs).

> 🎯 If you're a potential client, think of this as a proof-of-concept. We can
> extend the same architecture to support additional LLM providers, enterprise
> AI applications, prompt evaluation pipelines, and fully automated CI/CD
> workflows for Generative AI projects.

---

## 1. Prerequisites

Before running anything, make sure you have the following installed on your
machine:

* Python 3.11 or newer
* Git (for cloning the repository)
* Optional: Docker Desktop for containerized execution
* Optional: API keys for OpenRouter, Groq, or Hugging Face
* Optional: Ollama if you'd like to benchmark local LLMs

The project uses FastAPI and open-source Python libraries, making it easy to run
locally or deploy to cloud environments.

## 2. Local setup

```bash
# clone the repo
git clone https://github.com/vrqalabs/vrqalabs-llm-automation.git
cd vrqalabs-llm-automation

# create a virtual environment
python -m venv venv

# activate the environment (Windows)
venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# start the FastAPI server
uvicorn app.main:app --reload
```

Once running, the interactive API documentation is available at:

```text
http://localhost:8000/docs
```

Environment variables are used to configure API keys, database connections, and
supported LLM providers.

### OpenRouter configuration

1. Copy the example environment file:

```bash
copy .env.example .env
```

2. Open [.env](.env) and set your real OpenRouter key:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

3. Restart the app so the new settings are loaded.

Note: Never commit your local `.env` file to version control — it may contain
secrets. The repository includes `.env` in `.gitignore` and provides
`.env.example` as a safe template instead.

> If the key is missing, the app will return a clear error instead of sending an invalid request.

## 3. Project structure

```text
.
├── app/
│   ├── api/               # FastAPI routes
│   ├── core/              # configuration and settings
│   ├── schemas/           # request/response pydantic schemas
│   ├── services/          # LLM provider integrations and adapters
│   ├── utils/             # shared helper functions
│   └── main.py            # application entry point
├── dashboard/             # optional Streamlit dashboard
├── docs/                  # project documentation
├── notebooks/             # exploratory notebooks
├── scripts/               # helper scripts (run/stop server)
├── tests/                 # unit and integration tests
├── requirements.txt
└── README.md
```

The framework follows **Clean Architecture** principles and uses abstraction
layers so new LLM providers can be added with minimal changes to the rest of the
application.

## 4. Application & benchmark information

The example application demonstrates how to evaluate and compare responses from
multiple LLM providers using a common interface.

The repository will include benchmark scenarios such as:

1. Sending the same prompt to multiple LLM providers.
2. Measuring response latency and estimated token usage.
3. Comparing generated responses using automated evaluation metrics.
4. Recording benchmark results for later analysis and visualization.

Although the initial implementation focuses on a small benchmark workflow, the
framework is designed to scale. VRQA Labs can extend it with additional model
providers, prompt libraries, evaluation datasets, experiment tracking, and
enterprise reporting dashboards.

## 5. Why VRQA Labs?

This sample repository reflects how VRQA Labs approaches AI engineering
projects:

* **Modular architecture** separating API layers, providers, evaluation logic, and reporting.
* **Provider flexibility** supporting cloud and local LLMs through a common interface.
* **Scalable design** that makes adding new models and benchmark suites straightforward.
* **Production-ready practices** including testing, Docker, CI/CD, and automated reporting.

If your team is building Generative AI applications, internal AI platforms, or
needs a reliable framework for evaluating Large Language Models, VRQA Labs
offers end-to-end implementation, integration, and automation services.

Contact us at **[connect.vrqalabs@outlook.com](mailto:connect.vrqalabs@outlook.com)** to discuss your AI automation and
LLM engineering requirements.
