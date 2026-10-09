# AI Interview Coach

A local-first interview practice project built with **Phi-3 + Ollama, LangChain, LangGraph, Streamlit, DDGS, and SQLite**. It does not require a paid LLM API key.

## 1. Requirements

- Python 3.12 recommended
- Ollama installed and running
- `phi3:latest` available in Ollama
- Internet only if you choose optional web search

Confirm your model:

```bash
ollama list
```

If needed, download it once:

```bash
ollama pull phi3:latest
```

## 2. Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Ollama is not already running in the background, open another terminal and run:

```bash
ollama serve
```

## 3. Start the app

```bash
source .venv/bin/activate
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## 4. Features

- Dashboard with metrics computed from the SQLite database
- Topic/difficulty selection and sequential interview questions
- Local Phi-3 evaluation through LangChain
- LangGraph workflow: optional evidence retrieval → answer evaluation → weighted scoring
- Correctness, completeness, clarity and structure scores
- Strengths, points to improve, reference answer and improved answer
- Optional free DDGS web search; snippets cached in `data/processed/search_cache.json`
- SQLite interview history in `data/interview_history.db`
- Practice Hub and detailed history view

## 5. Project tree

```text
ai-interview-coach/
├── app.py
├── interview_agent.py
├── search_service.py
├── database.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   │   └── question_bank.json
│   └── processed/
│       └── .gitkeep
├── docs/
│   ├── project_scope.md
│   └── evaluation_rubric.md
└── tests/
    └── test_scoring.py
```

Runtime database and search cache files are created on first use.

## 6. Test the deterministic scoring formula

```bash
python -m unittest discover -s tests -v
```

## 7. How the workflow works

1. The UI selects questions from the question-only JSON bank.
2. Candidate submits an answer.
3. LangGraph optionally retrieves cached/free web snippets.
4. LangChain calls the local Phi-3 model to produce evaluation JSON.
5. Python validates category scores and calculates the weighted total.
6. SQLite saves the question, answer, scores, feedback and sources.
7. Dashboard/history queries show saved session metrics.

## Notes and limitations

- First-time search can take longer; repeat searches use the local cache. Uncheck web search for faster practice.
- Phi-3 can still make mistakes or return inconsistent evaluations. Treat feedback as a learning aid, not an authoritative hiring result.
- Search snippets can be incomplete or low-quality; review sources for important technical claims.
- No fine-tuning is included in this phase. First establish a functioning app and a reviewed evaluation dataset.
