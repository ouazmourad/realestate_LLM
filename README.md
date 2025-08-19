# CrewAI + Ollama + DuckDuckGo (no API keys)

Local agents using CrewAI + a custom DuckDuckGo tool and an Ollama model.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

# start Ollama in another terminal
ollama serve
ollama pull openhermes

python flow.py
