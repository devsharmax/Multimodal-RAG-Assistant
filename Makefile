PY := .venv/bin/python

.PHONY: setup api ui run eval dataset test

setup:            ## create venv + install deps (uses uv if present)
	uv venv --python 3.12 .venv && uv pip install --python $(PY) -r requirements.txt

api:              ## start the FastAPI backend on :8000
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

ui:               ## start the Streamlit UI on :8501
	$(PY) -m streamlit run ui/app.py --server.port 8501

run:              ## backend + UI together
	$(PY) -m uvicorn api.main:app --port 8000 & sleep 2; $(PY) -m streamlit run ui/app.py; kill %1

dataset:          ## regenerate the evaluation PDFs + questions
	$(PY) eval/build_dataset.py

eval: dataset     ## full evaluation (retrieval ablations + end-to-end answers)
	$(PY) -m eval.run_eval

test:
	$(PY) -m pytest -q tests
