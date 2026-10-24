PYTHON := ./.venv/bin/python
ENV := PYTHONPATH=src MPLCONFIGDIR=/tmp/kma-matplotlib-cache XDG_CACHE_HOME=/tmp/kma-cache MLFLOW_ALLOW_FILE_STORE=true MLFLOW_DISABLE_AGENT_HINT=1

.PHONY: setup data train test serve docker site all

setup:
	$(ENV) $(PYTHON) -c "import numpy,pandas,sklearn,torch,shap,mlflow,fastapi; print('Environment ready')"

data:
	@test -f data/raw/DSL-StrongPasswordData.csv && test -f data/raw/balabit/public_labels.csv || ./scripts/download_data.sh

train:
	$(ENV) $(PYTHON) -m kma.pipeline

test:
	$(ENV) $(PYTHON) -m pytest

serve:
	$(ENV) $(PYTHON) -m uvicorn kma.api:app --host 127.0.0.1 --port 8000

docker:
	docker build -t keystroke-mouse-anomaly .

all: data train test site

site:
	$(ENV) $(PYTHON) scripts/export_site_data.py
