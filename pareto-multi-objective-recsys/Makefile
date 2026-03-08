PY=python

.PHONY: inspect prepare-data eda train predict baselines pareto evaluate full-report

inspect:
	$(PY) scripts/inspect_dataset.py --out-root .

prepare-data:
	$(PY) scripts/run_preprocess.py --out-root .

eda:
	$(PY) scripts/run_eda.py --out-root .

train:
	$(PY) scripts/run_train.py --config configs/model.yaml --out-root .

predict:
	$(PY) scripts/run_predict.py --model artifacts/models/final_model.pt --out-root .

baselines:
	$(PY) scripts/run_baselines.py --predictions reports/predictions/test_predictions.csv --out-root .

pareto:
	$(PY) scripts/run_pareto_rerank.py --predictions reports/predictions/test_predictions.csv --out-root .

evaluate:
	$(PY) scripts/evaluate_model.py --predictions reports/predictions/test_predictions.csv --out-root .

full-report:
	$(PY) scripts/generate_final_analysis.py --out-root .

report-assets:
	$(PY) scripts/generate_report_assets.py --out-root .
PYTHON := python

.PHONY: install test lint format clean

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

format:
	@echo "Run your formatter (black/isort) as desired."

clean:
	@echo "Cleaning artifacts..."
	rmdir /s /q artifacts || true
