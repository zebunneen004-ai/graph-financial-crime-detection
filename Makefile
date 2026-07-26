PYTHON ?= python
CONFIG ?= config/config.yaml

.PHONY: install test run clean-final

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests -q

run: test
	$(PYTHON) src/corrected_master_pipeline.py --config $(CONFIG) --overwrite

clean-final:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) or p.mkdir(parents=True, exist_ok=True) for p in [Path('results/final'), Path('figures/final'), Path('models/final')]]"
