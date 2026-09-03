.PHONY: all test model-selection backtest site clean

# `make all` must reproduce every number in reports/ from a cold checkout:
# install deps, run the leakage-gated test suite, pick a model on validation
# only (Phase 3), then run the one frozen test evaluation (Phase 4) and
# regenerate the static site's data files.
all: test model-selection backtest

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

# Phase 3: validation-only comparison. Writes reports/chosen_model.json.
model-selection:
	python -m fplxp.model_selection

# Phase 4: the one frozen test-set run. Requires reports/chosen_model.json.
backtest:
	python -m fplxp.backtest

clean:
	rm -rf data/raw reports/backtest.md reports/chosen_model.json reports/validation_comparison.md
	rm -rf reports/*.png site/data/*.json
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
