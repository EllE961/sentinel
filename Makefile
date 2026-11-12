# sentinel - common tasks. Every target is a plain python command, so they also work on Windows
# without make (copy the command). Needs compass-recsys installed editable (pip install -e ../compass-recsys).
PY ?= python

.PHONY: help eval figures test lint all

help:
	@echo "eval figures test lint"

eval:              ## fit EASE before the first cutoff, measure drift+quality on 5 real windows -> results/sweep.json
	$(PY) -m sentinel.evaluate

figures:            ## README figure -> docs/assets/
	$(PY) scripts/make_figures.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

all: eval figures
