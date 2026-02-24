PYTHON ?= python

.PHONY: fmt lint

fmt:
	$(PYTHON) -m ruff check --fix *.py
	$(PYTHON) -m ruff format *.py

lint:
	$(PYTHON) -m ruff check *.py
