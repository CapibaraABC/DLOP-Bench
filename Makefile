PYTHON ?= python3

lint:
		$(PYTHON) -m flake8

format:
		yapf -i -r .
