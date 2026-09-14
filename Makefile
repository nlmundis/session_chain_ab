# Offline gate. No network, no writes outside the repo, no credentials.
#
#   make check      lint, type check, unit suite, then the curated mutation gate
#   make lint       ruff
#   make typecheck  mypy, strict, against Python 3.9
#   make test       unit suite only
#   make mutants    mutation gate only
#
# The tools are pinned in requirements-dev.txt: pip install -r requirements-dev.txt
# `check` is safe to run reflexively by design: it reads nothing outside this
# directory and mutates only a temp copy of it.

PYTHON ?= python3
RUFF ?= ruff
MYPY ?= mypy
MUTT_CHECK ?= mutt_check

.PHONY: check lint typecheck test mutants clean

check: lint typecheck test mutants

lint:
	@command -v $(firstword $(RUFF)) >/dev/null 2>&1 || { \
	  echo "ruff not installed: pip install -r requirements-dev.txt"; exit 1; }
	$(RUFF) check .

typecheck:
	@command -v $(firstword $(MYPY)) >/dev/null 2>&1 || { \
	  echo "mypy not installed: pip install -r requirements-dev.txt"; exit 1; }
	$(MYPY)

test:
	$(PYTHON) -B -m unittest discover -s tests -t . -v

mutants:
	@command -v $(firstword $(MUTT_CHECK)) >/dev/null 2>&1 || { \
	  echo "mutt_check not installed: pip install -r requirements-dev.txt"; exit 1; }
	$(MUTT_CHECK)

clean:
	rm -rf __pycache__ tests/__pycache__ .mypy_cache .ruff_cache
