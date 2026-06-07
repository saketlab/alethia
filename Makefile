.PHONY: clean clean-build clean-pyc clean-test coverage dist docs docs-strict docs-full docs-open docs-serve notebooks help install lint format test test-all release

.DEFAULT_GOAL := help

# Detect if uv is available
UV := $(shell command -v uv 2> /dev/null)

# Use uv if available, otherwise fallback to pip
ifdef UV
	PIP := uv pip
	BUILD := uv build
	VERSION_BUMP := uv version
else
	PIP := pip
	BUILD := python -m build
	VERSION_BUMP := @echo "uv not found. Install with: pip install uv"
endif

define BROWSER_PYSCRIPT
import os, webbrowser, sys
from urllib.request import pathname2url
webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys
for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test clean-docs ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

clean-docs: ## remove documentation build artifacts
	rm -rf docs/_build/
	rm -rf docs/api/generated/

lint: ## check style with ruff and mypy
	ruff check alethia/ tests/
	mypy alethia/

lint-fix: ## fix style issues with ruff
	ruff check --fix alethia/ tests/

format: ## format code with black and ruff
	black alethia/ tests/
	ruff check --fix alethia/ tests/

test: ## run tests quickly with the default Python
	pytest

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	pytest --cov=alethia --cov-report=html --cov-report=term
	$(BROWSER) htmlcov/index.html

NOTEBOOKS := $(wildcard docs/notebooks/*.ipynb)
SPHINX_BUILD := sphinx-build -b html docs docs/_build/html

notebooks: ## execute all vignette notebooks in place (refreshes committed outputs)
	jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=900 $(NOTEBOOKS)
	@echo ""
	@echo "Executed $(words $(NOTEBOOKS)) notebook(s). Review and commit the updated outputs."

docs: ## build Sphinx HTML docs (uses committed notebook outputs; run 'make notebooks' first if they changed)
	$(SPHINX_BUILD)
	@echo ""
	@echo "Documentation built successfully!"
	@echo "Location: docs/_build/html/"
	@echo "Open docs/_build/html/index.html in your browser"

docs-strict: ## build docs treating warnings as errors (matches CI / GitHub Pages)
	$(SPHINX_BUILD) -W --keep-going

docs-full: notebooks docs ## execute notebooks then build docs (full rebuild from source)

docs-open: docs ## build docs and open in browser
	$(BROWSER) docs/_build/html/index.html

docs-serve: docs ## build docs and serve on localhost:8000
	@echo "Serving documentation at http://localhost:8000"
	@cd docs/_build/html && python -m http.server 8000

servedocs: ## compile the docs watching for changes (requires sphinx-autobuild)
	sphinx-autobuild docs docs/_build/html --watch alethia --open-browser

dist: clean ## build source and wheel package
	$(BUILD)
	@echo ""
	@echo "Distribution packages created:"
	@ls -lh dist/

install: ## install the package in editable mode
	$(PIP) install -e .

install-dev: ## install the package with dev dependencies
	$(PIP) install -e ".[dev,test,docs]"

install-full: ## install the package with all dependencies
	$(PIP) install -e ".[full,dev,test,docs]"

install-cpu: ## install the package with CPU dependencies
	$(PIP) install -e ".[cpu,dev,test,docs]"

install-gpu: ## install the package with GPU dependencies
	$(PIP) install -e ".[gpu,dev,test,docs]"

uninstall: ## uninstall the package
	$(PIP) uninstall -y alethia

sync: ## sync dependencies (uv only)
ifdef UV
	uv sync
else
	@echo "uv not found. Install with: pip install uv"
	@echo "Or use: make install-dev"
endif

lock: ## lock dependencies (uv only)
ifdef UV
	uv lock
else
	@echo "uv not found. Install with: pip install uv"
endif

check-dist: dist ## check distribution files with twine
	twine check dist/*

release: dist check-dist ## package and upload a release to PyPI
	@echo "Uploading to PyPI..."
	twine upload dist/*

release-test: dist check-dist ## package and upload a release to TestPyPI
	@echo "Uploading to TestPyPI..."
	twine upload --repository testpypi dist/*

bump-patch: ## bump patch version (0.1.0 -> 0.1.1)
ifdef UV
	$(VERSION_BUMP) --bump patch
	@echo "Version bumped to: $$(grep '^version = ' pyproject.toml | cut -d'"' -f2)"
else
	@echo "Current version: $$(grep '^version = ' pyproject.toml | cut -d'"' -f2)"
	@echo "Install uv for automatic version bumping: pip install uv"
	@echo "Or manually update version in pyproject.toml and HISTORY.rst"
endif

bump-minor: ## bump minor version (0.1.0 -> 0.2.0)
ifdef UV
	$(VERSION_BUMP) --bump minor
	@echo "Version bumped to: $$(grep '^version = ' pyproject.toml | cut -d'"' -f2)"
else
	@echo "Current version: $$(grep '^version = ' pyproject.toml | cut -d'"' -f2)"
	@echo "Install uv for automatic version bumping: pip install uv"
	@echo "Or manually update version in pyproject.toml and HISTORY.rst"
endif

bump-major: ## bump major version (0.1.0 -> 1.0.0)
ifdef UV
	$(VERSION_BUMP) --bump major
	@echo "Version bumped to: $$(grep '^version = ' pyproject.toml | cut -d'"' -f2)"
else
	@echo "Current version: $$(grep '^version = ' pyproject.toml | cut -d'"' -f2)"
	@echo "Install uv for automatic version bumping: pip install uv"
	@echo "Or manually update version in pyproject.toml and HISTORY.rst"
endif

version: ## show current version
	@echo "Current version: $$(grep '^version = ' pyproject.toml | cut -d'"' -f2)"

check: lint test ## run linting and tests

pre-commit: format lint test ## run formatting, linting, and tests before commit

show-tools: ## show which tools are being used
	@echo "Package manager: $(PIP)"
	@echo "Build tool: $(BUILD)"
ifdef UV
	@echo "uv version: $$(uv --version)"
	@echo "✓ Using uv for fast operations"
else
	@echo "⚠ uv not found - using traditional tools"
	@echo "Install uv for faster operations: pip install uv"
endif
	@echo ""
	@echo "Python: $$(python --version)"
	@echo "pip: $$(pip --version)"
