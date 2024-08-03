PROJECT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
SHELL:=/bin/bash
VENV=.venv
PYTHON_VERSION=3.10
PYTHON=$(VENV)/bin/python3
PIP=$(VENV)/bin/pip
ACT="./bin/act"

install:
	@if ! [ -d $(VENV) ]; then\
        python${PYTHON_VERSION} -m venv $(VENV);\
		$(PIP) install --upgrade pip;\
	fi
	$(PIP) install ${PROJECT_DIR};\


install-all: install
	$(PIP) install --editable "${PROJECT_DIR}[dev,book]"

install-dev: install-all
	pre-commit install
	curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

install-automl: install-all
	$(PIP) install autogluon[common,feature_generators,tabular]==0.8.2

update:
	$(PIP) install --upgrade ${PROJECT_DIR}

test-cicd-lint-local:
	${ACT} -j markdown-lint -W ${PROJECT_DIR}/.github/workflows/doc.yml

test-cicd-code-local:
	${ACT} -j python-lint -W ${PROJECT_DIR}/.github/workflows/code.yml

test-code:
	$(PYTHON) -m pycodestyle ${PROJECT_DIR}/src
	$(PYTHON) -m pylint ${PROJECT_DIR}/src

test-lint: test-cicd-lint-local

clean:
	rm -rf __pycache__
	rm -rf $(VENV)

.PHONY: run clean
