PYTHON ?= python3
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
PIP := $(VENV_PYTHON) -m pip

.DEFAULT_GOAL := help
.PHONY: help venv install data image train export export-model pipeline clean-artifacts

help:
	@printf '%s\n' \
		'Commands:' \
		'  make install       Create .venv and install Python dependencies.' \
		'  make data          Download the CIFAR-10 train and test datasets.' \
		'  make image         Export the configured CIFAR-10 sample as .hex, .coe, and .S.' \
		'  make train         Train the configured INT4 QAT model and export layer-1 verification files.' \
		'  make export       Export layer-1 weights and verification outputs as .hex, .coe, and .S.' \
		'  make export-model Alias for make export.' \
		'  make pipeline      Run install, data, image, and train in order.' \
		'  make clean-artifacts  Remove all generated hardware artifacts.'

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)

venv: $(VENV_PYTHON)

$(VENV)/.installed: $(VENV_PYTHON) requirements.txt
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $@

install: $(VENV)/.installed

data: install
	$(VENV_PYTHON) -m scripts.download_cifar10 --config config.yaml

image: data
	$(VENV_PYTHON) -m scripts.export_cifar10_image --config config.yaml

train: data
	$(VENV_PYTHON) main.py --train --quant-bits INT4
	$(VENV_PYTHON) -m scripts.export_hardware_bundle --config config.yaml --quant-bits INT4

export: data
	$(VENV_PYTHON) main.py --quant-bits INT4
	$(VENV_PYTHON) -m scripts.export_hardware_bundle --config config.yaml --quant-bits INT4

export-model: export

pipeline: image train

clean-artifacts:
	rm -rf artifacts
