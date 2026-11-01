PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
PYTEST ?= $(shell if [ -x ../../.venv-shared/bin/pytest ]; then echo ../../.venv-shared/bin/pytest; else command -v pytest; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)

.PHONY: all exp01 exp02 exp03 exp04 exp05 verify test lint security

all: lint test verify

exp01:
	$(PYTHON) experiments/01_lazy_tensor_5_lines.py

exp02:
	$(PYTHON) experiments/02_hunting_graph_breaks.py

exp03:
	$(PYTHON) experiments/03_three_xla_traps_fixed.py

exp04:
	XLA_FLAGS="--xla_force_host_platform_device_count=4" $(PYTHON) experiments/04_spmd_sharding_tpu.py

exp05:
	$(PYTHON) experiments/05_call_jax_from_pytorch.py

verify:
	$(PYTHON) verify_playbook.py

test:
	$(PYTEST) -v

lint:
	$(RUFF) check experiments/ tests/ verify_playbook.py

security:
	gitleaks detect --source . --no-git
