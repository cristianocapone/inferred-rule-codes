PYTHON ?= python3

.PHONY: all figures tables pdf verify
all figures tables pdf verify:
	$(PYTHON) scripts/reproduce.py $@
