.PHONY: install test lint run demo clean

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts

run:
	lstm-stock --config configs/default.yaml

demo:
	python scripts/generate_synthetic_data.py --output data/synthetic.csv
	lstm-stock --config configs/fast.yaml --csv data/synthetic.csv

clean:
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['artifacts', '.pytest_cache', '.ruff_cache']]"
