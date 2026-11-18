.PHONY: install data synthetic retail backtest-synthetic backtest-retail lint test demo clean

# Thread cap for the gradient-boosting fits (OpenMP crawls on a busy machine) and
# worker processes for the per-series Holt-Winters fits.
THREADS ?= 1
JOBS ?= 1

install:
	uv sync --locked

synthetic:
	uv run demand-forecast generate --out data/synthetic.csv

retail:
	uv run demand-forecast download --out data/online_retail.csv

data: synthetic retail

backtest-synthetic:
	uv run demand-forecast backtest --data data/synthetic.csv --name "Synthetic (5 stores x 20 items)" \
		--out reports/synthetic --gbm-threads $(THREADS) --jobs $(JOBS)

backtest-retail:
	uv run demand-forecast backtest --data data/online_retail.csv --name "UCI Online Retail II (top 100 products)" \
		--out reports/online_retail --gbm-threads $(THREADS) --jobs $(JOBS)

lint:
	uv run ruff check
	uv run ruff format --check

test:
	uv run pytest -q

demo: data backtest-synthetic backtest-retail

clean:
	rm -rf data .pytest_cache .ruff_cache
