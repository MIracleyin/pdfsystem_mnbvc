.PHONY: lint test boundary gc ci sync

sync:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest tests/ -v

boundary:
	uv run pytest tests/architecture/test_boundary.py -v

gc: boundary
	@echo "[gc] Architecture boundary check passed."

ci: lint test
