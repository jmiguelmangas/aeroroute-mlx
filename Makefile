.PHONY: bootstrap format lint typecheck test build check
bootstrap:
	uv sync --all-groups
format:
	uv run ruff format src tests
lint:
	uv run ruff check src tests
typecheck:
	@echo "Type checks are added with generated contract models."
test:
	uv run pytest
build:
	uv build
check: lint typecheck test build
