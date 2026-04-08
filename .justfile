set shell := ["bash", "-c"]

default:
    @just --list

clean:
    rm -rf dist
    rm -f .coverage*
    rm -rf htmlcov
    rm -rf .ruff_cache
    rm -rf .pytest_cache

format:
    uv run --locked ruff format

lint +targets="src tests examples":
    uv run --locked ruff check {{targets}}

pre-commit:
    uv run --locked pre-commit run --all-files

sync:
    uv sync --locked --all-groups --all-extras

test testfile:
    uv run --locked pytest -v --color=yes --no-cov {{testfile}}

test-examples:
    uv run --locked pytest -n auto -vv --no-cov --durations=0 --durations-min=1 --color=yes ./tests/test_examples.py

test-all:
    uv run --locked pytest -n auto -v --deselect tests/test_examples.py --durations=0 --durations-min=0.5 --color=yes --cov-report=html
