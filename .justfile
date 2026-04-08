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
    uv run ruff format

lint +targets="src tests examples":
    uv run ruff check {{targets}}

pre-commit:
    uv run pre-commit run --all-files

sync:
    uv sync --locked --all-groups --all-extras

test testfile:
    uv run pytest -v --color=yes --no-cov {{testfile}}

test-examples:
    uv run pytest -n auto -vv --no-cov --durations=0 --durations-min=1 --color=yes ./tests/test_examples.py

test-all:
    uv run pytest -n auto -v --deselect tests/test_examples.py --durations=0 --durations-min=0.5 --color=yes --cov-report=html
