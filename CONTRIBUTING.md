# Contributing to NeuralMPCX

We welcome contributions to NeuralMPCX! This document explains how to get involved.

## Getting Started

1. Fork the repository and clone your fork:

```bash
git clone https://github.com/<your-username>/neural-mpcx.git
cd neural-mpcx
```

2. Install the development dependencies:

```bash
pip install -e ".[dev]"
pre-commit install
```

## Development Workflow

### Code Style

This project uses:

- **Black** for code formatting (line length 88)
- **Ruff** for linting
- **mypy** for type checking
- **NumPy-style docstrings** for all public APIs

Run all checks before submitting:

```bash
black src tests
ruff check --fix src tests
ruff check src tests
mypy src
pytest -q
```

Or use pre-commit to run them automatically:

```bash
pre-commit run --all-files
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `test:` adding or updating tests
- `refactor:` code changes that neither fix a bug nor add a feature

### Pull Requests

1. Create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Ensure all checks pass (`black`, `ruff`, `mypy`, `pytest`)
4. Open a pull request with a clear description of the changes

## Contributor License Agreement

External contributors may be asked to sign a Contributor License Agreement (CLA) before their contributions can be merged. This ensures that the Helmholtz-Zentrum Dresden-Rossendorf e.V. (HZDR) holds the necessary rights for redistribution.

## Reporting Issues

Open an issue at https://github.com/hzdr/neural-mpcx/issues with:

- A clear description of the problem
- Steps to reproduce (if applicable)
- Expected vs. actual behavior
- Python version and OS

## Code of Conduct

Be respectful and constructive. We are committed to providing a welcoming and inclusive environment for everyone.
