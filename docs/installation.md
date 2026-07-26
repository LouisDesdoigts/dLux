# Installation

## Install dLux

dLux requires Python 3.10 or newer and is available from PyPI:

```bash
python -m pip install dLux
```

JAX selects its compute backend independently. Follow the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) if
you want GPU or TPU acceleration.

## Development installation

Clone the repository and install the development dependencies in editable mode:

```bash
git clone https://github.com/LouisDesdoigts/dLux.git
cd dLux
python -m pip install -e ".[dev]"
pre-commit install
```

Run the tests and build the documentation before submitting changes:

```bash
pytest
zensical build --strict
```

## Tutorial notebooks

The runnable notebooks and their dependencies live in the separate
[dLux tutorials repository](https://github.com/LouisDesdoigts/dLux_tutorials).
Install that project in its own environment:

```bash
git clone https://github.com/LouisDesdoigts/dLux_tutorials.git
cd dLux_tutorials
python -m pip install -e .
jupyter lab
```

The editable install is useful when developing notebooks and dLux together:
install your local dLux checkout into the same environment with
`python -m pip install -e /path/to/dLux`.
