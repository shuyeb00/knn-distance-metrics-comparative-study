# When does the distance metric matter for k-nearest-neighbour classification?

This repository contains the code, configurations and results of a study on how the choice of distance metric affects k-nearest-neighbour (k-NN) classification, and which measurable properties of a data representation decide whether it matters. The study covers tabular benchmarks (OpenML-CC18), pretrained image and text embeddings, and controlled synthetic data.

**Status:** work in progress. The pre-registered analysis plan is in [`docs/analysis_plan.md`](docs/analysis_plan.md). The earlier version of the study (git tag `v1.0`) is archived in [`legacy/v1/`](legacy/v1/).

## Installation

Python 3.11 to 3.14.

```bash
python -m venv venv
./venv/Scripts/python -m pip install -e ".[dev]"   # Windows; use ./venv/bin/python on Linux and macOS
```

The package versions used for every result are pinned in `pyproject.toml` and recorded per run.

## Repository layout

```text
src/knnmetrics/   library code
scripts/          numbered entry points for each stage of the study
configs/          benchmark and experiment configurations
tests/            unit tests
docs/             analysis plan and data cards
results/          result tables and run manifests
legacy/v1/        archived v1 study
```

## License

MIT. See [`LICENSE`](LICENSE). Citation metadata is in [`CITATION.cff`](CITATION.cff).
