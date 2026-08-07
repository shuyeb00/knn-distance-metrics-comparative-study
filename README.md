# Impact of Distance Metrics on KNN Classification

Reproducibility repository for the study **“Impact of Distance Metrics on KNN Classification: A Hardware-Conscious Comparative Study Across Low- to High-Dimensional Regimes.”**

This repository contains the research notebook used for the experiments, the generated result files, and documentation for reproducing the study.


## Experimental Scope

The study compares four KNN distance metrics:

- Euclidean
- Manhattan
- Chebyshev
- Cosine

The experiments evaluate these metrics across five benchmark datasets spanning low- to high-dimensional settings:

- Iris
- Wine
- Breast Cancer Wisconsin
- LFW People
- Olivetti Faces

The experimental analysis includes:

- Classification accuracy
- Macro-F1 score
- Cohen’s κ
- Uniform and distance-weighted voting
- Multiple stratified train/test seeds
- Sensitivity to the choice of K
- Friedman statistical significance testing
- Nemenyi post-hoc analysis where applicable
- CPU inference latency profiling
- Comparison with Logistic Regression and RBF-kernel SVM baselines

## Reproducing the Experiments

The complete experimental implementation is provided in:

```text
notebooks/Impact_of_distance_matrics_on_KNN_classification_sourceCode.ipynb
```

The notebook can be opened using Jupyter Notebook, JupyterLab, or Google Colab.

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Then execute the notebook cells sequentially.

The notebook downloads the required public datasets through scikit-learn and performs the experimental workflow.


## Datasets

All datasets used in the experiments are publicly available benchmark datasets accessed through scikit-learn.

The datasets are not redistributed in this repository. The notebook retrieves the required datasets programmatically.

See `data/README.md` for additional information.


## Code Availability

The source code used for the experimental analysis is provided as a Jupyter notebook in the `notebooks/` directory.

The notebook contains the complete workflow for dataset loading, preprocessing, KNN experiments, evaluation, statistical testing, computational benchmarking, K-sensitivity analysis, and baseline comparisons.

## Citation

If you use this code or the associated experimental results in academic work, please cite the corresponding paper:

> **Impact of Distance Metrics on KNN Classification: A Hardware-Conscious Comparative Study Across Low- to High-Dimensional Regimes**

## License

The code and accompanying repository materials are released under the MIT License. The datasets used by the experiments remain subject to the terms and licenses of their respective providers and distributions.
