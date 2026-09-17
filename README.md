# Impact of Distance Metrics on KNN Classification: A Preprocessing- and Dimensionality-Sensitive Comparative Study

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.13.15](https://img.shields.io/badge/python-3.13.15-blue.svg)](https://www.python.org/)
[![scikit-learn: 1.6.1](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)

Reproducibility repository for the research study:
> **“Impact of Distance Metrics on KNN Classification: A Preprocessing- and Dimensionality-Sensitive Comparative Study”**

This repository contains the complete experimental workflow, empirical benchmarks, statistical tests, and result datasets comparing classical distance metrics across low- to high-dimensional regimes.

---

## 📌 Overview & Key Research Contributions

While K-Nearest Neighbors (KNN) classification depends fundamentally on metric geometry, published rankings are often confounded by unexamined preprocessing choices and dimensional scaling. This study investigates whether metric rankings reflect intrinsic metric geometry or empirical artifacts across five benchmark datasets ($D = 4$ to $4,096$).

### Key Contributions & Findings:
1. **Preprocessing Sensitivity:** Evaluates metrics across **Raw (unscaled)**, **Z-Score standardization**, and **Min-Max scaling**. The winning metric changes in **8 of 10** dataset–weighting pairs, proving that low-dimensional rankings are preprocessing-conditioned rather than universal.
2. **Curse of Dimensionality & Class Collapse:** Highlights the structural failure of Chebyshev distance at high dimensions ($D \ge 1,850$). On the imbalanced LFW dataset, Chebyshev collapses toward the majority class (e.g., 52.5% of *Ariel Sharon* instances misclassified as *George W. Bush*), while Cosine achieves the smallest accuracy–macro-F1 gap.
3. **High-Dimensional Remedy (PCA):** Demonstrates that training-fold PCA dramatically recovers Chebyshev on Olivetti Faces ($37.6\% \to 71.5\%$ uniform; $41.9\% \to 81.4\%$ distance-weighted) and elevates Manhattan beyond all unreduced KNN configurations.
4. **Hardware & Latency Profiling:** Evaluates inference wall-clock latency under a randomized execution order across 400 configurations. Identifies an order-of-magnitude speedup for Euclidean and Cosine over Manhattan and Chebyshev at $D \ge 1,850$. A controlled implementation comparison demonstrates a genuine implementation-path factor ($2.0\times$–$3.0\times$), while thread-count sweeps contextualize BLAS behavior.
5. **Rigorous Statistical Testing:** Applies multi-seed Friedman omnibus tests and Nemenyi post-hoc analysis across both **uniform** and **distance-weighted** voting schemes.
6. **Baseline Comparisons:** Benchmarks KNN against untuned Logistic Regression and RBF-kernel SVM classifiers.

---

## 🔬 Experimental Scope

### 1. Evaluated Distance Metrics
* **Euclidean ($L_2$ norm):** $d(x, y) = \sqrt{\sum_{i=1}^D (x_i - y_i)^2}$
* **Manhattan ($L_1$ norm):** $d(x, y) = \sum_{i=1}^D |x_i - y_i|$
* **Chebyshev ($L_\infty$ norm):** $d(x, y) = \max_i |x_i - y_i|$
* **Cosine Distance:** $d(x, y) = 1 - \frac{x \cdot y}{\|x\|_2 \|y\|_2}$

### 2. Benchmark Datasets
Spanning low-, moderate-, and high-dimensional regimes:

| Dataset | Samples ($N$) | Features ($D$) | Classes ($C$) | Domain / Nature |
| :--- | :---: | :---: | :---: | :--- |
| **Iris** | 150 | 4 | 3 | Botany / Morphological |
| **Wine** | 178 | 13 | 3 | Analytical Chemistry |
| **Breast Cancer Wisconsin** | 569 | 30 | 2 | Oncology / Histopathology |
| **LFW People** | 1,288 | 1,850 | 7 | Biometrics (Unconstrained, Skewed) |
| **Olivetti Faces** | 400 | 4,096 | 40 | Biometrics (Controlled, Balanced) |

---

## 📊 Summary of Primary Results ($K=5$, Min-Max Scaled, 10 Seeds)

| Dataset | Metric | Mean Acc. (U) % | Macro-F1 (U) % | Mean Acc. (D) % | Macro-F1 (D) % | Cohen's $\kappa$ (U) | Cohen's $\kappa$ (D) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iris** ($D=4$) | Euclidean | **96.00** | 95.99 | **96.00** | 95.99 | 0.940 | 0.940 |
| | Manhattan | 95.00 | 94.99 | 95.00 | 94.99 | 0.925 | 0.925 |
| | Chebyshev | 94.33 | 94.32 | 95.00 | 94.98 | 0.915 | 0.925 |
| | Cosine | 81.67 | 81.15 | 80.33 | 79.79 | 0.725 | 0.705 |
| **Wine** ($D=13$) | Euclidean | 96.39 | 96.43 | 96.94 | 96.96 | 0.946 | 0.954 |
| | Manhattan | **97.22** | **97.27** | **97.22** | **97.27** | 0.958 | 0.958 |
| | Chebyshev | 95.28 | 95.37 | 95.56 | 95.61 | 0.929 | 0.933 |
| | Cosine | 94.44 | 94.69 | 96.39 | 96.53 | 0.917 | 0.946 |
| **Breast Cancer** ($D=30$) | Euclidean | **97.46** | **97.25** | **97.46** | **97.25** | 0.945 | 0.945 |
| | Manhattan | 97.28 | 97.05 | 97.28 | 97.05 | 0.941 | 0.941 |
| | Chebyshev | 95.35 | 95.01 | 95.35 | 95.01 | 0.900 | 0.900 |
| | Cosine | 92.98 | 92.57 | 92.89 | 92.47 | 0.851 | 0.850 |
| **LFW People** ($D=1,850$) | Euclidean | 58.84 | 44.28 | 59.38 | 46.15 | 0.431 | 0.440 |
| | Manhattan | 61.78 | 46.75 | 64.53 | 52.42 | 0.468 | 0.507 |
| | Chebyshev | 37.21 | 19.17 | 36.86 | 19.33 | 0.110 | 0.107 |
| | Cosine | **62.67** | **50.05** | **64.69** | **52.93** | **0.480** | **0.510** |
| **Olivetti Faces** ($D=4,096$) | Euclidean | 85.00 | 83.90 | 89.50 | 88.48 | 0.846 | 0.892 |
| | Manhattan | **85.50** | **84.43** | **90.75** | **89.93** | **0.851** | **0.905** |
| | Chebyshev | 37.62 | 32.76 | 41.88 | 37.28 | 0.360 | 0.404 |
| | Cosine | 80.00 | 78.77 | 87.75 | 87.01 | 0.795 | 0.874 |

*(U = Uniform Voting, D = Distance-Weighted Voting)*

---

## 📁 Repository Structure

```text
├── data/
│   └── README.md                         # Dataset retrieval and caching instructions
├── notebooks/
│   └── Impact_of_distance_matrics_on_KNN_classification_sourceCode.ipynb
├── requirements.txt                      # Pinned Python package dependencies
├── LICENSE                               # MIT License
└── README.md                             # Project documentation
```

### Generated Experimental Artifacts:
Executing the notebook produces all empirical artifacts referenced in the paper:
* `high_dim_computational_benchmarks.csv`: Master run log containing accuracy, F1, and Cohen's $\kappa$.
* `preprocessing_sensitivity_results.csv`: Multi-seed evaluations across Raw, Z-Score, and Min-Max scaling.
* `lfw_perclass_f1_results.csv`: Per-identity F1 scores and class-level diagnostic tables.
* `pca_remedy_results.csv`: PCA dimensionality reduction evaluations (50, 100, and 95% variance).
* `k_sensitivity_results.csv` & `k_sensitivity_diagnostic.csv`: Sensitivity sweeps across $K \in \{1, 3, 5, 7, 9, 15\}$.
* `latency_remeasured_results.csv`: Latency profiles measured under randomized configuration orders.
* `latency_uncertainty_comparison.csv`: Within-seed IQR vs. across-seed SD uncertainty metrics.
* `blas_thread_sweep_results.csv`: Thread-limit scaling benchmarks (1 vs. 2 OpenBLAS/OpenMP threads).
* `blas_control_condition_results.csv`: Implementation-path control (`scikit-learn` predict vs. `scipy.spatial.distance.cdist`).
* `baseline_results.csv`: Logistic Regression and RBF-kernel SVM comparative baselines.
* `dimensional_stress_benchmarks.png`: Accuracy distributions and log-scale latency scaling figures.

---

## 🚀 Setup & Reproducibility

### Pinned Runtime Environment (Environment B)
The experimental protocol was conducted on an x86_64 Linux environment (Intel Xeon CPU @ 2.20GHz, OpenBLAS/OpenMP threading backend):
* **Python:** `3.13.15`
* **scikit-learn:** `1.6.1`
* **NumPy:** `2.1.3`
* **Pandas:** `2.2.3`
* **SciPy:** `1.16.3`
* **scikit-posthocs:** `0.17.0`
* **threadpoolctl:** `3.6.0`

### Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/shuyeb00/knn-distance-metrics-comparative-study.git
cd knn-distance-metrics-comparative-study
pip install -r requirements.txt
```

### Executing the Study
Launch Jupyter Notebook or Google Colab and run the cells sequentially:

```bash
jupyter notebook notebooks/Impact_of_distance_matrics_on_KNN_classification_sourceCode.ipynb
```
All datasets are automatically downloaded via `scikit-learn` with zero manual configuration required.

---

## 📖 Citation

If you use this repository, the experimental methodology, or the benchmark results in your research, please cite:

```bibtex
@article{numan2026impact,
  title={Impact of Distance Metrics on KNN Classification: A Preprocessing- and Dimensionality-Sensitive Comparative Study},
  author={Numan, Shohid Ahmed and others},
  journal={AIUB Journal of Science and Engineering (AJSE)},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). Datasets fetched programmatically are subject to their respective original licenses.
