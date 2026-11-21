# Analysis Plan: v2 Study

**Working title.** When does the distance metric matter for k-nearest-neighbour classification? Preprocessing, norm information, ties and hubness across tabular data and pretrained embeddings.

**Status.** Draft. The plan is frozen with the git tag `analysis-plan-v2` before the main experimental grid is run. Any change after that tag is recorded in Section 12 with a date and a reason.

---

## 1. Motivation and scope

The v1 study (tag `v1.0`) compared four metrics on five scikit-learn datasets. Its most robust results were conditional rather than universal: the winning metric changed with preprocessing in 8 of 10 dataset-weighting pairs, Chebyshev never ranked first, and on raw face pixels Chebyshev drifted toward the majority class. These results point to a better question than "which metric wins". The question for v2 is which measurable properties of a representation decide whether the metric matters, in which direction, and by how much.

v2 answers it with four components:

1. short formal results that make each dependence explicit (Section 3);
2. controlled interventions on real and synthetic data that test those results causally rather than by correlation (Section 6);
3. a benchmark large enough that datasets, not random splits, are the unit of statistical inference (Section 4);
4. the same analysis on pretrained image and text embeddings, where cosine similarity is the unexamined default (Section 4.2, 4.3).

v2 does not propose a new distance. Its contribution is explanatory and predictive.

**Prior work this study is positioned against.** Large metric leaderboards (Abu Alfeilat et al., 2019; Hu et al., 2016); normalization studies (Singh & Singh, 2020; Abdalla & Altaf, 2023); distance concentration and fractional norms (Beyer et al., 1999; Aggarwal et al., 2001; François et al., 2007; Mirkes et al., 2020); hubness and hubness reduction (Radovanović et al., 2010; Schnitzer et al., 2012; Feldbauer & Flexer, 2019); k-NN evaluation of self-supervised features (Wu et al., 2018; Caron et al., 2021); and distance-measure recommendation by meta-learning in clustering. Every reference is checked against its DOI before it is cited.

---

## 2. Research questions

- **RQ1 (Magnitude).** Across a curated benchmark, how much classification performance is lost by a default metric relative to the per-dataset oracle and to inner-CV selection, and how does the metric effect compare with the effects of preprocessing, K and vote weighting?
- **RQ2 (Mechanism).** Which training-set properties (norm informativeness, feature scale heterogeneity, quantization and ties, hubness, intrinsic dimension, relative contrast) explain when and why metric rankings differ?
- **RQ3 (Prediction).** Can those properties, computed on training data only, choose a metric and preprocessing for an unseen dataset with low regret, and at what cost relative to inner-CV selection?
- **RQ4 (Transfer).** Do the answers change when the same data are represented by pretrained embeddings instead of raw features?
- **RQ5 (Cost).** How do exact-search inference costs differ across metrics, implementations, thread pools and hardware, and what explains the differences?

---

## 3. Formal results and the hypotheses they imply

Each result below is stated and proved in the manuscript. Each implies a prediction that is tested in Sections 6 and 9.

**P1 (Feature scaling is a choice of metric).** Let T(x) = Ax + b with A = diag(a), a_i > 0. For p in (0, ∞), d_p(Tx, Ty) = (Σ_i a_i^p |x_i − y_i|^p)^{1/p}, and d_∞(Tx, Ty) = max_i a_i |x_i − y_i|. The translation b cancels, so scaling followed by an L_p metric equals a weighted L_p metric on the original data. Min-max and z-score scaling therefore give the same k-NN classifier under any L_p metric whenever range_i / σ_i is constant over features. Cosine distance is not translation invariant, so for cosine the two scalings also differ through the centering shift.

> **H1.** Neighbour-set disagreement between min-max and z-score under L_p grows with scale-ratio heterogeneity, SRH = SD_i(log(range_i / σ_i)). At matched SRH, the disagreement is larger for cosine than for any L_p metric.

**P2 (Cosine and Euclidean coincide on a sphere).** If ‖x‖ = r for every point, then d_2(x, y)^2 = 2r^2(1 − cos(x, y)), a strictly increasing function of cosine distance. Uniform-vote k-NN with L2 and with cosine distance then produce identical neighbour sets, ties included, for every K. With distance-weighted votes they differ only through the weight function. Consequently, on unnormalized data every uniform-vote accuracy difference between L2 and cosine is caused by variation in vector norms.

> **H2.** The difference Acc(L2) − Acc(cos) increases with norm informativeness (NI), defined as the training-fold cross-validated balanced-accuracy gain of a classifier on ‖x‖ alone over the chance rate. Destroying norm information by random per-row rescaling leaves cosine unchanged by construction and degrades L2 at a rate that increases with NI.

**P3 (Correlation distance is cosine after row centering).** d_corr(x, y) = d_cos(x − x̄1, y − ȳ1). It is invariant to per-sample affine intensity changes x ↦ αx + β1 with α > 0. Cosine is invariant to α only, and L2 to neither.

> **H3.** On raw pixel representations with per-image brightness and contrast variation, the ordering correlation ≥ cosine ≥ L2 holds. Injecting a random per-image affine intensity change leaves correlation unaffected, affects cosine only through β, and affects L2 through both.

**P4 (Chebyshev ties on quantized features).** If every feature takes values in a common set of q levels (for example 8-bit pixels, q = 256), the L∞ distance between a query and the training set takes at most q distinct values, independent of D. Under L1 and L2 the number of attainable values grows with D. Let the boundary tie mass be τ_k = |{j : d_j = d_(k)}| / k. When τ_k is large, the neighbourhood is decided by the tie-breaking rule rather than by distance. Under any label-independent tie-breaking rule, the vote then tends to the class composition of the tie set, which reverts toward the class prior.

> **H4.** On raw quantized pixels, Chebyshev's τ_k is much greater than 1 and predicts its accuracy loss and its majority-class drift. Under controlled quantization (q from 2 to 256, and continuous via sub-level jitter), majority drift tracks τ_k. Per-feature scaling changes the level sets per feature, so τ_k is measured after every preprocessing rather than assumed.

**P5 (Concentration).** Not re-proved. The asymptotic results of Beyer et al. (1999) and Aggarwal et al. (2001) motivate measuring relative contrast for each metric.

> **H5.** Following Mirkes et al. (2020), relative contrast does not predict per-metric accuracy across datasets once log N, log D and the number of classes are controlled. Hubness (k-occurrence skewness) and the ratio of intrinsic to ambient dimension do. The hypothesis is directional, and either outcome is reported.

**H6 (Embeddings).** For L2-normalized embeddings, L2 and cosine are equivalent by P2, so the choices that matter are centering, whitening, hubness reduction and the order p. Centering followed by L2 normalization, and the hubness-reduction methods, give the largest gains where hubness skewness is highest.

**H0 (Variance share).** On tabular data, preprocessing explains a larger share of performance variance than the metric does. The metric matters more on raw pixels than on pretrained embeddings.

---

## 4. Benchmark

### 4.1 Tabular

- **Source.** OpenML-CC18 (Bischl et al., 2021), 72 classification tasks.
- **Primary set.** Tasks whose features are all numeric. The expected size is about 35 to 45 tasks, and the final list is fixed before the grid is run.
- **Secondary set.** The remaining tasks, with one-hot encoded categoricals and train-fold median imputation. They are used only as a sensitivity analysis.
- **Size cap.** Tasks with more than 20,000 rows are subsampled to 20,000 with a fixed seed, stratified by class. The subsample indices are stored.
- **Duplicates.** Exact duplicate rows are counted, together with their cross-fold leakage. The main analysis uses deduplicated data, dropping every copy of a duplicate with conflicting labels. The original data are analysed as a sensitivity check.

### 4.2 Images

- **Datasets.** CIFAR-10, CIFAR-100, STL-10, Oxford-IIIT Pets, Flowers-102, DTD, EuroSAT, LFW (at least 70 images per person) and Olivetti Faces.
- **Representations.**
  - raw pixels at a fixed resolution;
  - ResNet-50 with ImageNet supervision;
  - DINOv2 ViT-S/14 and ViT-B/14;
  - CLIP ViT-B/32;
  - for the two face datasets, a face-recognition backbone (FaceNet, InceptionResnetV1 trained on VGGFace2).
- **Splits.** Official train/test splits wherever they exist.
- **Integrity.** Array checksums and library versions are recorded for every extracted feature file.

### 4.3 Text

- **Datasets.** AG News, 20 Newsgroups, DBpedia-14 (subsampled) and Banking77.
- **Representations.** TF-IDF (optional, sparse), all-MiniLM-L6-v2, all-mpnet-base-v2 and e5-base-v2.

### 4.4 Synthetic

Class-conditional Gaussian mixtures in intrinsic dimension d, linearly embedded in ambient dimension D with additive noise. The controlled factors are d, D, norm-label coupling, per-feature scale heterogeneity, quantization level q and class imbalance. The design is a fractional factorial with 10 replicates per cell.

---

## 5. Experimental factors

| Factor | Levels |
| :--- | :--- |
| Metric | L_0.5 (quasi-metric), L1, L2, L3, L∞, cosine, correlation, Mahalanobis (Ledoit–Wolf shrinkage, fitted on the training fold) |
| Secondary distances (on L2 and cosine) | mutual proximity (Gaussian), local scaling (NICDM), CSLS |
| Preprocessing, tabular | none, min-max, z-score, robust (median/IQR), quantile-to-normal |
| Preprocessing, embeddings | none, L2 normalization, centering + L2, z-score, PCA whitening |
| K | 1, 3, 5, 7, 11, 15, 21, 31 |
| Vote weighting | uniform, inverse distance, and for embeddings exp(similarity / 0.07) |
| Tie policy | stable training order (scikit-learn behaviour, default); random seeded tie-break and include-all-ties vote as diagnostics |
| Reference learners | NCA (tasks with N ≤ 5,000 and D ≤ 200), logistic regression, RBF-SVM, histogram gradient boosting, each with the same inner-CV budget as k-NN |

Neighbour search is exact brute force throughout. Every preprocessing step and every data-dependent metric (Mahalanobis, the hubness-reduction statistics) is fitted on the training part of each fold only.

---

## 6. Interventions

| ID | Intervention | Tests |
| :--- | :--- | :--- |
| I1 | Random per-row rescaling, s ~ LogUniform(1/c, c), with c on a grid | P2, H2 |
| I2 | Class-dependent norm injection | P2, H2 |
| I3 | Random per-feature scaling of raw data, applied before each preprocessing | P1, H1 |
| I4 | Random per-image affine intensity change (brightness and contrast) | P3, H3 |
| I5 | Quantization of features to q ∈ {2, 4, 8, 16, 64, 256} levels, and continuous via jitter | P4, H4 |
| I6 | Appending irrelevant Gaussian noise dimensions | P5, H5 |
| I7 | Controlled class-imbalance resampling, crossed with I5 | P4, H4 |

Each intervention is applied only to the representation, never to the labels. Its strength is varied on a grid, and the predicted direction is stated before the run.

---

## 7. Evaluation protocol

- **Tabular.** 3 × 5 repeated stratified cross-validation (15 outer folds). Two regimes are reported:
  - *grid*: every factor combination, for the variance analysis;
  - *tuned*: K and vote weighting chosen by inner stratified 5-fold CV on each outer training fold. In the inner-CV selection baseline, metric and preprocessing are chosen the same way.
- **Images and text.** The official test split, with tuning by 5-fold CV on the training split. Uncertainty comes from 2,000 test-set bootstrap resamples.
- **Scores.** Balanced accuracy is the primary score. Accuracy, macro-F1 and Cohen's κ are secondary. Per-class F1 and confusion matrices are kept for the imbalanced face and text tasks.

---

## 8. Diagnostics

All diagnostics are computed on training folds only:

- N, D, number of classes C and imbalance ratio;
- intrinsic dimension (TwoNN; Levina–Bickel MLE averaged over k = 10 to 20) and its ratio to D;
- relative contrast for each metric;
- hubness for each metric: k-occurrence skewness at k = 10, the largest k-occurrence, and the antihub rate;
- norm informativeness NI and the coefficient of variation of the norms;
- scale-ratio heterogeneity SRH;
- feature discreteness, as the median number of distinct values per feature divided by N;
- boundary tie mass τ_k for each metric;
- sparsity.

---

## 9. Statistical analysis

- **Unit of inference.** For every cross-dataset claim the unit is the dataset. Folds and bootstrap resamples are used only for within-dataset uncertainty.
- **Omnibus tests.** The Friedman test with the Iman–Davenport correction, over the metric levels within each preprocessing level and over the joint metric × preprocessing cells.
- **Pairwise tests.** Wilcoxon signed-rank tests with Holm correction, following Benavoli et al. (2016) rather than mean-rank post-hoc tests. The effect size is the Hodges–Lehmann median difference with a 95% bootstrap CI. Critical-difference diagrams use Wilcoxon–Holm cliques.
- **Bayesian comparison.** The Bayesian signed-rank test (Benavoli et al., 2017) with a ROPE of ±1 percentage point of balanced accuracy. P(left), P(ROPE) and P(right) are reported.
- **Within-dataset tests.** The corrected repeated k-fold t-test (Nadeau & Bengio, 2003; Bouckaert & Frank, 2004), with Holm correction within each dataset.
- **Variance decomposition.** A linear mixed model, score ~ metric × preprocessing + K + weighting + (1 | dataset), with partial η² for each term. A nonparametric check reports the per-dataset spread attributable to each factor.
- **Ranking stability.** Kendall's τ between metric rankings across preprocessings, and the winner-flip rate with a dataset-bootstrap CI.
- **Diagnostics and outcomes.** The pre-specified pairs are:
  - NI with L2 − cos (H2);
  - SRH with the min-max/z-score disagreement (H1);
  - τ_k with the L∞ loss (H4);
  - hubness, the ID/D ratio and relative contrast with per-metric loss (H5).

  They are tested by Spearman partial correlation controlling for log N, log D and C, with Holm correction across these pairs. Every other association is labelled exploratory.
- **Meta-prediction (RQ3).** Leave-one-dataset-out. The candidate selectors are:
  - a rule derived from P1 to P4;
  - an L1-regularized logistic model;
  - a shallow random forest.

  The target is the best (metric, preprocessing) cell. Regret is the oracle score minus the score of the chosen cell. The comparators are fixed defaults (L2 with z-score; cosine with L2 normalization), inner-CV selection together with its compute cost, and random choice. Significance comes from a permutation test that shuffles the meta-targets.

---

## 10. Cost study

- **Workload.** A controlled synthetic sweep at N_train = 20,000 and N_query = 2,000, with D = 2^2 to 2^12.
- **Backends.**
  - scikit-learn brute force (`algorithm="brute"` set explicitly);
  - SciPy `cdist` with `argpartition`;
  - a NumPy GEMM formulation;
  - PyTorch on CPU and on CUDA (T4);
  - FAISS `IndexFlat` for L2 and inner product.
- **Threads.** 1, 2, 4, 8 and 16, with the BLAS and OpenMP pools limited separately through `threadpoolctl`.
- **Hardware.** AMD Ryzen 7 laptop (exact model, power plan and AC state recorded), Kaggle CPU and Kaggle T4.
- **Protocol.** Randomized interleaved order, warm-up calls, at least 30 timed repetitions, and medians with IQR plus the across-block SD. Analytic FLOP counts give throughput in queries per second and GFLOP/s.
- **Hypothesis.** The L2 and cosine advantage comes from the GEMM formulation and scales with the BLAS pool, while L1 and L∞ scale with the OpenMP pool. Limiting each pool separately tests this.

---

## 11. Decision and reporting rules

- Every result is reported, including null and contrary results.
- Datasets are excluded only by the rules in Section 4, and each exclusion is logged with its reason.
- Primary claims are limited to H0 to H6 and RQ1 to RQ5. Everything else is labelled exploratory.
- Every number in the manuscript is generated from the result files by script. No number is typed by hand.
- Code, configuration files, result tables and extracted-feature checksums are archived with a DOI at submission.

---

## 12. Change log

| Date | Change | Reason |
| :--- | :--- | :--- |
| 2026-11-21 | Draft created | Scope of v2 defined after the v1 review history |
