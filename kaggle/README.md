# Kaggle runs

Each job is a single notebook cell. Notebook settings: accelerator **GPU T4 x2**, **Internet on**, no secrets. Run with *Save Version → Save & Run All (Commit)* so the job continues in the background. Outputs are written to `/kaggle/working/features` and published as a private Kaggle Dataset shared with the co-authors.

The cell clones the repository, records the exact commit in the output, installs the package without touching Kaggle's preinstalled PyTorch, and runs one extraction script. Re-running a job skips feature files that already exist.

## Image features, part 1 (CIFAR-10, CIFAR-100, STL-10)

```python
!git clone -q https://github.com/shuyeb00/knn-distance-metrics-comparative-study.git /kaggle/working/repo
!mkdir -p /kaggle/working/features && git -C /kaggle/working/repo rev-parse HEAD | tee /kaggle/working/features/source_commit.txt
!pip install -q --no-deps -e /kaggle/working/repo && pip install -q --no-deps facenet-pytorch && pip install -q open_clip_torch ftfy
!cd /kaggle/working/repo && python scripts/02_extract_image_embeddings.py --out /kaggle/working/features --datasets cifar10 cifar100 stl10 --workers 4
!rm -rf /kaggle/working/repo && du -sh /kaggle/working/features && cat /kaggle/working/features/image_sanity_knn.json
```

## Image features, part 2 (Oxford-IIIT Pets, Flowers-102, DTD, EuroSAT, LFW, Olivetti)

Same cell with `--datasets pets flowers102 dtd eurosat lfw olivetti`.

## Text features (AG News, DBpedia-14, Banking77, 20 Newsgroups)

```python
!git clone -q https://github.com/shuyeb00/knn-distance-metrics-comparative-study.git /kaggle/working/repo
!mkdir -p /kaggle/working/features && git -C /kaggle/working/repo rev-parse HEAD | tee /kaggle/working/features/source_commit.txt
!pip install -q --no-deps -e /kaggle/working/repo
!cd /kaggle/working/repo && python scripts/03_extract_text_embeddings.py --out /kaggle/working/features --batch-size 256
!rm -rf /kaggle/working/repo && du -sh /kaggle/working/features && head -40 /kaggle/working/features/feature_manifest.csv
```

## Tabular main grid

Settings: **None** (CPU, 4 cores, 31 GB) for tasks with D ≤ 1000 except the four image-like tasks, **GPU T4 x2** for CIFAR_10, Devnagari-Script, mnist_784 and Fashion-MNIST. Kaggle's Python 3.13 image has no `ensurepip`, so the pinned packages go to `/tmp/pydeps` and Kaggle's own PyTorch is kept.

```python
!git clone -q https://github.com/shuyeb00/knn-distance-metrics-comparative-study.git /kaggle/working/repo
!pip install -q --target /tmp/pydeps numpy==2.4.6 scipy==1.17.1 scikit-learn==1.9.1 pandas==3.1.1 pyarrow==26.0.1 openml==0.15.1 threadpoolctl==3.8.0 pyyaml==6.0.4
!pip install -q --no-deps --target /tmp/pydeps /kaggle/working/repo
!nproc; free -g | head -2; find /kaggle/input -name "repeat*_fold*_*.parquet" | wc -l
!cd /kaggle/working/repo && PYTHONPATH=/tmp/pydeps python scripts/04_run_main_grid.py --tasks <TASK IDS> --jobs 4 --chunk-mb 256 --max-hours 11 --prepared /tmp/prepared --openml-cache /tmp/openml --out /kaggle/working/grid --seed-from /kaggle/input
!git -C /kaggle/working/repo rev-parse HEAD > /kaggle/working/grid/source_commit.txt; rm -rf /kaggle/working/repo; du -sh /kaggle/working/grid; find /kaggle/working/grid -name "*.parquet" | wc -l
```

For the GPU tasks replace `--jobs 4 --chunk-mb 256` with `--backend torch --devices cuda:0 cuda:1 --chunk-mb 1024`.

- Kaggle stops a cell after 12 hours. `--max-hours` only stops new units from starting, so set it to 12 h minus the longest unit time minus about 20 minutes of preparation (CIFAR_10 units take about 33 min on one T4: use `--max-hours 9.5`).
- To resume a stopped job, create a private dataset from the stopped notebook's output, add it as input to a new notebook and run the same cell. `--seed-from /kaggle/input` copies the finished unit files, so only missing units run. The `find ... | wc -l` line must show the number of finished units before the run starts.
- Unit files carry the configuration hash; all runs must show `config f41275325d95`.

## Certified search check (GPU T4)

Inputs: the image and text feature datasets (and the corrected CLIP root, if available). The script compares the certified float32/float64 backend with the float64 engine on synthetic and real features and measures throughput. Every comparison must report `identical=True` before the embedding grid uses `--backend torch_certified`.

```python
import glob, os
!git clone -q https://github.com/shuyeb00/knn-distance-metrics-comparative-study.git /kaggle/working/repo
!pip install -q --target /tmp/pydeps numpy==2.4.6 scipy==1.17.1 scikit-learn==1.9.1 pandas==3.1.1 pyarrow==26.0.1 openml==0.15.1 threadpoolctl==3.8.0 pyyaml==6.0.4 pytest==9.2.0
!pip install -q --no-deps --target /tmp/pydeps /kaggle/working/repo
manifests = sorted(glob.glob("/kaggle/input/**/feature_manifest.csv", recursive=True))
clip_roots = " ".join(os.path.dirname(m) for m in manifests if "features_clip" in m or "images-2" in m)
print(*manifests, sep="\n"); print("clip roots:", clip_roots)
!cd /kaggle/working/repo && PYTHONPATH=/tmp/pydeps python -m pytest -q -p no:cacheprovider tests/test_certified_search.py 2>&1 | tail -3
!cd /kaggle/working/repo && PYTHONPATH=/tmp/pydeps python scripts/04c_check_certified_search.py --device cuda:0 --features /kaggle/input --clip-features {clip_roots} --out /kaggle/working/certified_search_check_t4.json
```

CLIP features are read only from `--clip-features`: the CLIP folders of image part 1 were extracted with the wrong activation and must not be used.
