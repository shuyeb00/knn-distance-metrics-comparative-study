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
