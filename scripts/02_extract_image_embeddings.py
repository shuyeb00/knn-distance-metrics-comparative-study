import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from knnmetrics.data_embeddings import save_split, write_manifest

SEED = 20261121
COLOUR = ("cifar10", "cifar100", "stl10", "pets", "flowers102", "dtd", "eurosat")
FACES = ("lfw", "olivetti")
DATASETS = COLOUR + FACES
MODELS = ("raw", "resnet50", "dinov2_s14", "dinov2_b14", "clip_b32", "facenet")
FACE_ONLY = {"facenet"}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class ArrayImages:
    """Uniform (PIL image, label) access for in-memory uint8 arrays."""

    def __init__(self, images, labels):
        self.images, self.labels = images, np.asarray(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        from PIL import Image
        return Image.fromarray(self.images[i]), int(self.labels[i])


def _labels_of(ds):
    for attr in ("targets", "labels", "_labels"):
        if hasattr(ds, attr):
            return np.asarray(getattr(ds, attr))
    return np.array([ds[i][1] for i in range(len(ds))])


def load_image_dataset(name, root):
    """Returns {split: dataset of (PIL image, label)} using the official splits of Section 4.2."""
    import torch.utils.data as tud
    import torchvision.datasets as tvd
    r = str(root)
    if name == "cifar10":
        return {s: tvd.CIFAR10(r, train=(s == "train"), download=True) for s in ("train", "test")}
    if name == "cifar100":
        return {s: tvd.CIFAR100(r, train=(s == "train"), download=True) for s in ("train", "test")}
    if name == "stl10":
        return {s: tvd.STL10(r, split=s, download=True) for s in ("train", "test")}
    if name == "pets":
        return {"train": tvd.OxfordIIITPet(r, split="trainval", target_types="category", download=True),
                "test": tvd.OxfordIIITPet(r, split="test", target_types="category", download=True)}
    if name == "flowers102":
        parts = [tvd.Flowers102(r, split=s, download=True) for s in ("train", "val")]
        train = tud.ConcatDataset(parts)
        train.labels = np.concatenate([_labels_of(p) for p in parts])
        return {"train": train, "test": tvd.Flowers102(r, split="test", download=True)}
    if name == "dtd":
        parts = [tvd.DTD(r, split=s, partition=1, download=True) for s in ("train", "val")]
        train = tud.ConcatDataset(parts)
        train.labels = np.concatenate([_labels_of(p) for p in parts])
        return {"train": train, "test": tvd.DTD(r, split="test", partition=1, download=True)}
    if name == "eurosat":
        from sklearn.model_selection import train_test_split
        full = tvd.EuroSAT(r, download=True)
        y = _labels_of(full)
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
        tr, te = np.sort(tr), np.sort(te)
        out = {}
        for s, idx in (("train", tr), ("test", te)):
            sub = tud.Subset(full, idx)
            sub.labels = y[idx]
            sub.source_indices = idx
            out[s] = sub
        return out
    if name == "lfw":
        from sklearn.datasets import fetch_lfw_people
        b = fetch_lfw_people(min_faces_per_person=70, color=True, resize=1.0, data_home=r)
        imgs = np.clip(np.round(b.images * 255.0), 0, 255).astype(np.uint8)
        return {"all": ArrayImages(imgs, b.target)}
    if name == "olivetti":
        from sklearn.datasets import fetch_olivetti_faces
        b = fetch_olivetti_faces(data_home=r)
        imgs = np.clip(np.round(b.images * 255.0), 0, 255).astype(np.uint8)
        return {"all": ArrayImages(imgs, b.target)}
    raise ValueError(name)


class Transformed:
    def __init__(self, base, transform):
        self.base, self.transform = base, transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        return self.transform(img.convert("RGB")), int(label)


def raw_pixel_transform(face):
    import torchvision.transforms as T
    if face:
        return T.Compose([T.Grayscale(1), T.Resize((64, 64), antialias=True), T.PILToTensor()])
    return T.Compose([T.Resize((32, 32), antialias=True), T.PILToTensor()])


def build_model(name, device):
    """Returns (callable batch -> features, transform, description)."""
    import torch
    import torchvision.transforms as T
    if name == "resnet50":
        import torchvision.models as tvm
        w = tvm.ResNet50_Weights.IMAGENET1K_V2
        m = tvm.resnet50(weights=w)
        m.fc = torch.nn.Identity()
        return m.eval().to(device), w.transforms(), "torchvision resnet50 IMAGENET1K_V2, global-average-pooled 2048-d"
    if name in ("dinov2_s14", "dinov2_b14"):
        from transformers import AutoModel
        repo = {"dinov2_s14": "facebook/dinov2-small", "dinov2_b14": "facebook/dinov2-base"}[name]
        m = AutoModel.from_pretrained(repo).eval().to(device)
        tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        return (lambda x: m(pixel_values=x).pooler_output), tf, f"{repo}, layer-normalised CLS token"
    if name == "clip_b32":
        import open_clip
        m, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
        m = m.eval().to(device)
        return m.encode_image, preprocess, "open_clip ViT-B-32-quickgelu openai, image projection before L2 normalisation"
    if name == "facenet":
        from facenet_pytorch import InceptionResnetV1
        m = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        captured = {}
        # forward() L2-normalises its output; the embedding is taken before that from last_bn.
        m.last_bn.register_forward_hook(lambda mod, inp, out: captured.__setitem__("z", out))

        def embed(x):
            m(x)
            return captured["z"]
        tf = T.Compose([T.Resize((160, 160), antialias=True), T.ToTensor(),
                        T.Lambda(lambda t: (t * 255.0 - 127.5) / 128.0)])
        return embed, tf, "facenet-pytorch InceptionResnetV1 vggface2, 512-d before L2 normalisation"
    raise ValueError(name)


def run_loader(dataset, transform, batch_size, workers, fn=None, device="cpu"):
    import torch
    loader = torch.utils.data.DataLoader(Transformed(dataset, transform), batch_size=batch_size,
                                         shuffle=False, num_workers=workers,
                                         pin_memory=(device == "cuda"))
    feats, labels = [], []
    with torch.inference_mode():
        for xb, yb in loader:
            if fn is None:
                feats.append(xb.reshape(xb.shape[0], -1).numpy().astype(np.uint8))
            else:
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                    z = fn(xb.to(device, non_blocking=True))
                feats.append(z.float().cpu().numpy().astype(np.float16))
            labels.append(yb.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def cosine_knn_sanity(Xtr, ytr, Xte, yte, k=20, device="cpu"):
    import torch
    a = torch.nn.functional.normalize(torch.as_tensor(Xtr, dtype=torch.float32, device=device), dim=1)
    b = torch.nn.functional.normalize(torch.as_tensor(Xte, dtype=torch.float32, device=device), dim=1)
    ytr_t = torch.as_tensor(ytr, device=device)
    correct = 0
    for i in range(0, b.shape[0], 1024):
        idx = (b[i:i + 1024] @ a.T).topk(min(k, a.shape[0]), dim=1).indices
        votes = torch.nn.functional.one_hot(ytr_t[idx], int(ytr.max()) + 1).sum(1)
        correct += (votes.argmax(1).cpu().numpy() == yte[i:i + 1024]).sum()
    return correct / len(yte)


def main():
    p = argparse.ArgumentParser(description="Extract raw-pixel and pretrained image features.")
    p.add_argument("--out", required=True)
    p.add_argument("--data-root", default="/tmp/knnmetrics_image_data")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="debug only: first n images per split")
    args = p.parse_args()

    import torch
    import torch.utils.data as tud
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    env = {"torch": torch.__version__, "python": platform.python_version(),
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}
    sanity = {}

    for dname in args.datasets:
        t0 = time.perf_counter()
        splits = load_image_dataset(dname, Path(args.data_root))
        if args.limit:
            splits = {s: tud.Subset(d, range(min(args.limit, len(d)))) for s, d in splits.items()}
        print(f"{dname}: " + ", ".join(f"{s}={len(d)}" for s, d in splits.items())
              + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
        for s, d in splits.items():
            src = getattr(d, "source_indices", None)
            if src is not None:
                (out / dname).mkdir(parents=True, exist_ok=True)
                np.save(out / dname / f"{s}_source_indices.npy", np.asarray(src, dtype=np.int64))

        face = dname in FACES
        for mname in args.models:
            if mname in FACE_ONLY and not face:
                continue
            last = "all" if face else "test"
            if (out / dname / mname / f"{last}_meta.json").exists():
                print(f"  {mname}: already done", flush=True)
                continue
            if mname == "raw":
                fn, tf, desc = None, raw_pixel_transform(face), ("grayscale 64x64 uint8" if face else "RGB 32x32 uint8")
            else:
                fn, tf, desc = build_model(mname, device)
            feats = {}
            for s, d in splits.items():
                t1 = time.perf_counter()
                X, y = run_loader(d, tf, args.batch_size, args.workers, fn, device)
                meta = {"description": desc, "transform": repr(tf), "seconds": round(time.perf_counter() - t1, 1), **env}
                save_split(out, dname, mname, s, X, y, meta)
                feats[s] = (X, y)
                print(f"  {mname} {s}: {X.shape} {X.dtype} in {meta['seconds']}s", flush=True)
            if "train" in feats and mname != "raw":
                acc = cosine_knn_sanity(*feats["train"], *feats["test"], device=device)
                sanity[f"{dname}/{mname}"] = round(float(acc), 4)
                print(f"  {mname}: cosine 20-NN test accuracy {acc:.4f}", flush=True)
            del fn
            if device == "cuda":
                torch.cuda.empty_cache()

    n = write_manifest(out)
    prev = out / "image_sanity_knn.json"
    if prev.exists():
        sanity = {**json.loads(prev.read_text(encoding="utf-8")), **sanity}
    prev.write_text(json.dumps(sanity, indent=2) + "\n", encoding="utf-8")
    (out / "image_environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {n} feature files", flush=True)


if __name__ == "__main__":
    main()
