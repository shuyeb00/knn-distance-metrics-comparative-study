import numpy as np

WEIGHTINGS = ("uniform", "distance", "softmax")


def neighbour_weights(distances, weighting, temperature=0.07):
    if weighting == "uniform":
        return np.ones_like(distances)
    if weighting == "distance":
        # scikit-learn convention: a query with zero-distance neighbours is decided by those alone.
        zero = distances == 0
        with np.errstate(divide="ignore"):
            w = 1.0 / distances
        has_zero = zero.any(axis=1, keepdims=True)
        return np.where(has_zero, zero.astype(float), w)
    if weighting == "softmax":
        # For cosine distance d, similarity is 1 - d; weights exp(sim / T) up to a per-row constant.
        logits = -distances / temperature
        logits -= logits.max(axis=1, keepdims=True)
        return np.exp(logits)
    raise ValueError(f"unknown weighting {weighting!r}")


def class_scores(neighbour_labels, weights, n_classes):
    scores = np.zeros((neighbour_labels.shape[0], n_classes))
    rows = np.repeat(np.arange(neighbour_labels.shape[0]), neighbour_labels.shape[1])
    np.add.at(scores, (rows, neighbour_labels.ravel()), weights.ravel())
    return scores


def predict_from_neighbours(neighbour_labels, distances, K, n_classes, weighting="uniform",
                            vote_tie="lowest", seed=None, temperature=0.07):
    """Predict with the first K neighbours. Returns predictions and a flag for vote ties.

    vote_tie="lowest" picks the lowest class code among tied classes (scikit-learn's argmax);
    "random" picks uniformly among them with a seeded generator.
    """
    lab = neighbour_labels[:, :K]
    w = neighbour_weights(distances[:, :K], weighting, temperature)
    scores = class_scores(lab, w, n_classes)
    best = scores.max(axis=1, keepdims=True)
    tied = scores == best
    vote_tied = tied.sum(axis=1) > 1
    if vote_tie == "lowest":
        pred = np.argmax(tied, axis=1)
    elif vote_tie == "random":
        rng = np.random.default_rng(seed)
        noise = rng.random(scores.shape)
        pred = np.argmax(np.where(tied, noise, -1.0), axis=1)
    else:
        raise ValueError(f"unknown vote_tie {vote_tie!r}")
    return pred, vote_tied
