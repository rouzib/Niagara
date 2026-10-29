"""Vector figures with short titles and captions outside the artwork."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save(fig, path):
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figures(observation, mean, residual, draws, names, truth, out):
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    low, high = np.percentile(observation, [1, 99])
    for ax, image, title in zip(
        axes, (observation, mean, residual), ("Observation", "Posterior mean", "Residuals")
    ):
        im = ax.imshow(
            image,
            origin="lower",
            cmap="RdBu_r" if title == "Residuals" else "magma",
            vmin=-4 if title == "Residuals" else low,
            vmax=4 if title == "Residuals" else high,
        )
        ax.set_title(title)
        ax.set_xlabel("Pixel")
        fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, out / "images")
    flat = draws.reshape(-1, len(names))
    fig, axes = plt.subplots(7, 3, figsize=(10, 15), constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        ax.hist(flat[:, i], bins=45, density=True, color="#36759a", alpha=0.8)
        ax.axvline(truth[i], color="#ae4937", linewidth=1)
        ax.set_title(names[i])
    save(fig, out / "marginals")
    quantiles = np.quantile(flat, [0.025, 0.5, 0.975], axis=0)
    # Standardize all physical intervals by their prior-free posterior spread,
    # and label the reference truth explicitly in the caption.
    scale = flat.std(0)
    fig, ax = plt.subplots(figsize=(7, 8), constrained_layout=True)
    y = np.arange(len(names))
    center = (quantiles[1] - truth) / scale
    ax.errorbar(
        center,
        y,
        xerr=np.vstack((quantiles[1] - quantiles[0], quantiles[2] - quantiles[1])) / scale,
        fmt="o",
        markersize=3,
        color="#36759a",
    )
    ax.axvline(0, color=".4", linewidth=1)
    ax.set_yticks(y, names)
    ax.set_xlabel("Offset from truth / posterior standard deviation")
    ax.set_title("Parameter intervals")
    save(fig, out / "intervals")
    (out / "captions.txt").write_text(
        "Images: posterior predictive mean from 128 draws; residuals are divided by the fixed "
        "noise standard deviation. Marginals: red lines show synthetic truth. Intervals: "
        "median and central 95 percent posterior intervals, expressed relative to truth in "
        "posterior standard deviations. Parameter uncertainty excludes measurement noise. "
        "Consult result.json for convergence diagnostics before interpreting posterior estimates.\n"
    )
