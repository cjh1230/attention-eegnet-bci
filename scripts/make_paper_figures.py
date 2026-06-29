"""
Generate paper figures and statistical tests for the SPDNet study.

Outputs:
  results/figures/
    fig1_main_results.png       — bar chart: all methods
    fig2_architecture_ablation.png  — bar chart: BiMap dims
    fig3_ea_gain.png            — bar chart: EA gain per method
    fig4_spdnet_vs_tangent.png  — scatter: per-subject accuracy
    fig5_fewshot.png            — line: few-shot degradation
    fig6_tsne.png               — t-SNE: SPDNet vs Tangent
    stats_tests.txt             — paired statistical tests
"""

import sys
from pathlib import Path
import numpy as np
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT = Path("results/figures")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Per-subject accuracy data (from experiments)
# ---------------------------------------------------------------------------

SPDNET_EA = np.array([
    0.733, 0.600, 0.489, 0.511, 0.467, 0.511, 0.844, 0.622, 0.578, 0.600,
    0.511, 0.644, 0.733, 0.556, 0.756, 0.711, 0.511, 0.489, 0.733, 0.533,
    0.711, 0.556, 0.600, 0.511, 0.622, 0.622, 0.578, 0.556, 0.600, 0.622,
])

SPDNET_NOEA = np.array([
    0.511, 0.511, 0.511, 0.511, 0.467, 0.533, 0.511, 0.489, 0.533, 0.533,
    0.511, 0.467, 0.511, 0.489, 0.511, 0.489, 0.511, 0.489, 0.511, 0.511,
    0.533, 0.489, 0.489, 0.489, 0.489, 0.533, 0.511, 0.489, 0.511, 0.533,
])

TANGENT_EA = np.array([
    0.578, 0.667, 0.556, 0.533, 0.533, 0.489, 0.844, 0.689, 0.556, 0.667,
    0.511, 0.644, 0.711, 0.622, 0.756, 0.756, 0.556, 0.489, 0.600, 0.467,
    0.667, 0.578, 0.644, 0.444, 0.511, 0.756, 0.622, 0.556, 0.511, 0.622,
])

SPDNET_886_EA = np.array([
    0.733, 0.533, 0.533, 0.600, 0.511, 0.511, 0.889, 0.556, 0.600, 0.622,
    0.467, 0.667, 0.711, 0.622, 0.689, 0.622, 0.533, 0.622, 0.667, 0.489,
    0.511, 0.644, 0.667, 0.533, 0.511, 0.644, 0.556, 0.533, 0.578, 0.578,
])

SPDNET_864_EA = np.array([
    0.644, 0.533, 0.511, 0.600, 0.533, 0.533, 0.800, 0.533, 0.578, 0.600,
    0.511, 0.489, 0.600, 0.667, 0.622, 0.689, 0.511, 0.511, 0.644, 0.578,
    0.467, 0.556, 0.556, 0.511, 0.533, 0.533, 0.467, 0.489, 0.711, 0.556,
])

SPDNET_SSL_CONTRASTIVE = np.array([
    0.689, 0.578, 0.467, 0.511, 0.511, 0.511, 0.822, 0.711, 0.600, 0.600,
    0.511, 0.467, 0.711, 0.689, 0.756, 0.644, 0.511, 0.578, 0.800, 0.533,
    0.467, 0.578, 0.644, 0.511, 0.511, 0.711, 0.556, 0.533, 0.778, 0.556,
])

SPDNET_SSL_MASKED = np.array([
    0.711, 0.556, 0.533, 0.511, 0.556, 0.511, 0.822, 0.489, 0.556, 0.578,
    0.467, 0.578, 0.689, 0.644, 0.489, 0.711, 0.511, 0.578, 0.778, 0.533,
    0.600, 0.533, 0.556, 0.489, 0.533, 0.711, 0.556, 0.489, 0.622, 0.533,
])

SPDNET_FEWSHOT_20 = np.array([
    0.489, 0.511, 0.489, 0.489, 0.533, 0.467, 0.511, 0.600, 0.467, 0.467,
    0.511, 0.489, 0.511, 0.422, 0.489, 0.489, 0.489, 0.489, 0.489, 0.489,
    0.467, 0.467, 0.578, 0.511, 0.489, 0.600, 0.489, 0.644, 0.467, 0.533,
])

SPDNET_FEWSHOT_10 = np.array([
    0.467, 0.489, 0.489, 0.511, 0.467, 0.533, 0.467, 0.489, 0.533, 0.533,
    0.489, 0.467, 0.511, 0.489, 0.489, 0.489, 0.511, 0.511, 0.511, 0.511,
    0.467, 0.533, 0.489, 0.511, 0.489, 0.533, 0.489, 0.489, 0.489, 0.467,
])

SPDNET_FEWSHOT_5 = np.array([
    0.511, 0.533, 0.467, 0.489, 0.533, 0.533, 0.800, 0.356, 0.356, 0.467,
    0.511, 0.467, 0.267, 0.533, 0.467, 0.467, 0.422, 0.489, 0.511, 0.511,
    0.511, 0.489, 0.511, 0.511, 0.467, 0.467, 0.511, 0.511, 0.578, 0.467,
])

# Known baselines (single value per subject not available, use summary stats)
METHODS_MAIN = {
    "EEG Conformer\n+EA":      (63.93, 9.58),
    "EEG-TCNet\n+EA":          (63.41, 10.51),
    "FBCNet\n+EA":             (61.11, 11.69),
    "SPDNet [8,8]\n+EA (ours)": (61.04, 10.42),
    "Tangent Space\n+LDA +EA": (60.44, 9.64),
    "FgMDM\n+EA":              (59.18, 8.12),
    "EEGNet\n+EA":             (58.00, 10.06),
}

EA_GAINS = {
    "SPDNet\n[8,8]":    (50.59, 1.87, 61.04, 10.42, 10.45),
    "FBCNet":            (49.70, 2.66, 61.11, 11.69, 11.41),
    "EEGNet":            (51.93, 7.20, 58.00, 10.06, 6.07),
    "EEGNet\n+Spatiotemp": (55.04, 8.55, 57.78, 8.55, 2.74),
    "Tangent\nSpace+LDA":  (60.44, 9.64, 60.44, 9.64, 0.00),
}

FEWSHOT = {
    "Full\n(1305)": 61.04,
    "20-shot\n(40)": np.mean(SPDNET_FEWSHOT_20) * 100,
    "10-shot\n(20)": np.mean(SPDNET_FEWSHOT_10) * 100,
    "5-shot\n(10)": np.mean(SPDNET_FEWSHOT_5) * 100,
}


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests():
    """Paired t-tests and Wilcoxon tests."""
    results = []
    tests = [
        ("SPDNet+EA vs Tangent+LDA", SPDNET_EA, TANGENT_EA),
        ("SPDNet+EA vs SPDNet no-EA", SPDNET_EA, SPDNET_NOEA),
        ("SPDNet [8,8] vs [8,8,8]", SPDNET_EA, SPDNET_886_EA),
        ("SPDNet [8,8] vs [8,6,4]", SPDNET_EA, SPDNET_864_EA),
        ("SPDNet+EA vs SSL Contrastive", SPDNET_EA, SPDNET_SSL_CONTRASTIVE),
        ("SPDNet+EA vs SSL Masked", SPDNET_EA, SPDNET_SSL_MASKED),
        ("SPDNet Full vs 20-shot", SPDNET_EA, SPDNET_FEWSHOT_20),
        ("SPDNet Full vs 10-shot", SPDNET_EA, SPDNET_FEWSHOT_10),
        ("SPDNet Full vs 5-shot", SPDNET_EA, SPDNET_FEWSHOT_5),
    ]

    results.append("=" * 70)
    results.append("Statistical Tests (paired, n=30 subjects)")
    results.append("=" * 70)

    for name, a, b in tests:
        t_stat, t_p = scipy_stats.ttest_rel(a, b)
        w_stat, w_p = scipy_stats.wilcoxon(a, b)
        diff_mean = np.mean(a - b) * 100
        results.append(f"\n{name}")
        results.append(f"  Mean diff: {diff_mean:+.2f} pp")
        results.append(f"  Paired t-test:  t={t_stat:.3f}, p={t_p:.4f} {'***' if t_p<0.001 else '**' if t_p<0.01 else '*' if t_p<0.05 else 'n.s.'}")
        results.append(f"  Wilcoxon:       W={w_stat:.0f}, p={w_p:.4f} {'***' if w_p<0.001 else '**' if w_p<0.01 else '*' if w_p<0.05 else 'n.s.'}")

    # Correlation SPDNet vs Tangent
    r, r_p = scipy_stats.pearsonr(SPDNET_EA, TANGENT_EA)
    results.append(f"\nSPDNet vs Tangent Space Pearson r: {r:.4f} (p={r_p:.4f})")

    # Write to file
    text = "\n".join(results)
    with open(OUTPUT / "stats_tests.txt", "w") as f:
        f.write(text)
    print(text)


# ---------------------------------------------------------------------------
# Figure 1: Main results bar chart
# ---------------------------------------------------------------------------

def fig1_main_results():
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(METHODS_MAIN.keys())
    means = np.array([v[0] for v in METHODS_MAIN.values()])
    stds = np.array([v[1] for v in METHODS_MAIN.values()])
    colors = ["#4472C4"] * 3 + ["#ED7D31"] + ["#A5A5A5"] * 3

    bars = ax.bar(names, means, yerr=stds, color=colors, capsize=4, edgecolor="white")
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, label="Chance (50%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("PhysioNet MI 8ch Binary LOSO — Main Results")
    ax.set_ylim(45, 70)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT / "fig1_main_results.png", dpi=150)
    plt.close()
    print("Saved fig1_main_results.png")


# ---------------------------------------------------------------------------
# Figure 2: Architecture ablation
# ---------------------------------------------------------------------------

def fig2_architecture():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    configs = {
        "[8,8]\n(1 layer)":    (np.mean(SPDNET_EA), np.std(SPDNET_EA)),
        "[8,8,8]\n(2 layers)": (np.mean(SPDNET_886_EA), np.std(SPDNET_886_EA)),
        "[8,10,8]\n(expand-contract)": (59.18, 8.20),
        "[8,6,4]\n(compress)":  (np.mean(SPDNET_864_EA), np.std(SPDNET_864_EA)),
    }
    names = list(configs.keys())
    means = np.array([v[0] for v in configs.values()]) * 100
    stds = np.array([v[1] for v in configs.values()]) * 100
    params = [260, 392, 422, 94]
    colors = ["#ED7D31", "#A5A5A5", "#A5A5A5", "#A5A5A5"]

    bars = ax.bar(names, means, yerr=stds, color=colors, capsize=4, edgecolor="white")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("SPDNet Architecture Ablation")
    ax.set_ylim(50, 68)
    for bar, mean, p in zip(bars, means, params):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{mean:.1f}%\n({p} params)", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT / "fig2_architecture_ablation.png", dpi=150)
    plt.close()
    print("Saved fig2_architecture_ablation.png")


# ---------------------------------------------------------------------------
# Figure 3: EA gain
# ---------------------------------------------------------------------------

def fig3_ea_gain():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(EA_GAINS.keys())
    noea = np.array([v[0] for v in EA_GAINS.values()])
    ea = np.array([v[2] for v in EA_GAINS.values()])
    gains = np.array([v[4] for v in EA_GAINS.values()])

    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width / 2, noea, width, label="No EA", color="#D9D9D9", edgecolor="white")
    bars2 = ax.bar(x + width / 2, ea, width, label="+ EA", color="#4472C4", edgecolor="white")

    for i, (b1, b2, g) in enumerate(zip(bars1, bars2, gains)):
        ax.text(i, max(b1.get_height(), b2.get_height()) + 0.8,
                f"+{g:.1f}pp", ha="center", fontsize=9, fontweight="bold", color="#C00000")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("EA Gain Analysis")
    ax.legend(fontsize=8)
    ax.set_ylim(45, 70)
    plt.tight_layout()
    fig.savefig(OUTPUT / "fig3_ea_gain.png", dpi=150)
    plt.close()
    print("Saved fig3_ea_gain.png")


# ---------------------------------------------------------------------------
# Figure 4: SPDNet vs Tangent scatter
# ---------------------------------------------------------------------------

def fig4_scatter():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(TANGENT_EA * 100, SPDNET_EA * 100, alpha=0.7, s=40, color="#4472C4")
    ax.plot([40, 90], [40, 90], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Tangent Space + LDA + EA (%)")
    ax.set_ylabel("SPDNet + EA (%)")
    ax.set_title("Per-Subject Accuracy: SPDNet vs Tangent Space")
    ax.set_xlim(40, 90)
    ax.set_ylim(40, 90)
    ax.set_aspect("equal")

    r, p = scipy_stats.pearsonr(SPDNET_EA, TANGENT_EA)
    wins_spd = np.sum(SPDNET_EA > TANGENT_EA)
    wins_tan = np.sum(TANGENT_EA > SPDNET_EA)
    ax.text(0.05, 0.95, f"r = {r:.3f} (p={'<0.001' if p<0.001 else f'{p:.3f}'})\n"
            f"SPDNet > Tangent: {wins_spd}/30\n"
            f"Tangent > SPDNet: {wins_tan}/30",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    plt.tight_layout()
    fig.savefig(OUTPUT / "fig4_spdnet_vs_tangent.png", dpi=150)
    plt.close()
    print("Saved fig4_spdnet_vs_tangent.png")


# ---------------------------------------------------------------------------
# Figure 5: Few-shot degradation
# ---------------------------------------------------------------------------

def fig5_fewshot():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    shots = [1305, 40, 20, 10]
    labels = ["Full\n(1305 trials)", "20-shot\n(40 trials)", "10-shot\n(20 trials)", "5-shot\n(10 trials)"]
    accs = [61.04, np.mean(SPDNET_FEWSHOT_20) * 100, np.mean(SPDNET_FEWSHOT_10) * 100,
            np.mean(SPDNET_FEWSHOT_5) * 100]
    stds = [np.std(SPDNET_EA) * 100, np.std(SPDNET_FEWSHOT_20) * 100,
            np.std(SPDNET_FEWSHOT_10) * 100, np.std(SPDNET_FEWSHOT_5) * 100]

    ax.errorbar(range(len(shots)), accs, yerr=stds, marker="o", linewidth=2,
                markersize=8, color="#ED7D31", capsize=6)
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_xticks(range(len(shots)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("SPDNet Few-Shot Performance Degradation")
    ax.set_ylim(45, 68)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT / "fig5_fewshot.png", dpi=150)
    plt.close()
    print("Saved fig5_fewshot.png")


# ---------------------------------------------------------------------------
# Figure 6: t-SNE comparison
# ---------------------------------------------------------------------------

def fig6_tsne():
    """Load t-SNE data from analysis script or recompute quickly."""
    # Recompute t-SNE from cached features
    subjects = []
    for i in range(1, 31):
        X = np.load(f"data/loso_binary/subj_{i:02d}/X.npy").astype(np.float32)
        y = np.load(f"data/loso_binary/subj_{i:02d}/y.npy")
        subjects.append({"id": i, "X": X, "y": y})

    from sklearn.manifold import TSNE
    from features.spd_covariance import compute_covariance
    from preprocessing.alignment import EuclideanAlignment
    from features.riemann import HAS_PYRIEMANN
    import torch
    from models.spd_models import create_spdnet

    # Collect features using a single model trained on all data
    ea = EuclideanAlignment()
    ea.fit([s["X"] for s in subjects])
    X_all = np.concatenate([ea.transform(s["X"]) for s in subjects], axis=0)
    y_all = np.concatenate([s["y"] for s in subjects])
    C_all = compute_covariance(X_all, reg=1e-4)

    # SPDNet features
    device = "cpu"
    model = create_spdnet(n_channels=8, n_classes=2, bimap_dims=[8, 8])
    model.to(device)
    # Quick train
    from torch.utils.data import DataLoader, TensorDataset
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ds = TensorDataset(torch.from_numpy(C_all).float(), torch.from_numpy(y_all).long())
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    for _ in range(30):
        for Xb, yb in dl:
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(Xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        C_t = torch.from_numpy(C_all).float()
        feats = model.spd_blocks(C_t)
        feats = model.log_eig(feats)
        idx = model._get_triu_idx(8, device)
        spd_feats = feats[:, idx[0], idx[1]].cpu().numpy()

    # Tangent Space features
    tangent_feats = None
    if HAS_PYRIEMANN:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        cov_est = Covariances(estimator="scm")
        C_raw = cov_est.fit_transform(X_all)
        ts = TangentSpace(metric="riemann")
        tangent_feats = ts.fit_transform(C_raw)

    # Subsample for t-SNE
    n_sample = 600
    idx_sample = np.random.choice(len(y_all), size=n_sample, replace=False)
    y_sample = y_all[idx_sample]

    spd_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(spd_feats[idx_sample])

    fig, axes = plt.subplots(1, 2 if tangent_feats is not None else 1, figsize=(12, 5))
    if tangent_feats is not None:
        tan_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(tangent_feats[idx_sample])
        axes = axes.flatten()
    else:
        axes = [axes]

    classes = [0, 1]
    colors = ["#4472C4", "#ED7D31"]
    labels = ["Left MI", "Right MI"]

    for c, color, label in zip(classes, colors, labels):
        mask = y_sample == c
        axes[0].scatter(spd_tsne[mask, 0], spd_tsne[mask, 1], c=color, label=label, alpha=0.5, s=10)
    axes[0].set_title("SPDNet (LogEig features)")
    axes[0].legend(fontsize=7, markerscale=3)

    if tangent_feats is not None:
        for c, color, label in zip(classes, colors, labels):
            mask = y_sample == c
            axes[1].scatter(tan_tsne[mask, 0], tan_tsne[mask, 1], c=color, label=label, alpha=0.5, s=10)
        axes[1].set_title("Tangent Space features")
        axes[1].legend(fontsize=7, markerscale=3)

    fig.suptitle("t-SNE Visualization: SPDNet vs Tangent Space Features", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUTPUT / "fig6_tsne.png", dpi=150)
    plt.close()
    print("Saved fig6_tsne.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating figures and statistical tests...\n")
    run_statistical_tests()
    fig1_main_results()
    fig2_architecture()
    fig3_ea_gain()
    fig4_scatter()
    fig5_fewshot()
    fig6_tsne()
    print(f"\nAll outputs saved to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
