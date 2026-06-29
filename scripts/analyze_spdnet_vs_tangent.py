"""
SPDNet vs Tangent Space + LDA: Comprehensive comparison.

Generates:
  1. Per-subject accuracy scatter + correlation
  2. t-SNE of LogEig features vs Tangent Space features
  3. Covariance response analysis per class
  4. Subject "difficulty" ranking across methods

Usage:
    python scripts/analyze_spdnet_vs_tangent.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.spd_covariance import compute_covariance
from features.riemann import HAS_PYRIEMANN
from preprocessing.alignment import EuclideanAlignment

import torch
from models.spd_models import create_spdnet


# ---------------------------------------------------------------------------
# Load per-subject data
# ---------------------------------------------------------------------------

def load_subjects(data_dir: str, n: int) -> list[dict]:
    subs = []
    for i in range(1, n + 1):
        d = Path(data_dir) / f"subj_{i:02d}"
        X = np.load(d / "X.npy").astype(np.float32)
        y = np.load(d / "y.npy").astype(np.int64)
        subs.append({"id": i, "X": X, "y": y})
    return subs


# ---------------------------------------------------------------------------
# Extract SPDNet features (LogEig output) for all subjects
# ---------------------------------------------------------------------------

def extract_spdnet_features(subjects, bimap_dims=(8, 8), device="cpu"):
    """LOSO-style SPDNet feature extraction. Returns per-subject features."""
    model = create_spdnet(n_channels=8, n_classes=2, bimap_dims=list(bimap_dims))
    model.to(device)
    model.eval()

    per_subject = []
    for test_idx, test_subj in enumerate(subjects):
        train_subjs = [s for s in subjects if s["id"] != test_subj["id"]]

        # EA per fold
        ea = EuclideanAlignment()
        ea.fit([s["X"] for s in train_subjs])
        X_train = np.concatenate([ea.transform(s["X"]) for s in train_subjs], axis=0)
        y_train = np.concatenate([s["y"] for s in train_subjs])
        X_test = ea.transform(test_subj["X"])
        y_test = test_subj["y"]

        C_train = compute_covariance(X_train, reg=1e-4)
        C_test = compute_covariance(X_test, reg=1e-4)

        # Train SPDNet
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, TensorDataset

        n_classes = len(np.unique(y_train))
        model2 = create_spdnet(n_channels=8, n_classes=n_classes, bimap_dims=list(bimap_dims))
        model2.to(device)

        # Simple train loop
        opt = torch.optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        C_t = torch.from_numpy(C_train).float()
        y_t = torch.from_numpy(y_train).long()
        ds = TensorDataset(C_t, y_t)
        dl = DataLoader(ds, batch_size=64, shuffle=True)

        for ep in range(60):
            model2.train()
            for Xb, yb in dl:
                Xb, yb = Xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model2(Xb), yb)
                loss.backward()
                opt.step()

        # Extract features
        model2.eval()
        with torch.no_grad():
            C_test_t = torch.from_numpy(C_test).float().to(device)
            feats_spd = model2.spd_blocks(C_test_t)
            feats_log = model2.log_eig(feats_spd)
            d = feats_log.shape[-1]
            idx = model2._get_triu_idx(d, device)
            features = feats_log[:, idx[0], idx[1]].cpu().numpy()
            logits = model2(C_test_t).cpu().numpy()
            preds = logits.argmax(-1)

        acc = (preds == y_test).mean()
        per_subject.append({
            "id": test_subj["id"],
            "features": features,  # (N, feat_dim) LogEig features
            "labels": y_test,
            "preds": preds,
            "spdnet_acc": acc,
        })
        print(f"  S{test_idx+1:02d} acc={acc:.3f}", end="\r")

    print()
    return per_subject


# ---------------------------------------------------------------------------
# Extract Tangent Space features
# ---------------------------------------------------------------------------

def extract_tangent_features(subjects):
    """LOSO-style Tangent Space feature extraction."""
    if not HAS_PYRIEMANN:
        raise ImportError("pyriemann required")
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    per_subject = []
    for test_idx, test_subj in enumerate(subjects):
        train_subjs = [s for s in subjects if s["id"] != test_subj["id"]]

        ea = EuclideanAlignment()
        ea.fit([s["X"] for s in train_subjs])
        X_train = np.concatenate([ea.transform(s["X"]) for s in train_subjs], axis=0)
        y_train = np.concatenate([s["y"] for s in train_subjs])
        X_test = ea.transform(test_subj["X"])
        y_test = test_subj["y"]

        cov_est = Covariances(estimator="scm")
        C_train = cov_est.fit_transform(X_train)
        C_test = cov_est.fit_transform(X_test)

        ts = TangentSpace(metric="riemann")
        feats_train = ts.fit_transform(C_train)
        feats_test = ts.transform(C_test)

        lda = LinearDiscriminantAnalysis()
        lda.fit(feats_train, y_train)
        preds = lda.predict(feats_test)
        acc = (preds == y_test).mean()

        per_subject.append({
            "id": test_subj["id"],
            "features": feats_test,
            "labels": y_test,
            "preds": preds,
            "tangent_acc": acc,
        })
        print(f"  S{test_idx+1:02d} acc={acc:.3f}", end="\r")

    print()
    return per_subject


# ---------------------------------------------------------------------------
# Analysis 1: Per-subject accuracy correlation
# ---------------------------------------------------------------------------

def analyze_accuracy_correlation(spd_results, tangent_results):
    """Correlate per-subject accuracy between SPDNet and Tangent Space."""
    spd_accs = np.array([r["spdnet_acc"] for r in spd_results])
    tan_accs = np.array([r["tangent_acc"] for r in tangent_results])
    corr = np.corrcoef(spd_accs, tan_accs)[0, 1]

    print("\n" + "=" * 60)
    print("Analysis 1: Per-subject Accuracy Correlation")
    print("=" * 60)
    print(f"{'Subject':<10} {'SPDNet':>8} {'Tangent':>8} {'Delta':>8}")
    print("-" * 40)
    for i in range(len(spd_accs)):
        delta = spd_accs[i] - tan_accs[i]
        marker = " *" if abs(delta) > 0.1 else ""
        print(f"S{spd_results[i]['id']:02d}       {spd_accs[i]:8.3f} {tan_accs[i]:8.3f} {delta:+8.3f}{marker}")

    print("-" * 40)
    print(f"Mean:     {spd_accs.mean():8.3f} {tan_accs.mean():8.3f}")
    print(f"Std:      {spd_accs.std():8.3f} {tan_accs.std():8.3f}")
    print(f"Pearson r: {corr:.3f}")

    # Subjects where SPDNet wins/loses
    wins = (spd_accs > tan_accs).sum()
    losses = (spd_accs < tan_accs).sum()
    print(f"SPDNet wins on {wins}/30 subjects, Tangent wins on {losses}/30")

    return corr


# ---------------------------------------------------------------------------
# Analysis 2: t-SNE visualization
# ---------------------------------------------------------------------------

def analyze_tsne(spd_results, tangent_results):
    """t-SNE projection of SPDNet and Tangent Space features."""
    # Collect a sample of features for t-SNE (all subjects, class-balanced)
    all_spd_feats = []
    all_tan_feats = []
    all_labels = []

    for spd_r, tan_r in zip(spd_results, tangent_results):
        # Sample up to 10 per class per subject to keep t-SNE manageable
        for c in np.unique(spd_r["labels"]):
            c_idx = np.where(spd_r["labels"] == c)[0]
            n_sample = min(10, len(c_idx))
            sampled = np.random.choice(c_idx, size=n_sample, replace=False)
            all_spd_feats.append(spd_r["features"][sampled])
            all_tan_feats.append(tan_r["features"][sampled])
            all_labels.extend([c] * n_sample)

    X_spd = np.concatenate(all_spd_feats, axis=0)
    X_tan = np.concatenate(all_tan_feats, axis=0)
    labels = np.array(all_labels)

    print(f"\nAnalysis 2: t-SNE ({X_spd.shape[0]} samples)")
    print(f"  SPDNet feature dim: {X_spd.shape[1]}")
    print(f"  Tangent feature dim: {X_tan.shape[1]}")

    # t-SNE on SPDNet features
    tsne_spd = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_spd)
    tsne_tan = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_tan)

    # Compute class separability (silhouette-like: ratio of between-class to within-class scatter)
    def class_separation(X_2d, labels):
        classes = np.unique(labels)
        centroids = np.array([X_2d[labels == c].mean(0) for c in classes])
        overall_mean = X_2d.mean(0)
        between = np.sum([(c - overall_mean) ** 2 for c in centroids]) * len(X_2d) / len(classes)
        within = np.sum([np.sum((X_2d[labels == c] - centroids[i]) ** 2) for i, c in enumerate(classes)]) / len(X_2d)
        return between / (within + 1e-10)

    sep_spd = class_separation(tsne_spd, labels)
    sep_tan = class_separation(tsne_tan, labels)

    print(f"  SPDNet t-SNE class separation: {sep_spd:.4f}")
    print(f"  Tangent t-SNE class separation: {sep_tan:.4f}")

    # Correlation between feature spaces
    # (Use mean per-class features per subject)
    return {"tsne_spd": tsne_spd, "tsne_tan": tsne_tan, "labels": labels}


# ---------------------------------------------------------------------------
# Analysis 3: Covariance response patterns
# ---------------------------------------------------------------------------

def analyze_covariance_patterns(subjects):
    """Analyze what SPD patterns correspond to which class."""
    ea = EuclideanAlignment()
    ea.fit([s["X"] for s in subjects])
    X_all = np.concatenate([ea.transform(s["X"]) for s in subjects], axis=0)
    y_all = np.concatenate([s["y"] for s in subjects])

    C_all = compute_covariance(X_all, reg=1e-4)

    # Per-class mean covariance
    classes = np.unique(y_all)
    print(f"\nAnalysis 3: Covariance Response Patterns")
    print("=" * 60)

    for c in classes:
        C_c = C_all[y_all == c]
        C_mean = C_c.mean(0)
        # Diagonal (channel power)
        diag = np.diag(C_mean)
        top_ch = np.argsort(diag)[-3:][::-1]
        ch_names = ["FC3", "C3", "Cz", "C4", "FC4", "CP3", "CPz", "CP4"]
        cname = ["Rest", "Left", "Right"][c]
        print(f"  Class {c} ({cname}): n={len(C_c)}")
        print(f"    Top channels: {[ch_names[i] for i in top_ch]} "
              f"powers={[f'{diag[i]:.4f}' for i in top_ch]}")
        # Off-diagonal (inter-channel correlation)
        C_corr = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                C_corr[i, j] = C_mean[i, j] / np.sqrt(C_mean[i, i] * C_mean[j, j])
        top_pairs = []
        for i in range(8):
            for j in range(i + 1, 8):
                top_pairs.append((i, j, C_corr[i, j]))
        top_pairs.sort(key=lambda x: -abs(x[2]))
        print(f"    Top correlations: "
              f"{ch_names[top_pairs[0][0]]}-{ch_names[top_pairs[0][1]]}={top_pairs[0][2]:.3f}, "
              f"{ch_names[top_pairs[1][0]]}-{ch_names[top_pairs[1][1]]}={top_pairs[1][2]:.3f}")

    return C_all, y_all, classes


# ---------------------------------------------------------------------------
# Analysis 4: Subject difficulty ranking
# ---------------------------------------------------------------------------

def analyze_subject_difficulty(spd_results, tangent_results):
    """Rank subjects by difficulty and check cross-method agreement."""
    spd_accs = np.array([r["spdnet_acc"] for r in spd_results])
    tan_accs = np.array([r["tangent_acc"] for r in tangent_results])

    # Average accuracy across methods
    avg_accs = (spd_accs + tan_accs) / 2
    ranking = np.argsort(avg_accs)  # easiest to hardest

    print(f"\nAnalysis 4: Subject Difficulty Ranking")
    print("=" * 60)
    print(f"{'Rank':<6} {'Subj':<6} {'SPDNet':>8} {'Tangent':>8} {'Avg':>8}")
    print("-" * 45)
    for rank, idx in enumerate(ranking):
        print(f"{rank+1:<6} S{spd_results[idx]['id']:02d}   "
              f"{spd_accs[idx]:8.3f} {tan_accs[idx]:8.3f} {avg_accs[idx]:8.3f}")

    # Hardest and easiest subjects
    hardest = ranking[:5]
    easiest = ranking[-5:]
    hard_ids = [spd_results[i]["id"] for i in hardest]
    easy_ids = [spd_results[i]["id"] for i in easiest]
    print(f"  Hardest subjects:  {[f'S{sid:02d}' for sid in hard_ids]}")
    print(f"  Easiest subjects:  {[f'S{sid:02d}' for sid in easy_ids]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading subjects...")
    subjects = load_subjects("data/loso_binary", 30)
    print(f"Loaded {len(subjects)} subjects")

    # Extract SPDNet features (LOSO)
    print("\nExtracting SPDNet features (30-fold LOSO)...")
    spd_results = extract_spdnet_features(subjects)

    # Extract Tangent Space features (LOSO)
    print("\nExtracting Tangent Space features (30-fold LOSO)...")
    tangent_results = extract_tangent_features(subjects)

    # Analysis 1: Accuracy correlation
    corr = analyze_accuracy_correlation(spd_results, tangent_results)

    # Analysis 2: t-SNE
    tsne_data = analyze_tsne(spd_results, tangent_results)

    # Analysis 3: Covariance patterns
    cov_data = analyze_covariance_patterns(subjects)

    # Analysis 4: Subject difficulty
    analyze_subject_difficulty(spd_results, tangent_results)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    spd_accs = [r["spdnet_acc"] for r in spd_results]
    tan_accs = [r["tangent_acc"] for r in tangent_results]
    print(f"SPDNet mean:   {np.mean(spd_accs):.4f} ± {np.std(spd_accs):.4f}")
    print(f"Tangent mean:  {np.mean(tan_accs):.4f} ± {np.std(tan_accs):.4f}")
    print(f"Correlation:   {corr:.3f}")
    print(f"SPDNet > Tangent on {sum(np.array(spd_accs) > np.array(tan_accs))}/30 subjects")


if __name__ == "__main__":
    main()
